import argparse
import os
from pathlib import Path
from typing import Optional, cast

import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from pyopf.ext.pix4d_region_of_interest import Pix4DRegionOfInterest
from pyopf.ext.pix4d_region_of_interest import format as RoiFormat
from pyopf.io import load, save
from pyopf.pointcloud import GlTFPointCloud, Matches, Node
from pyopf.project import Calibration, ProjectObjects
from pyopf.resolve import resolve
from pyopf.util import IntType


def _get_xy_polygon(points: list[np.ndarray]) -> Polygon:
    """Transform a list of points in a 2D polygon.
    The third component of the points is ignored.

    :param points: An array of points
    :return: The corresponding shapely polygon
    """
    return Polygon([(p[0], p[1]) for p in points])


def _make_polygon(boundary: list[IntType], roi: Pix4DRegionOfInterest) -> Polygon:
    """Transforms an OPF plane boundary in a 2D polygon by projecting the points on the XY plane.

    :param boundary: An OPF plane boundary, i.e. a list of indices in the plane.vertices3d array
    :param roi: The region of interest where the boundary comes from

    :return: The corresponding shapely polygon
    """
    return _get_xy_polygon([roi.plane.vertices3d[p_idx] for p_idx in boundary])


class RoiPolygons:
    """Small wrapper class over the Pix4DRegionOfInterest to allow inside queries for a point.
    Assumes the normal is perpendicular to the plane and that the plane is parallel to the XY plane.
    From this assumption it follows that the only allowed normals are (0,0,1) and (0,0,-1).
    """

    roi: Pix4DRegionOfInterest
    outer_boundary: Polygon
    inner_boundaries: list[Polygon]
    height: Optional[float]

    def __init__(self, roi: Pix4DRegionOfInterest, matrix: Optional[np.ndarray] = None):
        """Construct a RoiPolygons wrapper for a region of interest.
        :raise ValueError: If the plane is not parallel to the XY plane.
        :raise ValueError: If the plane normal is not (0,0,1) or (0,0,-1).
        """

        if len(set([p[2] for p in roi.plane.vertices3d])) != 1:
            raise ValueError("The plane is not parallel to the XY plane")

        if not np.array_equal(
            roi.plane.normal_vector, [0, 0, 1]
        ) and not np.array_equal(roi.plane.normal_vector, [0, 0, -1]):
            raise ValueError(
                "The only supported plane normals are (0,0,1) and (0,0,-1)"
            )

        self.outer_boundary = _make_polygon(roi.plane.outer_boundary, roi)

        self.inner_boundaries = []
        if roi.plane.inner_boundaries is not None:
            self.inner_boundaries = [
                _make_polygon(boundary, roi) for boundary in roi.plane.inner_boundaries
            ]

        self.height = roi.height
        self.matrix = matrix

        self.roi = roi

    def _is_inside_boundaries(self, point: np.ndarray) -> bool:
        """Check if a point is inside the outer boundary and outside all inner boundaries.
        The point must be in the same system of coordinates as the boudnaries.
        :param point: A 3D point
        """

        xy_point = Point(point[0], point[1])

        return self.outer_boundary.contains(xy_point) and not any(
            boundary.contains(xy_point) for boundary in self.inner_boundaries
        )

    def _is_inside_elevation_bounds(self, point: np.ndarray) -> bool:
        """Check if a point is within the elevation bounds of the region of interest.
        The point must be in the same system of coordinates as the boudnaries.
        """
        if self.height is None:
            return True

        elevation_difference = point[2] - self.roi.plane.vertices3d[0][2]

        elevation_along_normal = elevation_difference * self.roi.plane.normal_vector[2]

        return elevation_along_normal > 0 and elevation_along_normal < self.height

    def is_inside(self, point: np.ndarray) -> bool:
        """Check if a point is inside the ROI.
        :param point: The pooint to check.
        :return: True if the point is inside the region of interest, False otherwise.
        """
        homogeneous_point = np.append(point, 1)
        if self.matrix is not None:
            opf_matrix_inverse = np.array(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
            )
            homogeneous_point = opf_matrix_inverse @ self.matrix @ homogeneous_point

        p = homogeneous_point[:3]

        return self._is_inside_boundaries(p) and self._is_inside_elevation_bounds(p)


def _filter_array(
    array: np.ndarray, flags: np.ndarray, output_file: Path
) -> np.ndarray:
    """Filter the array with a binary flag array, writing the output at the indicated path.
    The output file will be overwritten if it is already present.

    :param array: The array to filter. It is assumed to be two dimensional.
    :param flags: A one dimensional boolean array used to filter the array (by rows).
    :param output_file: The file to save the filtered array.

    :return: A memory mapped array representing the filtered data.

    :raise ValueError: If the length of the flags is not the same as the first dimension of the array.
    """

    if len(flags) != len(array):
        raise ValueError("The size of the array and the filter do not match")

    kept_entries = np.sum(flags)

    filtered = np.memmap(
        output_file,
        mode="w+",
        dtype=array.dtype,
        offset=0,
        shape=(kept_entries, array.shape[1]),
    )

    filtered[:] = array[flags]

    return filtered


def _copy_array(array: np.memmap | np.ndarray, output_path: Path) -> np.memmap:
    """Copies the array to a new location and returns the newly opened array"""

    new_array = np.memmap(output_path, mode="w+", dtype=array.dtype, shape=array.shape)
    new_array[:] = array[:]

    return new_array


def _filter_matches(matches: Matches, flags: np.ndarray, output_gltf_dir: Path):
    matches.point_index_ranges.ranges = _filter_array(
        matches.point_index_ranges.ranges,
        flags,
        output_gltf_dir / "matchPointIndexRanges.bin",
    )

    matches.camera_ids = _copy_array(
        matches.camera_ids, output_gltf_dir / "matchCameraIds.bin"
    )

    if matches.image_points is not None:
        matches.image_points.featureIds = _copy_array(
            matches.image_points.featureIds, output_gltf_dir / "matchFeatureIds.bin"
        )

        matches.image_points.scales = _copy_array(
            matches.image_points.scales, output_gltf_dir / "matchScales.bin"
        )

        matches.image_points.pixelCoordinates = _copy_array(
            matches.image_points.pixelCoordinates,
            output_gltf_dir / "matchPixelCoordinates.bin",
        )

        if matches.image_points.depths is not None:
            matches.image_points.depths = _copy_array(
                matches.image_points.depths, output_gltf_dir / "matchDepths.bin"
            )


def _filter_node(
    node: Node, roi: Pix4DRegionOfInterest, output_gltf_dir: Path
) -> Optional[Node]:
    """Filters in place a node and returns a reference to it.
    If the node is outside the ROI, None is returned and the initial value of the node is left unchanged.
    """

    roi_polygons = RoiPolygons(roi, node.matrix)

    # A list of flags, indicating whether each point should be kept or not
    flags = np.apply_along_axis(roi_polygons.is_inside, 1, node.position)

    if not np.any(flags):
        return None

    node.position = _filter_array(
        node.position, flags, output_gltf_dir / "positions.bin"
    )

    if node.normal is not None:
        node.normal = _filter_array(node.normal, flags, output_gltf_dir / "normals.bin")
    if node.color is not None:
        node.color = _filter_array(node.color, flags, output_gltf_dir / "colors.bin")
    if node.custom_attributes is not None:
        for name, attribute in node.custom_attributes.items():
            attribute = _filter_array(
                attribute, flags, output_gltf_dir / (name + ".bin")
            )

    if node.matches:
        _filter_matches(node.matches, flags, output_gltf_dir)

    return node


def filter_pointcloud(
    pointcloud: GlTFPointCloud, roi: Pix4DRegionOfInterest, output_gltf_dir: Path
) -> None:
    """Filters in-place a point cloud with a region of interest, keeping only the points that are inside the ROI.
    If a node is outside the region of interest, it will be removed.
    If all nodes are outside the ROI, the returned point cloud will not have any nodes.
    """

    maybe_nodes = [
        _filter_node(node, roi, output_gltf_dir) for node in pointcloud.nodes
    ]
    pointcloud.nodes = [node for node in maybe_nodes if node is not None]


def _get_region_of_interest(project: ProjectObjects) -> Pix4DRegionOfInterest:
    """Gets the region of interest extension from an OPF project.

    :param project: The project to query.
    :return: The region of interest.

    :raise ValueError: If the project contains zero or more than one regions of interest.
    """

    regions_of_interest = project.get_extensions_by_format(RoiFormat)

    if len(regions_of_interest) == 0:
        raise ValueError(
            "The project has no extensions of type Pix4DRegionOfInterest. Cropping is not supported"
        )
    elif len(regions_of_interest) > 1:
        raise ValueError(
            "The project has multiple extensions of type Pix4DRegionOfInterest. Cropping is not supported"
        )

    return cast(Pix4DRegionOfInterest, regions_of_interest[0])


def _get_pointcloud_objects(
    project: ProjectObjects,
) -> list[tuple[str, GlTFPointCloud]]:
    def get_point_cloud(obj):
        if isinstance(obj, GlTFPointCloud):
            return obj
        elif isinstance(obj, Calibration):
            return obj.tracks
        else:
            return None

    pointclouds: list[tuple[str, GlTFPointCloud]] = []

    for key, objects in project.__dict__.items():
        if key.startswith("_"):
            continue
        for object in objects:
            point_cloud = get_point_cloud(object)
            if point_cloud is None:
                continue
            pointclouds.append((str(object.metadata.id), point_cloud))

    return pointclouds


def _used_camera_uids_in_pointcloud(pointcloud: GlTFPointCloud) -> set[str]:

    used_camera_uids = set()

    for node in pointcloud.nodes:
        matches = node.matches
        if matches:
            for o, c in matches.point_index_ranges:
                ids = matches.camera_ids[o : o + c].flatten().tolist()
                uids = [matches.camera_uids[id] for id in ids]
                used_camera_uids.update(uids)

    return used_camera_uids


def _used_camera_uids_in_project(project: ProjectObjects) -> set[str]:

    pointclouds = _get_pointcloud_objects(project)

    used_camera_uids = set()
    for _, pointcloud in pointclouds:
        used_camera_uids.update(_used_camera_uids_in_pointcloud(pointcloud))

    return used_camera_uids


def _filter_cameras_without_points(project: ProjectObjects) -> None:
    """Remove all cameras from project which do not see any point in point_cloud"""

    used_camera_uids = _used_camera_uids_in_project(project)

    if project.input_cameras is None or project.projected_input_cameras is None:
        raise RuntimeError("Project does not contain input cameras")
    if project.calibration is None or project.calibration.calibrated_cameras is None:
        raise RuntimeError("Project is not calibrated")

    for capture in project.input_cameras.captures:
        capture.cameras = [
            camera for camera in capture.cameras if camera.id in used_camera_uids
        ]
    project.input_cameras.captures = [
        capture
        for capture in project.input_cameras.captures
        if len(capture.cameras) > 0
    ]

    used_capture_ids = [capture.id for capture in project.input_cameras.captures]
    project.projected_input_cameras.captures = [
        capture
        for capture in project.projected_input_cameras.captures
        if capture.id in used_capture_ids
    ]

    project.calibration.calibrated_cameras.cameras = [
        camera
        for camera in project.calibration.calibrated_cameras.cameras
        if camera.id in used_camera_uids
    ]


def _filter_control_points(project: ProjectObjects, roi: Pix4DRegionOfInterest) -> None:
    """Filter the control points (MTPs, GCPs) of a project with a region of interest
    The filtering is based on the positions of the calibrated control points with respect to the ROI.
    The corresponding input and projected control points are filtered based on their IDs.
    """

    if (
        project.calibration is None
        or project.calibration.calibrated_control_points is None
    ):
        return

    roi_polygons = RoiPolygons(roi)

    cps = project.calibration.calibrated_control_points
    cps.points = list(
        filter(lambda p: roi_polygons.is_inside(p.coordinates), cps.points)
    )

    if project.input_control_points is None or project.projected_control_points is None:
        raise RuntimeError("Input and projected control points must be present")

    remaining_ids = [cp.id for cp in cps.points]
    project.input_control_points.gcps = [
        gcp for gcp in project.input_control_points.gcps if gcp.id in remaining_ids
    ]
    project.input_control_points.mtps = [
        mtp for mtp in project.input_control_points.mtps if mtp.id in remaining_ids
    ]
    project.projected_control_points.projected_gcps = [
        projected_gcp
        for projected_gcp in project.projected_control_points.projected_gcps
        if projected_gcp.id in remaining_ids
    ]


def crop(
    project: ProjectObjects, output_path: Path, roi: Pix4DRegionOfInterest
) -> None:
    """Crop a collection of project objects using the pix4d region of interest extension.
    The project must contain exactly one instance of the region of interest extension
    and must be calibrated.

    :param project: A resolved project.
    :param output_path: A path where to save the new project.

    :raise ValueError: If the project does not contain a unique of the region of interest extension
    """

    output_path = output_path.absolute()

    pointclouds = _get_pointcloud_objects(project)

    for pointcloud_id, pointcloud in pointclouds:
        output_dir = output_path / pointcloud_id
        os.mkdir(output_dir)
        filter_pointcloud(pointcloud, roi, output_dir)

    _filter_cameras_without_points(project)

    _filter_control_points(project, roi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple OPF project cropping tool")
    parser.add_argument(
        "input",
        type=str,
        help="An OPF project file",
    )
    parser.add_argument(
        "outdir", type=str, help="Output directory for the cropped project"
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Optional path to a Region of Interest json file",
    )
    parser.add_argument(
        "--force",
        "-f",
        dest="force",
        action="store_true",
        default=False,
        help="Do not ask for confirmation when overwriting output files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.outdir)
    if os.path.exists(output_dir):
        if len(os.listdir(output_dir)) != 0 and not args.force:
            while True:
                try:
                    answer = input(
                        "The output directory is not empty do you want to procced [yN]? "
                    )
                except KeyboardInterrupt:
                    print()
                    exit(-1)
                except EOFError:
                    answer = ""
                if answer == "n" or answer == "N" or answer == "":
                    exit(0)
                if answer == "y" or answer == "Y":
                    break
    else:
        os.makedirs(output_dir)

    project = load(args.input)
    project = resolve(project, supported_extensions=["ext_pix4d_region_of_interest"])

    roi = load(args.roi) if args.roi else _get_region_of_interest(project)

    crop(project, Path(output_dir), roi=roi)

    save(project, str(output_dir / "project.opf"))
