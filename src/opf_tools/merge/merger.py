import argparse
import copy
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, TypeVar, overload
from uuid import UUID

import numpy as np

import pyopf.pointcloud.merge as pcl_merge
from pyopf.cameras import (
    CalibratedCameras,
    CameraList,
    Capture,
    GpsBias,
    InputCameras,
    ProjectedCapture,
    ProjectedInputCameras,
    ProjectedSensor,
    Sensor,
)
from pyopf.cps import (
    CalibratedControlPoint,
    CalibratedControlPoints,
    Constraints,
    InputControlPoints,
    Mark,
    OrientationConstraint,
    ProjectedControlPoints,
    ScaleConstraint,
)
from pyopf.crs import Crs, Geolocation, SceneReferenceFrame
from pyopf.io import load, save
from pyopf.pointcloud import GlTFPointCloud
from pyopf.pointcloud.pcl import (
    opf_axis_rotation_matrix,
    opf_axis_rotation_matrix_inverse,
)
from pyopf.project import (
    Calibration,
    Metadata,
    Project,
    ProjectObjects,
    ProjectSource,
)
from pyopf.resolve import resolve
from pyopf.types import OpfObject
from pyopf.uid64 import Uid64, uid64

Object = TypeVar("Object", bound=OpfObject)


def _clear_unsupported_attributes(object: Object) -> Object:
    object.unknown_properties = None
    object.extensions = None
    return object


def _are_crss_equal(lhs: Crs, rhs: Crs) -> bool:
    # This implementation is very naïve at the moment
    return lhs.definition == rhs.definition and lhs.geoid_height == rhs.geoid_height


def _are_geolocations_equal(
    lhs: Optional[Geolocation], rhs: Optional[Geolocation]
) -> bool:

    return (lhs is None) == (rhs is None) and (
        lhs is None
        or (
            bool(
                rhs is not None
                and _are_crss_equal(lhs.crs, rhs.crs)
                and np.equal(lhs.coordinates, rhs.coordinates).all()
                and np.equal(lhs.sigmas, rhs.sigmas).all()
            )
        )
    )


def _find_or_remap_id(id: Uid64, mapping: dict[Uid64, Uid64]) -> Uid64:
    try:
        return mapping[id]
    except KeyError:
        # We don't do anything to avoid colissions here based on the fact that the
        # probability of having a collision with 1,000,000 elements is 2.7e-8 which
        # is just a bit higher than the probability of winning EuroMillions. We have
        # to play this game ~ 4,000,000 times to have a 10% chance of having one or
        # more collisions.
        new_id = uid64()
        mapping[id] = new_id
        return new_id


def _merge_names(objects: list[Any]):

    names = [o.metadata.name for o in objects if o.metadata.name is not None]

    unnamed = len(objects) - len(names)
    if unnamed == len(objects):
        return None  # don't give any name to a union of unamed items
    if unnamed == 1:
        names.append("1 unnamed item")
    elif unnamed != 0:
        names.append("%d unnamed items" % unnamed)

    return " + ".join(names)


def _make_temporary_sources(objects: list[Any]):
    """This function is meant to return the union of all original sources.
    The purpose is to use it after all objects are merged to replace all sources
    of the same type by the object in which they have been merged"""
    sources = []
    for object in objects:
        if isinstance(object.metadata.sources, list):
            sources += object.metadata.sources
        else:
            for source in object.metadata.sources.__dict__.values():
                sources.append(
                    ProjectSource(id=source.metadata.id, type=source.metadata.type)
                )
    return sources


def _merge_subojects_by_key(
    key, containers, shifts=None, sensor_id_mappings=None, **kwargs
):
    attributes = [
        getattr(container, key) for container in containers if hasattr(container, key)
    ]
    for o in attributes:
        if isinstance(o, list) and len(o) > 1:
            raise ValueError(
                "Impossible to merge objects with more than one instance per suboject type"
            )
    # Flattenning the attributes which are lists to their first element only
    objects = []
    for attribute in attributes:
        if isinstance(attribute, list):
            if len(attribute) == 1:
                objects.append(attribute[0])
        elif attribute is not None:
            objects.append(attribute)

    if len(objects) == 0:
        return None

    kwargs = copy.copy(kwargs)

    # Filtering of arguments that are lists with as many elements as projects to pass the
    # right elements depending on the presence of the current object key in each project.
    def filter_list(elements):
        def accept(container, key):
            if not hasattr(container, key):
                return False
            attribute = getattr(container, key)
            if isinstance(attribute, list):
                return len(attribute) != 0
            else:
                return attribute is not None

        return [
            element
            for container, element in zip(containers, elements)
            if accept(container, key)
        ]

    if shifts:
        kwargs["shifts"] = filter_list(shifts)
    if sensor_id_mappings:
        kwargs["sensor_id_mappings"] = filter_list(sensor_id_mappings)

    return merge(*objects, **kwargs)


def _fix_sources(project: ProjectObjects, uuid_mapping: dict[UUID, UUID]) -> None:
    for key, items in project.__dict__.items():

        if key.startswith("_"):
            continue

        for item in items:
            sources = {
                uuid_mapping[source.id]: source.type for source in item.metadata.sources
            }
            item.metadata.sources = [
                ProjectSource(id=id, type=type) for id, type in sources.items()
            ]


def _merge_labels(objs: list[Any]):
    labels = list(
        {
            label
            for o in objs
            if o.metadata.labels is not None
            for label in o.metadata.labels
        }
    )
    if len(labels) == 0:
        return None
    return labels


def _merge_scene_reference_frames(
    srss: list[SceneReferenceFrame], **kwargs
) -> SceneReferenceFrame:
    # Nothing smart here, just return the first one after verifying they are
    # all compatible.
    srs = copy.deepcopy(srss[0])
    for other in srss[1:]:
        if (
            other.crs.definition != srs.crs.definition
            or other.base_to_canonical.swap_xy != srs.base_to_canonical.swap_xy
            or (other.base_to_canonical.scale != srs.base_to_canonical.scale).any()
        ):
            raise RuntimeError("Incompatible spatial reference frames for merging")
    _clear_unsupported_attributes(srs)

    if srs.metadata is None:
        raise RuntimeError("SceneReferenceFrame metadata is None")

    srs.metadata.sources = []
    return srs


def _merge_metadata(objects: list[Any]):
    # This function assumes that all objects are of the same type
    return Metadata(
        type=objects[0].metadata.type,
        name=_merge_names(objects),
        labels=_merge_labels(objects),
        sources=_make_temporary_sources(objects),
    )


def _merge_camera_lists(camera_objs: list[CameraList], **kwargs) -> CameraList:

    id_to_uri = {}

    # The consistency of the camera UIDs is checked while creating the final list of cameras
    cameras = []
    for camera_list in camera_objs:
        for camera in camera_list.cameras:
            try:
                if id_to_uri[camera.id] != camera.uri:
                    raise RuntimeError(
                        "Fatal error: Camera UID inconsistency found for UID %s"
                        % camera.id
                    )
            except KeyError:
                id_to_uri[camera.id] = camera.uri
                cameras.append(_clear_unsupported_attributes(copy.copy(camera)))

    result = CameraList(cameras)
    result.metadata = _merge_metadata(camera_objs)
    return result


def _merge_marks(list1: list[Mark], list2: list[Mark]) -> list[Mark]:

    marks = {m.camera_id: m for m in list1}
    for mark in list2:
        try:
            previous = marks[mark.camera_id]
            if (
                previous.position_px != mark.position_px
            ).any() or previous.accuracy != mark.accuracy:
                raise RuntimeError("Inconsistent marks")
        except KeyError:
            marks[mark.camera_id] = mark

    return list(marks.values())


def _merge_input_control_points(
    cps_objs: list[InputControlPoints], **kwargs
) -> InputControlPoints:

    gcps = {}
    mtps = {}

    def check_gcp_compatible(lhs, rhs) -> bool:
        return lhs.is_checkpoint == rhs.is_checkpoint and _are_geolocations_equal(
            lhs.geolocation, rhs.geolocation
        )

    def check_mtp_compatible(lhs, rhs) -> bool:
        return lhs.is_checkpoint == rhs.is_checkpoint

    def process_tie_point(tp, output, other_type_output, comparison_fun):

        if tp.id in other_type_output:
            raise RuntimeError("Tie point found as both MTP and GCP")
        try:
            existing = output[tp.id]
            if not comparison_fun(tp, existing):
                raise RuntimeError(
                    "Two incompatible instances of the same tie point found"
                )
            existing.marks = _merge_marks(existing.marks, tp.marks)
        except KeyError:
            output[tp.id] = _clear_unsupported_attributes(copy.deepcopy(tp))
        except RuntimeError as e:
            raise RuntimeError(
                "Fatal error: Merging tie point %s: %s" % (tp.id, e.args[0])
            ) from e

    for cps in cps_objs:
        for gcp in cps.gcps:
            process_tie_point(gcp, gcps, mtps, check_gcp_compatible)
        for mtp in cps.mtps:
            process_tie_point(mtp, mtps, gcps, check_mtp_compatible)

    result = InputControlPoints(gcps=list(gcps.values()), mtps=list(mtps.values()))
    result.metadata = _merge_metadata(cps_objs)
    return result


def _merge_projected_control_points(cps_objs: list[ProjectedControlPoints], **kwargs):

    gcps = {}

    shifts = kwargs.get("shifts", [np.zeros(3) for i in range(len(cps_objs))])

    for cps, shift in zip(cps_objs, shifts):
        for gcp in cps.projected_gcps:

            gcp = _clear_unsupported_attributes(copy.deepcopy(gcp))
            gcp.coordinates += shift

            try:
                existing = gcps[gcp.id]
                if not np.allclose(
                    gcp.coordinates, existing.coordinates
                ) or not np.allclose(gcp.sigmas, existing.sigmas):
                    raise RuntimeError(
                        "Fatal error: Repeated ID merging incompatible projected GCP: %s"
                        % gcp.id
                    )
            except KeyError:
                gcps[gcp.id] = gcp

    result = ProjectedControlPoints(projected_gcps=list(gcps.values()))
    result.metadata = _merge_metadata(cps_objs)
    return result


def _merge_orientation_constraints(
    constraint_objs: list[Constraints],
) -> list[OrientationConstraint]:

    orientations = {}
    for constraints in constraint_objs:
        for orientation in constraints.orientation_constraints:
            try:
                existing = orientations[orientation.id]
                if (
                    existing.id_from != orientation.id_from
                    or existing.id_to != orientation.id_to
                    or existing.sigma_deg != orientation.sigma_deg
                    or (existing.unit_vector != orientation.unit_vector).any()
                ):
                    raise RuntimeError(
                        "FatalError: Repeated ID merging incompatible orientation contraint: %s"
                        % orientation.id
                    )
            except KeyError:
                orientations[orientation.id] = orientation

    return list(orientations.values())


def _merge_scale_constraints(
    constraint_objs: list[Constraints],
) -> list[ScaleConstraint]:

    scales = {}
    for constraints in constraint_objs:
        for scale in constraints.scale_constraints:
            try:
                existing = scales[scale.id]
                if (
                    existing.id_from != scale.id_from
                    or existing.id_to != scale.id_to
                    or existing.sigma != scale.sigma
                    or existing.distance != scale.distance
                ):
                    raise RuntimeError(
                        "FatalError: Repeated ID merging incompatible scale contraint: %s"
                        % scale.id
                    )
            except KeyError:
                scales[scale.id] = scale

    return list(scales.values())


def _merge_constraints(constraint_objs: list[Constraints], **kwargs) -> Constraints:

    result = Constraints(
        orientation_constraints=_merge_orientation_constraints(constraint_objs),
        scale_constraints=_merge_scale_constraints(constraint_objs),
    )
    result.metadata = _merge_metadata(constraint_objs)
    return result


@overload
def _merge_sensors(
    camera_objs: list[InputCameras],
    sensor_id_mappings: list[dict[Uid64, Uid64]],
    input_type_name="input cameras",
) -> list[Sensor]:
    ...


@overload
def _merge_sensors(
    camera_objs: list[ProjectedInputCameras],
    sensor_id_mappings: list[dict[Uid64, Uid64]],
    input_type_name="input cameras",
) -> list[ProjectedSensor]:
    ...


def _merge_sensors(
    camera_objs: list[InputCameras] | list[ProjectedInputCameras],
    sensor_id_mappings: list[dict[Uid64, Uid64]],
    input_type_name="input cameras",
) -> list[Sensor] | list[ProjectedSensor]:

    """Merge all sensors in a single list.
    Sensors in each InputCameras object get reassigned ID based on the input mapping or
    new random IDs are produced and stored in the mapping if the sensor is not found."""

    sensors = []
    for input_cameras, mapping in zip(camera_objs, sensor_id_mappings):
        used_sensor_ids = set()
        for sensor in input_cameras.sensors:
            if sensor.id in used_sensor_ids:
                raise RuntimeError(
                    "Fatal error: Repeated sensor ID found in the same %s: %s"
                    % (input_type_name, sensor.id)
                )
            used_sensor_ids.add(sensor.id)

            sensor = _clear_unsupported_attributes(copy.deepcopy(sensor))
            sensor.id = _find_or_remap_id(sensor.id, mapping)
            sensors.append(sensor)

    return sensors


def _merge_captures(
    camera_objs: list[InputCameras], sensor_id_mappings: list[dict[Uid64, Uid64]]
) -> list[Capture]:
    """Merge all captures in a single list, applying a sensor_id remapping.
    It's not allowed to have two captures with the same ID in two different input camera lists
    or the same camera in to different captures.
    """
    captures = []
    used_capture_ids = set()
    used_camera_ids = set()

    for input_cameras, mapping in zip(camera_objs, sensor_id_mappings):
        for capture in input_cameras.captures:
            if capture.id in used_capture_ids:
                raise RuntimeError(
                    "Fatal error: Repeated capture found in input cameras: %s"
                    % capture.id
                )
            used_capture_ids.add(capture.id)
            capture = _clear_unsupported_attributes(copy.deepcopy(capture))
            for camera in capture.cameras:
                if camera.id in used_camera_ids:
                    raise RuntimeError(
                        "Fatal error: Repeated camera found in input cameras: %s"
                        % camera.id
                    )
                used_camera_ids.add(camera.id)
                camera.sensor_id = _find_or_remap_id(camera.sensor_id, mapping)

            captures.append(capture)

    return captures


def _merge_input_cameras(camera_objs: list[InputCameras], **kwargs) -> InputCameras:
    """Merges all input cameras into a single list.
    Sensor IDs are remapped to ensure that each sub project uses different IDs. Camera UIDs are not
    remapped as the input lists are assumed to be using non-overlapping sets of UIDs.
    :param camera_objs: a list of :class:`InputCameras`
    :param kwargs: may contain an input/output `sensor_id_mappings`. This parameter is a list
    of dicts to be used for reusing and/or completing the sensor id mapping. Each element in the list
    is the mapping to be used for each InputCameras in the same order.
    :return: a :class:`InputCameras` instance
    """
    mappings = kwargs.get("sensor_id_mappings", [{} for i in range(len(camera_objs))])
    sensors = _merge_sensors(camera_objs, mappings)
    captures = _merge_captures(camera_objs, mappings)

    result = InputCameras(captures=captures, sensors=sensors)
    result.metadata = _merge_metadata(camera_objs)

    return result


def _merge_projected_sensors(
    camera_objs: list[ProjectedInputCameras],
    sensor_id_mappings: list[dict[Uid64, Uid64]],
) -> list[ProjectedSensor]:
    return _merge_sensors(camera_objs, sensor_id_mappings, "projected input cameras")


def _merge_projected_captures(
    camera_objs: list[ProjectedInputCameras], shifts: list[np.ndarray]
) -> list[ProjectedCapture]:
    """Merge all projected captures in a single list, applying a shift to projected coordinates.
    It's not allowed to have two captures with the same ID in two different projected camera lists.
    """
    captures = []
    used_capture_ids = set()

    for projected_cameras, shift in zip(camera_objs, shifts):
        for capture in projected_cameras.captures:
            if capture.id in used_capture_ids:
                raise RuntimeError(
                    "Fatal error: Repeated capture found in projected input cameras: %s"
                    % capture.id
                )
            used_capture_ids.add(capture.id)
            capture = _clear_unsupported_attributes(copy.deepcopy(capture))
            if capture.geolocation is not None:
                capture.geolocation.position += shift

            captures.append(capture)

    return captures


def _merge_projected_input_cameras(
    camera_objs: list[ProjectedInputCameras], **kwargs
) -> ProjectedInputCameras:
    """Merges all projected input cameras into a single list.
    Sensor IDs are remapped to ensure that each sub project uses different IDs. Camera UIDs are not
    remapped as the input lists are assumed to be using non-overlapping sets of UIDs.
    :param camera_objs: a list of :class:`ProjectedInputCameras`
    :param kwargs: may contain:
      * a 'shifts' parameter with the shifts to apply to projected coordinates in each item
      * an input/output `sensor_id_mappings`. This parameter is a list
        of dicts to be used for reusing and/or completing the sensor id mapping. Each element in
        the list is the mapping to be used for each InputCameras in the same order.
    :return: a :class:`ProjectedInputCameras` instance
    """
    mappings = kwargs.get("sensor_id_mappings", [{} for i in range(len(camera_objs))])
    sensors = _merge_projected_sensors(camera_objs, mappings)

    shifts = kwargs.get("shifts", [np.zeros(3) for i in range(len(camera_objs))])
    captures = _merge_projected_captures(camera_objs, shifts)

    result = ProjectedInputCameras(captures=captures, sensors=sensors)
    result.metadata = _merge_metadata(camera_objs)

    return result


def _merge_calibrated_cameras(
    camera_objs: list[CalibratedCameras], **kwargs
) -> CalibratedCameras:

    mappings = kwargs.get("sensor_id_mappings", [{} for i in range(len(camera_objs))])
    shifts = kwargs.get("shifts", [np.zeros(3) for i in range(len(camera_objs))])

    sensors = []
    cameras = []
    used_camera_ids = set()

    for calibrated_cameras, shift, mapping in zip(camera_objs, shifts, mappings):
        for sensor in calibrated_cameras.sensors:
            sensor = _clear_unsupported_attributes(copy.deepcopy(sensor))
            sensor.id = _find_or_remap_id(sensor.id, mapping)
            sensors.append(sensor)

        for camera in calibrated_cameras.cameras:
            camera = _clear_unsupported_attributes(copy.deepcopy(camera))
            if camera.id in used_camera_ids:
                raise RuntimeError(
                    "Fatal error: Repeated camera found in calibrated cameras: %s"
                    % camera.id
                )
            used_camera_ids.add(camera.id)
            camera.sensor_id = _find_or_remap_id(camera.sensor_id, mapping)
            camera.position += shift
            cameras.append(camera)

    # No metadata needs to be merged or considered because this is not a top level project item
    return CalibratedCameras(cameras=cameras, sensors=sensors)


def _merge_calibrated_control_points(
    cp_objs: list[CalibratedControlPoints], **kwargs
) -> CalibratedControlPoints:

    shifts = kwargs.get("shifts", [np.zeros(3) for i in range(len(cp_objs))])

    points = defaultdict(list)
    for calibrated_cps, shift in zip(cp_objs, shifts):
        for point in calibrated_cps.points:
            points[point.id].append(point.coordinates + shift)

    # No metadata needs to be merged or considered because this is not a top level project item
    return CalibratedControlPoints(
        points=[
            CalibratedControlPoint(id=id, coordinates=np.mean(coords, 0))
            for id, coords in points.items()
        ]
    )


def _merge_gps_bias(gps_bias_objs: list[GpsBias], **kwargs):
    warnings.warn(
        "GPS bias resources cannot be merged, the output will not contain any"
    )


def _merge_calibrations(calibrations: list[Calibration], **kwargs) -> Calibration:

    result = Calibration()
    result.metadata = _merge_metadata(calibrations)

    kwargs = copy.copy(kwargs)
    shifts = kwargs.pop("shifts", [np.zeros(3) for i in range(len(calibrations))])
    sensor_id_mappings = kwargs.pop(
        "sensor_id_mappings", [{} for i in range(len(calibrations))]
    )

    kwargs["base_dir"] = kwargs.get("base_dir", Path(".")) / str(result.metadata.id)

    keys = {
        key
        for calibration in calibrations
        for key in calibration.__dict__
        if not key.startswith("_")
    }
    for key in keys:
        merged_item = _merge_subojects_by_key(
            key, calibrations, shifts, sensor_id_mappings, **kwargs
        )
        if merged_item:
            setattr(result, key, [merged_item])

    return result


def _verify_calibrated_control_point_consistency(
    projects: list[ProjectObjects], **kwargs
):
    """Verifies that the standard deviation of the distribution of calibrated control points positions
    is consistent with the measurement error given in the input control points.
    Raises an error in case of inconsistency."""
    pass


def _merge_point_clouds(point_clouds: list[GlTFPointCloud], **kwargs) -> GlTFPointCloud:

    shifts = kwargs.get("shifts", [np.zeros(3) for i in range(len(point_clouds))])
    base_dir = kwargs.get("base_dir", Path("."))

    if hasattr(point_clouds[0], "metadata"):
        metadata = _merge_metadata(point_clouds)
        output_dir = base_dir / str(metadata.id)
    else:
        metadata = None
        output_dir = base_dir

    result = pcl_merge.concatenate(point_clouds)

    shifts = [
        shift
        for shift, point_cloud in zip(shifts, point_clouds)
        for i in range(len(point_cloud.nodes))
    ]

    for node, shift in zip(result.nodes, shifts):
        if not (shift == np.zeros(3)).all():
            if node.matrix is not None:
                node.matrix = opf_axis_rotation_matrix_inverse @ node.matrix
                node.matrix[0, 3] += shift[0]
                node.matrix[1, 3] += shift[1]
                node.matrix[2, 3] += shift[2]
                node.matrix = opf_axis_rotation_matrix @ node.matrix
            else:
                # Since there was no matrix, we can't assume much about
                # the origin of the data.
                node.matrix = np.array(
                    [
                        [1, 0, 0, shift[0]],
                        [0, 1, 0, shift[1]],
                        [0, 0, 1, shift[2]],
                        [0, 0, 0, 1],
                    ]
                )

    os.mkdir(output_dir)

    result = pcl_merge.collapse(result, output_dir)
    if metadata:
        result.metadata = metadata

    return result


def _merge_project_objects(projects: list[ProjectObjects], **kwargs) -> ProjectObjects:

    assert len(projects) > 1  # This is guaranteed by the public merge function

    result = ProjectObjects()

    result.metadata.name = "Merged project"
    result.metadata.description = "Merge of projects: " + ", ".join(
        [str(project.metadata.id) for project in projects]
    )

    # This contains the mapping from the original item UUIDs to the UUID of the merged item,
    uuid_mapping = {}

    def merge_by_key(key, shifts=None, sensor_id_mappings=None, **kwargs):

        result = _merge_subojects_by_key(
            key, projects, shifts, sensor_id_mappings, **kwargs
        )

        if result is None:
            return None

        for project in projects:
            try:
                for object in getattr(project, key):
                    uuid_mapping[object.metadata.id] = result.metadata.id
            except AttributeError:
                pass

        return result

    # The scene reference frames require special treatment
    scene_reference_frame = merge_by_key("scene_reference_frame_objs")
    if scene_reference_frame:
        setattr(result, "scene_reference_frame_objs", [scene_reference_frame])

    if scene_reference_frame is None:
        raise RuntimeError("Could not obtain the scene reference frame")

    shifts = []
    for project in projects:
        if project.scene_reference_frame is not None:
            shifts.append(
                scene_reference_frame.base_to_canonical.shift
                - project.scene_reference_frame.base_to_canonical.shift
            )
        else:
            shifts.append(None)
    sensor_id_mappings = [{} for i in range(len(projects))]

    keys = {
        key
        for project in projects
        for key in project.__dict__
        if key != "scene_reference_frame_objs" and not key.startswith("_")
    }
    for key in keys:
        merged_item = merge_by_key(key, shifts, sensor_id_mappings, **kwargs)
        if merged_item:
            setattr(result, key, [merged_item])

    _fix_sources(result, uuid_mapping)
    _verify_calibrated_control_point_consistency(projects)

    return result


def _merge_projects(projects: list[Project], **kwargs):
    return _merge_project_objects([resolve(project) for project in projects], **kwargs)


def merge(first, *rest, **kwargs):

    for x in rest:
        if type(first) != type(x):
            raise TypeError("Objects of mixed types cannot be merged")

    objects = [first] + list(rest)

    for t, fun in [
        (Calibration, _merge_calibrations),
        (CalibratedCameras, _merge_calibrated_cameras),
        (CalibratedControlPoints, _merge_calibrated_control_points),
        (CameraList, _merge_camera_lists),
        (Constraints, _merge_constraints),
        (GlTFPointCloud, _merge_point_clouds),
        (GpsBias, _merge_gps_bias),
        (InputCameras, _merge_input_cameras),
        (InputControlPoints, _merge_input_control_points),
        (Project, _merge_projects),
        (ProjectObjects, _merge_project_objects),
        (ProjectedControlPoints, _merge_projected_control_points),
        (ProjectedInputCameras, _merge_projected_input_cameras),
        (SceneReferenceFrame, _merge_scene_reference_frames),
    ]:

        if type(first) == t:
            return fun(objects, **kwargs)

    raise TypeError("Unknown OPF type to merge: %s" % type(first))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple OPF project merging tool")
    parser.add_argument(
        "input",
        metavar="project.json",
        type=str,
        nargs="+",
        help="A list of OPF project files",
    )
    parser.add_argument(
        "outdir", type=str, help="Output directory for the merged project"
    )
    parser.add_argument(
        "-f",
        "--force",
        dest="force",
        action="store_true",
        default=False,
        help="Do not ask for confirmation when overwriting output files",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if len(args.input) == 1:
        print("Only one input project was given, is the output directory missing?")
        return 0

    output_dir = args.outdir
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

    projects = [load(input) for input in args.input]

    merged_project = merge(*projects, base_dir=Path(output_dir))

    save(merged_project, output_dir + "/project.opf")
