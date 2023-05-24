import copy
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .pcl import (
    GlTFPointCloud,
    ImagePoints,
    Matches,
    PointIndexRanges,
    opf_axis_rotation_matrix,
    opf_axis_rotation_matrix_inverse,
)
from .utils import merge_arrays


def _check_property(objs: list[Any], prop: str):
    """Check if a property exist in all items of a list
    :param objs: A list of objects where to check for the presence of the property
    :param prop: The name of the property to check

    :return: True if the property is present in all objects, False if the property is not present in any of the objects

    :raise ValueError: If the property is present only in some of the objects, but not in all of them
    """
    flags = [getattr(p, prop, None) is not None for p in objs]

    if any(flags) and not all(flags):
        raise ValueError("Not all pointclouds share property: " + prop)

    return all(flags)


def _apply_affine_transform(array: np.ndarray | np.memmap, matrix: np.ndarray) -> None:
    """Applies in-place the affine transform represented by matrix to the points of array.
    :raise ValueError: If array does not have the shape (,3) or if matrix does not have the shape (4,4)
    """
    upper_left_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]
    array[:] = array @ upper_left_matrix.transpose() + translation


def _merge_image_points(
    image_points: list[ImagePoints], output_gltf_dir: Path
) -> ImagePoints:
    """Merge the ImagePoints data structure used as part of the OPF_mesh_primitive_matches glTF extension.

    :param image_points: A list of ImagePoints structures to merge. It is modified in-place.
    :param output_gltf_dir: The output directory for the binary buffers. It is assumed to exist.

    :return: The merged ImagePoints structure.

    :raise ValueError: If the image_points list is empty.
    """

    if len(image_points) == 0:
        raise ValueError("Empty image_points list")

    image_points[0].featureIds = merge_arrays(
        [ip.featureIds for ip in image_points], output_gltf_dir / "matchFeatureIds.bin"
    )
    image_points[0].pixelCoordinates = merge_arrays(
        [ip.pixelCoordinates for ip in image_points],
        output_gltf_dir / "matchPixelCoordinates.bin",
    )
    image_points[0].scales = merge_arrays(
        [ip.scales for ip in image_points], output_gltf_dir / "matchScales.bin"
    )

    if _check_property(image_points, "depths"):
        image_points[0].depths = merge_arrays(
            [ip.depths for ip in image_points if ip.depths is not None],
            output_gltf_dir / "matchDepths.bin",
        )

    return image_points[0]


def _merge_matches(matches: list[Matches], output_gltf_dir: Path) -> Matches:
    """Merge the Matches data structure used as part of the OPF_mesh_primitive_matches glTF extension.

    :param matches: A list of Matches structures to merge. It is modified in-place.
    :param output_gltf_dir: The output directory for the binary buffers. It is assumed to exist.

    :return: The merged Matches structure.

    :raise ValueError: If the matches list is empty.
    """

    if len(matches) == 0:
        raise ValueError("Empty matches list")

    camera_uids = []
    for m in matches:
        camera_uids.extend(m.camera_uids)

    camera_ids = merge_arrays(
        [m.camera_ids for m in matches], output_gltf_dir / "matchCameraIds.bin"
    )

    offset = matches[0].camera_ids.shape[0]
    uid_offset = len(matches[0].camera_uids)
    for m in matches[1:]:
        camera_ids[offset : offset + len(m.camera_ids)] += uid_offset
        offset += len(m.camera_ids)
        uid_offset += len(m.camera_uids)

    new_ranges = merge_arrays(
        [m.point_index_ranges.ranges for m in matches],
        output_gltf_dir / "matchPointIndexRanges.bin",
    )
    point_index_ranges = PointIndexRanges(new_ranges)

    offset = 0
    camera_ids_offset = 0
    for m in matches:
        for i in range(len(m.point_index_ranges)):
            o, c = m.point_index_ranges[i]
            point_index_ranges[offset + i] = (o + camera_ids_offset, c)
        offset += len(m.point_index_ranges)
        camera_ids_offset += len(m.camera_ids)

    matches[0].camera_uids = camera_uids
    matches[0].camera_ids = camera_ids
    matches[0].point_index_ranges = point_index_ranges

    if _check_property(matches, "image_points"):
        matches[0].image_points = _merge_image_points(
            [m.image_points for m in matches if m.image_points is not None],
            output_gltf_dir,
        )

    return matches[0]


def _merge_custom_attributes(
    custom_attributes: list[dict[str, np.ndarray | np.memmap]], output_gltf_dir: Path
) -> Optional[dict[str, np.ndarray | np.memmap]]:
    """Merge a list of custom attributes.
    :param custom_attributes: A list of dictionaries, representing the custom attributes of multiple point clouds
    :param output_gltf_dir: The output directory for the binary buffers. It is assumed to exist.

    :return: A dictionary mapping the custom attribute name to the numpy buffer or None if no common attributes were found
    """

    if len(custom_attributes) == 0 or len(custom_attributes[0]) == 0:
        return None

    common_attributes = set.intersection(
        *[set(attributes.keys()) for attributes in custom_attributes]
    )

    attributes = {}
    for common_attribute in common_attributes:
        arrays = [attributes[common_attribute] for attributes in custom_attributes]
        merged_attribute = merge_arrays(
            arrays, output_gltf_dir / (common_attribute + ".bin")
        )
        attributes[common_attribute] = merged_attribute

    return attributes


def concatenate(pointclouds: list[GlTFPointCloud]) -> GlTFPointCloud:
    """Concatenate the nodes of all point clouds in a single point cloud.
    The nodes may not share the same properties.

    :param pointclouds: The list of pointclouds to concantenate
    :return: A pointcloud which has as nodes all the nodes of the other pointclouds
    """

    concatenated = copy.deepcopy(pointclouds[0])

    for pointcloud in pointclouds[1:]:
        concatenated.nodes.extend(pointcloud.nodes)

    return concatenated


def collapse(pointcloud: GlTFPointCloud, output_gltf_dir: Path) -> GlTFPointCloud:
    """Collapse all nodes in a point cloud into one.
    The first node keeps its matrix.
    All nodes must share the same properties, including extensions and custom attributes.

    :param pointcloud: The pointclouds whose nodes to collapse. The data is modified in place and not recommended to use after this call.
    :param output_gltf_dir: The output dir for the glTF point cloud. It is assumed to exist.

    :return pointcloud: A point cloud which has only one node, containing the merged information from all its nodes.

    :raise ValueError: If only some of the nodes have some optional property present.

    :raise FileNotFoundError: If output_gltf_dir does not exist.
    """

    if not os.path.exists(output_gltf_dir):
        raise FileNotFoundError(
            "Output directory %s does not exist " % str(output_gltf_dir)
        )

    position = merge_arrays(
        [n.position for n in pointcloud.nodes], output_gltf_dir / "positions.bin"
    )
    offset = 0

    for node in pointcloud.nodes:
        count = len(node.position)
        matrix = node.matrix if node.matrix is not None else np.eye(4)
        matrix = opf_axis_rotation_matrix_inverse @ matrix
        _apply_affine_transform(position[offset : offset + count], matrix)
        offset += count

    pointcloud.nodes[0].position = position
    pointcloud.nodes[0].matrix = opf_axis_rotation_matrix

    if _check_property(pointcloud.nodes, "color"):
        pointcloud.nodes[0].color = merge_arrays(
            [n.color for n in pointcloud.nodes if n.color is not None],
            output_gltf_dir / "colors.bin",
        )

    if _check_property(pointcloud.nodes, "normal"):
        pointcloud.nodes[0].normal = merge_arrays(
            [n.normal for n in pointcloud.nodes if n.normal is not None],
            output_gltf_dir / "normals.bin",
        )

    if _check_property(pointcloud.nodes, "matches"):
        pointcloud.nodes[0].matches = _merge_matches(
            [n.matches for n in pointcloud.nodes if n.matches is not None],
            output_gltf_dir,
        )

    if _check_property(pointcloud.nodes, "custom_attributes"):
        pointcloud.nodes[0].custom_attributes = _merge_custom_attributes(
            [
                n.custom_attributes
                for n in pointcloud.nodes
                if n.custom_attributes is not None
            ],
            output_gltf_dir,
        )

    pointcloud.nodes = [pointcloud.nodes[0]]

    return pointcloud
