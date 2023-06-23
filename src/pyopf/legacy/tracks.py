import struct
import warnings
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Optional

import numpy as np

from ..formats import UnknownFormat
from ..io.loaders import join_uris, loaders
from ..io.savers import savers, to_uri_reference
from ..pointcloud.pcl import (
    GlTFPointCloud,
    ImagePoints,
    Matches,
    Node,
    PointIndexRanges,
    opf_axis_rotation_matrix,
    opf_axis_rotation_matrix_inverse,
)
from ..project import ProjectResource
from ..uid64 import Uid64
from ..versions import VersionInfo


class TrackFormat:
    POSITIONS = UnknownFormat("application/opf-track-positions+bin")
    MATCH_RANGES = UnknownFormat("application/opf-track-match-ranges+bin")
    MATCHES = UnknownFormat("application/opf-track-matches+bin")


_header_format = "<QIIQQ"
_header_size = struct.calcsize(_header_format)
_attribute_file_signature = np.uint64(0x3C192C2A08BDC7A9)
_attribute_group_file_signature = np.uint64(0x4A192C2A51AAC2B2)
_version = (np.uint32(0), np.uint32(1))


_match_dtype = np.dtype(
    [
        ("cameraId", np.uint64),
        ("featureId", np.uint64),
        ("pixel", [("x", np.uint32), ("y", np.uint32)]),
        ("scale", np.double),
        ("depth", np.double),
    ]
)


class _Attribute(np.uint64, Enum):
    POSITIONS = (0,)
    MATCH_RANGES = (1,)


class _AttributeGroup(np.uint64, Enum):
    MATCHES = (0,)


def _read_and_check_header(
    f: BinaryIO, signature: np.uint64, attribute: np.uint64
) -> Optional[tuple]:
    header = struct.unpack(_header_format, f.read(_header_size))
    if (
        header[0] != signature
        or header[1] != _version[0]
        or header[2] != _version[1]
        or header[3] != attribute
    ):
        return None
    return header


def _read_attribute(file_path: Path, attribute: _Attribute) -> np.ndarray:
    with open(file_path, "rb") as file:
        header = _read_and_check_header(file, _attribute_file_signature, attribute)
        if header is None:
            raise RuntimeError(
                f"Invalid attribute file: {file_path} for attribute {attribute}"
            )
        if attribute == _Attribute.POSITIONS:
            return np.fromfile(file, dtype="f", count=header[4] * 3).reshape(
                header[4], 3
            )

        elif attribute == _Attribute.MATCH_RANGES:
            return np.fromfile(file, dtype=np.uint64, count=header[4])

        raise ValueError("Invalid binary track attribute")


def _read_attribute_group(file_path: Path, attribute: _AttributeGroup) -> np.ndarray:
    with open(file_path, "rb") as file:
        header = _read_and_check_header(
            file, _attribute_group_file_signature, attribute
        )
        if header is None:
            raise RuntimeError(
                f"Invalid attribute file: {file_path} for attribute group {attribute}"
            )
        if attribute == _AttributeGroup.MATCHES:
            return np.fromfile(file, dtype=_match_dtype)

        raise ValueError("Invalid binary track attribute")


def _write_data_file(
    file_path: Path,
    data: np.ndarray,
    header_signature: np.uint64,
    attribute: _Attribute | _AttributeGroup,
) -> None:
    with open(file_path, "wb") as file:
        header = struct.pack(
            _header_format,
            header_signature,
            _version[0],
            _version[1],
            attribute,
            len(data),
        )
        file.write(header)
        data.tofile(file)


def _write_attribute(file_path: Path, data: np.ndarray, attribute: _Attribute):
    _write_data_file(file_path, data, _attribute_file_signature, attribute)


def _write_attribute_group(
    file_path: Path, data: np.ndarray, attribute: _AttributeGroup
):
    _write_data_file(file_path, data, _attribute_group_file_signature, attribute)


def _compact_uids(uids: np.ndarray):
    """Takes an array of UIDs `x` and returns an array with the sorted unique UIDs `u` and a
    list of indices `is`, such as (u[is] == x).all() == True."""
    sorted_uids = np.sort(uids)

    offsets = np.zeros(len(sorted_uids), dtype=np.uint32)
    offsets[1:] = sorted_uids[1:] != sorted_uids[:-1]
    offsets = np.cumsum(offsets, dtype=np.uint32)

    indices = offsets[np.argsort(np.argsort(uids))]
    uids = np.unique(sorted_uids)

    return uids, indices


class Tracks:
    def __init__(
        self, positions: np.ndarray, match_ranges: np.ndarray, matches: np.ndarray
    ):
        self.positions = positions
        self.match_ranges = match_ranges
        self.matches = matches

    @staticmethod
    def open(positions: Path, match_ranges: Path, matches: Path):
        return Tracks(
            _read_attribute(positions, _Attribute.POSITIONS),
            _read_attribute(match_ranges, _Attribute.MATCH_RANGES),
            _read_attribute_group(matches, _AttributeGroup.MATCHES),
        )


def tracks_to_gltf(tracks: Tracks) -> GlTFPointCloud:

    pcl = GlTFPointCloud()

    image_points = ImagePoints()
    image_points.featureIds = np.frombuffer(
        tracks.matches["featureId"].tobytes(), dtype=np.uint32
    ).reshape(-1, 2)
    image_points.scales = np.array(tracks.matches["scale"], dtype="f4").reshape(-1, 1)
    if (tracks.matches["depth"] != -1).any():
        image_points.depths = np.array(tracks.matches["depth"], dtype="f4").reshape(
            -1, 1
        )
    image_points.pixelCoordinates = np.zeros((tracks.matches.shape[0], 2), dtype="f4")
    image_points.pixelCoordinates[:, 0] = tracks.matches["pixel"]["x"]
    image_points.pixelCoordinates[:, 1] = tracks.matches["pixel"]["y"]

    matches = Matches()
    matches.image_points = image_points
    matches.point_index_ranges = PointIndexRanges(
        np.frombuffer(tracks.match_ranges, dtype=np.uint32).reshape(
            len(tracks.match_ranges), 2
        )
    )

    camera_uids, camera_ids = _compact_uids(tracks.matches["cameraId"])
    matches.camera_ids = camera_ids.reshape(-1, 1)
    matches.camera_uids = [Uid64(int=x) for x in camera_uids]

    node = Node()
    node.matrix = opf_axis_rotation_matrix
    node.position = np.array(tracks.positions, dtype="f4")
    node.matches = matches

    pcl.nodes = [node]

    pcl._version = VersionInfo(1, 0, "draft7")

    return pcl


def gltf_to_tracks(pcl: GlTFPointCloud) -> Tracks:

    if len(pcl) != 1:
        raise RuntimeError(
            "Conversion from OPF glTF to legacy tracks expects a single node point cloud"
        )

    node = pcl.nodes[0]

    positions = np.array(node.position, dtype="f4")
    # Maybe we could skip the transformation, but it's considered just in case
    if node.matrix is not None:
        matrix = opf_axis_rotation_matrix_inverse @ node.matrix
        upper_left_matrix = matrix[:3, :3]
        translation = matrix[:3, 3]
        positions[:] = positions @ upper_left_matrix.transpose() + translation

    if node.color is not None:
        warnings.warn(
            "Color information will be lost converting glTF point cloud to legacy tracks"
        )

    if node.normal is not None or node.custom_attributes is not None:
        warnings.warn(
            "Unexpected node attribute converting glTF point cloud to legacy tracks"
        )

    if node.matches is None:
        raise RuntimeError("Tracks missing converting OPF glTF tracks to legacy tracks")

    match_ranges = np.frombuffer(node.matches.point_index_ranges.ranges, dtype=np.int64)
    matches = np.zeros(len(node.matches.camera_ids), dtype=_match_dtype)

    if node.matches.image_points is None:
        raise RuntimeError(
            "Image points missing converting OPF glTF tracks to legacy tracks"
        )

    matches["cameraId"] = np.array(
        [uid.int for uid in node.matches.camera_uids], dtype=np.uint64
    )[node.matches.camera_ids.flatten()]
    matches["featureId"] = np.frombuffer(
        node.matches.image_points.featureIds.tobytes(), dtype=np.uint64
    )
    matches["pixel"]["x"] = node.matches.image_points.pixelCoordinates[
        :, 0
    ]  # Implicit narrowing from float to int
    matches["pixel"]["y"] = node.matches.image_points.pixelCoordinates[
        :, 1
    ]  # Implicit narrowing from float to int
    matches["scale"] = node.matches.image_points.scales.flatten()
    if node.matches.image_points.depths is None:
        matches["depth"][:] = -1
    else:
        matches["depth"] = node.matches.image_points.depths.flatten()

    return Tracks(positions, match_ranges, matches)


def _saver(
    tracks: Tracks, path: Path, base_path: Optional[Path] = None, **kwargs
) -> list[ProjectResource]:

    positions_filepath = path / "trackPositions.bin"
    match_ranges_filepath = path / "trackMatchRanges.bin"
    matches_filepath = path / "trackMatches.bin"

    _write_attribute(positions_filepath, tracks.positions, _Attribute.POSITIONS)
    _write_attribute(
        match_ranges_filepath, tracks.match_ranges, _Attribute.MATCH_RANGES
    )
    _write_attribute_group(matches_filepath, tracks.matches, _AttributeGroup.MATCHES)

    return [
        ProjectResource(
            uri=to_uri_reference(positions_filepath, base_path),
            format=UnknownFormat(TrackFormat.POSITIONS),
        ),
        ProjectResource(
            uri=to_uri_reference(match_ranges_filepath, base_path),
            format=UnknownFormat(TrackFormat.MATCH_RANGES),
        ),
        ProjectResource(
            uri=to_uri_reference(matches_filepath, base_path),
            format=UnknownFormat(TrackFormat.MATCHES),
        ),
    ]


def _loader(*args):

    if len(args) == 0:
        # We get here when load is called with a track binary file resource which is not
        # the positions
        return None
    else:
        return Tracks.open(*args)


def _test_function(
    resource: str | ProjectResource,
    base_uri: str,
    additional_resources: list[ProjectResource],
) -> tuple[bool, Optional[list[Path]]]:

    # For simplicity this function doesn't try to make guesses and only accepts ProjectResource
    if not isinstance(resource, ProjectResource):
        return (False, None)

    if resource.format == TrackFormat.POSITIONS:
        params = [join_uris(resource.uri, base_uri)]
        for resource in additional_resources:
            if resource.format == TrackFormat.MATCH_RANGES:
                params.append(join_uris(resource.uri, base_uri))
        # We iterate twice because we want the ranges to appear first
        for resource in additional_resources:
            if resource.format == TrackFormat.MATCHES:
                params.append(join_uris(resource.uri, base_uri))

        return (True, params)

    elif resource.format == TrackFormat.MATCH_RANGES:
        return (True, [])

    elif resource.format == TrackFormat.MATCHES:
        return (True, [])

    return (False, None)


loaders.append((_test_function, _loader))
savers.append((Tracks, _saver))
