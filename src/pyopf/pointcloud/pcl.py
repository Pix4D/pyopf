import os
from dataclasses import fields
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import pygltflib
from pygltflib import GLTF2

from pyopf.uid64 import Uid64

from ..types import CoreFormat
from ..versions import FormatVersion, VersionInfo
from .utils import (
    Buffer,
    add_accessor,
    add_buffers,
    gl_to_numpy_shape,
    gl_to_numpy_type,
    write_buffers,
)

opf_axis_rotation_matrix = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
)
opf_axis_rotation_matrix_inverse = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
)


class PointIndexRanges:
    """A wrapper over the packed pointIndexRanges array used by the OPF_mesh_primitive_matches extension.

    The index ranges are represented by a pair of a 5 byte unsigned int for the offset and a 3 byte unsigned int for
    the count, which are packed and stored as 2 unsigned int values of 4 bytes.

    The raw data can be accessed by the `ranges` member, while the __getitem__ and __setitem__ functions can be used
    for packing/unpacking the data.
    """

    def __init__(self, ranges: np.ndarray | np.memmap):
        """Construct a PointIndexRanges wrapper over the raw data.
        :param ranges: The point index range information, stored as an array of shape (num_points, 2).
        :raise ValueError: If ranges does not have two columns
        """
        if ranges.shape[1] != 2:
            raise ValueError("The ranges array should have two columns")
        self.ranges = ranges
        self.nbytes = self.ranges.nbytes

    def __getitem__(self, index: int) -> tuple[int, int]:
        a = self.ranges[index, 0]
        b = self.ranges[index, 1]
        allbytes = a.tobytes() + b.tobytes()
        offset = int.from_bytes(allbytes[:5], "little")
        count = int.from_bytes(allbytes[5:], "little")
        return (offset, count)

    def __setitem__(self, index: int, item: tuple[int, int]):
        offset, count = item
        allbytes = offset.to_bytes(5, "little") + count.to_bytes(3, "little")
        a = int.from_bytes(allbytes[:4], "little")
        b = int.from_bytes(allbytes[4:], "little")
        self.ranges[index, 0] = a
        self.ranges[index, 1] = b

    def __len__(self) -> int:
        return self.ranges.shape[0]

    def total_count(self) -> int:
        return sum(count for _, count in self)


class ImagePoints:
    """Image Points, used by the OPF_mesh_primitive_matches extension"""

    featureIds: np.ndarray | np.memmap
    scales: np.ndarray | np.memmap
    pixelCoordinates: np.ndarray | np.memmap
    depths: Optional[np.ndarray | np.memmap]

    def __init__(self):
        self.depths = None
        self.featureIds = np.array([])
        self.pixelCoordinates = np.array([])
        self.scales = np.array([])

    @staticmethod
    def from_gltf(image_points_extension: dict, accessors: list):
        points = ImagePoints()
        if "depths" in image_points_extension:
            points.depths = accessors[image_points_extension["depths"]]
        else:
            points.depths = None
        points.featureIds = accessors[image_points_extension["featureIds"]]
        points.pixelCoordinates = accessors[image_points_extension["pixelCoordinates"]]
        points.scales = accessors[image_points_extension["scales"]]

        return points

    @property
    def buffer_filepaths(self) -> list[str | None]:
        """Return a list of absolute file paths to the memory mapped arrays"""

        if (
            not hasattr(self.featureIds, "filename")
            or not hasattr(self.pixelCoordinates, "filename")
            or not hasattr(self.scales, "filename")
            or not (self.depths is None or hasattr(self.depths, "filename"))
        ):
            raise ValueError("The image points have not been saved to disk")

        buffer_filenames = [
            self.featureIds.filename,  # type: ignore
            self.pixelCoordinates.filename,  # type: ignore
            self.scales.filename,  # type: ignore
        ]
        if self.depths is not None:
            buffer_filenames.append(self.depths.filename)  # type: ignore

        return buffer_filenames  # type: ignore

    def flush(self) -> None:
        """Write to disk any changes to the binary buffers"""

        if (
            not hasattr(self.featureIds, "flush")
            or not hasattr(self.pixelCoordinates, "flush")
            or not hasattr(self.scales, "flush")
            or not (self.depths is None or hasattr(self.depths, "flush"))
        ):
            raise ValueError("The image points have not been saved to disk")

        self.featureIds.flush()  # type: ignore
        self.pixelCoordinates.flush()  # type: ignore
        self.scales.flush()  # type: ignore

        if self.depths is not None:
            self.depths.flush()  # type: ignore

    def write(
        self, gltf: pygltflib.GLTF2, buffers: dict[Path, Buffer], output_gltf_dir: Path
    ) -> dict:
        """Adds the required accessors to the GLTF2 object (also creating the corresponding files)
        :param gltf: The gltf object to add the accessors to
        :param buffers: A dictionary mapping file paths with buffer objects
        :param output_gltf_dir: The output directory for the binary buffers
        :return: A dictionary with the accessor ids for the class members
        """
        gltf_image_points = {
            "pixelCoordinates": add_accessor(
                gltf,
                buffers,
                self.pixelCoordinates,
                output_gltf_dir / "matchPixelCoordinates.bin",
            ),
            "featureIds": add_accessor(
                gltf, buffers, self.featureIds, output_gltf_dir / "matchFeatureIds.bin"
            ),
            "scales": add_accessor(
                gltf, buffers, self.scales, output_gltf_dir / "matchScales.bin"
            ),
        }
        if self.depths is not None:
            gltf_image_points["depths"] = add_accessor(
                gltf, buffers, self.depths, output_gltf_dir / "matchDepths.bin"
            )

        return gltf_image_points


class Matches:
    """Used by the OPF_mesh_primitive_matches extension"""

    camera_uids: list[Uid64]
    camera_ids: np.memmap | np.ndarray
    point_index_ranges: PointIndexRanges
    image_points: Optional[ImagePoints]

    def __init__(self):
        self.camera_uids = []
        self.camera_ids = np.ndarray([])
        self.point_index_ranges = PointIndexRanges(np.array([]).reshape(-1, 2))
        self.image_points = None

    @staticmethod
    def from_gltf(extension: dict, accessors: list, version: VersionInfo):
        """
        Construct the matches object from glTF extension OPF_mesh_primitive_matches

        :param extension: The extension data
        :param accessors: The list of all accesors available - this class will keep referencing the ones it needs
        """
        matches = Matches()
        if (
            version > VersionInfo(1, 0, "draft7")
            and version <= FormatVersion.GLTF_OPF_ASSET
        ):
            matches.camera_uids = [Uid64(int=id) for id in extension["cameraUids"]]
        elif version <= VersionInfo(1, 0, "draft7"):
            matches.camera_uids = [Uid64(hex=id) for id in extension["cameraUids"]]
        else:
            raise ValueError(f"Unsupported OPF glTF version: {version}")

        matches.camera_ids = accessors[extension["cameraIds"]]
        matches.point_index_ranges = PointIndexRanges(
            accessors[extension["pointIndexRanges"]]
        )

        matches.image_points = (
            ImagePoints.from_gltf(extension["imagePoints"], accessors)
            if "imagePoints" in extension
            else None
        )
        return matches

    @property
    def buffer_filepaths(self) -> list[Path]:
        """Return a list of absolute file paths to the memory mapped arrays"""

        if not hasattr(self.camera_ids, "filename") or not hasattr(
            self.point_index_ranges.ranges, "filename"
        ):
            raise ValueError("The matches are not stored in a memory mapped array.")

        buffer_filenames = [
            self.camera_ids.filename,  # type: ignore
            self.point_index_ranges.ranges.filename,  # type: ignore
        ]

        if self.image_points is not None:
            buffer_filenames.extend(self.image_points.buffer_filepaths)

        return buffer_filenames  # type: ignore

    def flush(self) -> None:
        """Write to disk any changes to the binary buffers"""

        if not hasattr(self.camera_ids, "flush") or not hasattr(
            self.point_index_ranges.ranges, "flush"
        ):
            raise ValueError("The matches are not stored in a memory mapped array.")

        self.camera_ids.flush()  # type: ignore
        self.point_index_ranges.ranges.flush()  # type: ignore

        if self.image_points is not None:
            self.image_points.flush()

    def write(
        self,
        gltf: pygltflib.GLTF2,
        buffers: dict[Path, Buffer],
        output_gltf_dir: Path,
        version: VersionInfo,
    ) -> dict:
        """Adds accessors for this object's data to a glTF object and saves the corresponding buffers to files.
        :param gltf: The gltf object to add the accessors to
        :param buffers: A dictionary mapping file paths with buffer objects
        :param output_gltf_dir: The output directory for the binary buffers
        :return: A dictionary corresponding to the OPF gltf specification for the matches extension
        """
        opf_mesh_primitive_matches = {
            "cameraIds": add_accessor(
                gltf, buffers, self.camera_ids, output_gltf_dir / "matchCameraIds.bin"
            ),
            "pointIndexRanges": add_accessor(
                gltf,
                buffers,
                self.point_index_ranges.ranges,
                output_gltf_dir / "matchPointIndexRanges.bin",
            ),
        }
        if (
            version > VersionInfo(1, 0, "draft7")
            and version <= FormatVersion.GLTF_OPF_ASSET
        ):
            opf_mesh_primitive_matches["cameraUids"] = [
                int(id.int) for id in self.camera_uids
            ]
        elif version <= VersionInfo(1, 0, "draft7"):
            opf_mesh_primitive_matches["cameraUids"] = [
                str(id) for id in self.camera_uids
            ]
        else:
            raise ValueError(f"Unsupported OPF glTF version: {version}")

        if self.image_points:
            opf_mesh_primitive_matches["imagePoints"] = self.image_points.write(
                gltf, buffers, output_gltf_dir
            )

        return opf_mesh_primitive_matches


class Node:
    """A glTF node"""

    position: np.ndarray | np.memmap
    normal: np.ndarray | np.memmap | None
    color: np.ndarray | np.memmap | None
    matches: Matches | None
    matrix: np.ndarray | None
    custom_attributes: dict[str, np.ndarray | np.memmap] | None

    def __init__(self):
        self.position = np.zeros((0, 3))
        self.color = None
        self.normal = None
        self.matches = None
        self.custom_attributes = None
        self.matrix = None

    @staticmethod
    def from_gltf(
        node_id: int,
        gltf: GLTF2,
        accessors: list[np.ndarray] | list[np.memmap],
        version: VersionInfo,
    ) -> "Node":
        """Construct an object representing the glTF node with id node_id.
        The node id must be a valid node id.

        :param node_id: The id of the node to construct
        :param gltf: The glTF object
        :param accessors: A list of arrays representing the data of the accessors, sharing the same indices
        """

        mesh = gltf.nodes[node_id].mesh

        if mesh is None:
            raise ValueError("The node must have a mesh.")

        primitive = gltf.meshes[mesh].primitives[0]

        node = Node()

        if primitive.attributes.POSITION is None:
            raise ValueError("The mesh must have a position attribute.")

        node.position = accessors[primitive.attributes.POSITION]
        node.color = (
            accessors[primitive.attributes.COLOR_0]
            if primitive.attributes.COLOR_0
            else None
        )
        node.normal = (
            accessors[primitive.attributes.NORMAL]
            if primitive.attributes.NORMAL
            else None
        )

        node.matches = None
        if (
            primitive.extensions is not None
            and "OPF_mesh_primitive_matches" in primitive.extensions
        ):
            node.matches = Matches.from_gltf(
                primitive.extensions["OPF_mesh_primitive_matches"], accessors, version
            )

        node.custom_attributes = None
        if (
            primitive.extensions is not None
            and "OPF_mesh_primitive_custom_attributes" in primitive.extensions
        ):
            node.custom_attributes = {}
            for attribute_name, accessor_id in primitive.extensions[
                "OPF_mesh_primitive_custom_attributes"
            ]["attributes"].items():
                node.custom_attributes[attribute_name] = accessors[accessor_id]

        node.matrix = np.array(gltf.nodes[node_id].matrix)
        if node.matrix is not None:
            node.matrix = np.array(node.matrix).reshape((4, 4), order="F")

        return node

    def __len__(self):
        """The number of points in the node"""
        return self.position.shape[0]

    @property
    def buffer_filepaths(self) -> list[Path]:
        """Return a list of absolute file paths to the memory mapped arrays"""

        if not hasattr(self.position, "filename"):
            raise ValueError("The node is not stored in a memory mapped array.")

        buffer_filenames = [self.position.filename]  # type: ignore

        if self.color is not None:
            buffer_filenames.append(self.color.filename)  # type: ignore
        if self.normal is not None:
            buffer_filenames.append(self.normal.filename)  # type: ignore
        if self.matches is not None:
            buffer_filenames.extend(self.matches.buffer_filepaths)  # type: ignore
        if self.custom_attributes is not None:
            for custom_attribute_buffer in self.custom_attributes.values():
                buffer_filenames.append(custom_attribute_buffer.filename)  # type: ignore

        return buffer_filenames  # type: ignore

    def flush(self) -> None:
        """Write to disk any changes to the binary buffers"""

        if not hasattr(self.position, "flush"):
            raise ValueError("The node is not stored in a memory mapped array.")

        self.position.flush()  # type: ignore

        if self.color is not None:
            self.color.flush()  # type: ignore
        if self.normal is not None:
            self.normal.flush()  # type: ignore
        if self.matches is not None:
            self.matches.flush()  # type: ignore
        if self.custom_attributes is not None:
            for custom_attribute_buffer in self.custom_attributes.values():
                custom_attribute_buffer.flush()  # type: ignore

    def write(
        self,
        gltf: pygltflib.GLTF2,
        buffers: dict[Path, Buffer],
        output_gltf_dir: Path,
        version: VersionInfo,
    ):
        """Adds the node to the GLTF2 object in-place and writes the associated binary buffers
        :param gltf: The GLTF2 object to add the node to
        :param buffers: A dictionary mapping file paths with buffer objects
        :param output_gltf_dir: Path where to write the binary buffers.
                                It is assumed that the *.gltf file will be written in the same place.
        """
        gltf.nodes.append(pygltflib.Node(mesh=len(gltf.meshes)))
        gltf.meshes.append(
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(),
                        indices=None,
                        material=0,
                        mode=pygltflib.POINTS,
                        extensions={},
                    )
                ]
            )
        )

        primitive = gltf.meshes[-1].primitives[0]

        primitive.attributes.POSITION = add_accessor(
            gltf, buffers, self.position, output_gltf_dir / "positions.bin"
        )
        gltf.accessors[-1].min = self.position.min(axis=0).tolist()
        gltf.accessors[-1].max = self.position.max(axis=0).tolist()

        if self.normal is not None:
            primitive.attributes.NORMAL = add_accessor(
                gltf, buffers, self.normal, output_gltf_dir / "normals.bin"
            )

        if self.color is not None:
            primitive.attributes.COLOR_0 = add_accessor(
                gltf, buffers, self.color, output_gltf_dir / "colors.bin"
            )
            gltf.accessors[-1].normalized = True

        if self.matches is not None:
            gltf.extensionsUsed.append("OPF_mesh_primitive_matches")
            primitive.extensions["OPF_mesh_primitive_matches"] = self.matches.write(gltf, buffers, output_gltf_dir, version)  # type: ignore

        if self.custom_attributes is not None:
            gltf.extensionsUsed.append("OPF_mesh_primitive_custom_attributes")
            attributes = {
                name: add_accessor(
                    gltf, buffers, value, output_gltf_dir / (name + ".bin")
                )
                for name, value in self.custom_attributes.items()
            }

            assert primitive.extensions is not None

            primitive.extensions["OPF_mesh_primitive_custom_attributes"] = {
                "attributes": attributes
            }

        if self.matrix is not None:
            gltf.nodes[-1].matrix = self.matrix.flatten(order="F").tolist()


class GlTFPointCloud:
    """Open glTF point cloud"""

    _format: CoreFormat
    _version: VersionInfo
    nodes: list[Node] = []
    metadata: Optional["Metadata"]  # noqa: F821 # type: ignore

    mode_type = Literal[
        "r", "c", "r+", "w+", "readonly", "copyonwrite", "readwrite", "write"
    ]

    @property
    def format(self):
        return self._format

    @property
    def version(self):
        return self._version

    def _open_accessors(
        self, gltf: GLTF2, base_dir: Path, mode: mode_type = "r"
    ) -> list[np.memmap]:
        """
        Utility to read the accessors of a glTF point cloud and return them as a list of numpy memmap arrays
        :gltf GLTF2: A GLTF2 object
        :base_dir Path: Base path where the glTF binary files are located
        :mode str: Open mode for the memory mapped arrays

        :return list[numpy.memmap]: A list of numpy memmap arrays, indexed by their corresponding glTF indices
        """

        accessors = []

        for accessor in gltf.accessors:
            accessor_buffer_view = accessor.bufferView

            if accessor_buffer_view is None:
                raise RuntimeError("Accessor is missing bufferView")

            buffer_view = gltf.bufferViews[accessor_buffer_view]
            buffer_uri = gltf.buffers[buffer_view.buffer].uri
            buffer_view_offset = buffer_view.byteOffset

            if buffer_uri is None:
                raise RuntimeError("Buffer is missing uri")
            if buffer_view_offset is None:
                raise RuntimeError("BufferView is missing byteOffset")

            new_accessor = np.memmap(
                base_dir / buffer_uri,
                mode=mode,
                dtype=gl_to_numpy_type(accessor.componentType),
                offset=buffer_view_offset,
                shape=(accessor.count, gl_to_numpy_shape(accessor.type)),
            )
            accessors.append(new_accessor)

        return accessors

    def __init__(self):
        self._format = CoreFormat.GLTF_MODEL
        self._version = FormatVersion.GLTF_OPF_ASSET

    @staticmethod
    def open(gltf_path: Path, mode: mode_type = "r"):
        """Read a point cloud object.
        The accessors are resolved immediately and opened as numpy memmory mapped arrays, with the appropriate mode.
        The gltf file is assumed to be valid and nonempty.

        :param gltf_path: The path to the *.gltf file
        :param mode: Read mode for the accessors
        """

        gltf = GLTF2().load(gltf_path)

        if gltf is None:
            raise RuntimeError("The glTF file %s could not be loaded" % gltf_path)

        pcl = GlTFPointCloud()

        if gltf.asset.extensions is None:
            raise RuntimeError("GlTF asset has no extensions")
        if (
            "OPF_asset_version" not in gltf.asset.extensions
            or "version" not in gltf.asset.extensions["OPF_asset_version"]
        ):
            raise RuntimeError("OPF_asset_version extension missing or incorrect")

        pcl._version = VersionInfo.parse(
            gltf.asset.extensions["OPF_asset_version"]["version"]
        )

        accessors = pcl._open_accessors(gltf, gltf_path.parent, mode)

        pcl.nodes = [
            Node.from_gltf(node_id, gltf, accessors, pcl._version)  # type: ignore
            for node_id in range(len(gltf.nodes))
        ]

        return pcl

    def __len__(self) -> int:
        """The number of nodes in the glTF Point Cloud"""
        return len(self.nodes)

    def is_readonly(self):
        """Check if the binary buffers are read only"""
        if len(self) == 0:
            raise RuntimeError("Empty point cloud")
        return not self.nodes[0].position.flags.writeable

    def flush(self) -> None:
        """Write to disk any changes to the binary buffers"""
        for node in self.nodes:
            node.flush()

    def write(self, output_gltf_file: Path, save_buffers=True):
        """Write the object as glTF point cloud. The binary buffers will be saved in the parent directory of the
        glTF file, overriding any existing ones.
        :param output_gltf_file: Path to the final*.gltf file. The binary buffers will be saved in the same directory.
        :param save_buffers: If true, the binary buffers are written to files. Otherwise, their location is kept.
        :return: A list of paths to the existing or new location of the binary buffers.
        :raise FileNotFoundError: If the parent directory of `output_gltf_file` does not exist.
        """

        buffers = {}
        output_gltf_dir = output_gltf_file.parent

        if not os.path.exists(output_gltf_dir):
            raise FileNotFoundError(
                "Output directory %s does not exist " % str(output_gltf_dir)
            )

        gltf = GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            materials=[
                pygltflib.Material(
                    emissiveFactor=None,
                    alphaMode=None,
                    alphaCutoff=None,
                    doubleSided=None,
                    extensions={"KHR_materials_unlit": {}},
                )
            ],
        )

        gltf.extensionsUsed = ["KHR_materials_unlit", "OPF_asset_version"]

        for node in self.nodes:
            node.write(gltf, buffers, output_gltf_dir, self._version)

        if save_buffers:
            write_buffers(buffers)

        add_buffers(gltf, buffers, output_gltf_dir)

        asset = pygltflib.Asset(
            version="2.0",
            extensions={"OPF_asset_version": {"version": str(self._version)}},
        )

        gltf.save(str(output_gltf_file), asset)

        if save_buffers:
            return [buffer.filepath for buffer in buffers.values()]
        else:
            return [path for node in self.nodes for path in node.buffer_filepaths]
