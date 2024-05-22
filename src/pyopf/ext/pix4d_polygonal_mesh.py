from typing import Any, List, Optional

import numpy as np

from pyopf.types import OpfObject

from ..formats import ExtensionFormat
from ..items import ExtensionItem
from ..uid64 import Uid64
from ..util import (
    IntType,
    from_int,
    from_list,
    from_none,
    from_uid,
    from_union,
    to_class,
    to_float,
    vector_from_list,
)
from ..versions import VersionInfo, format_and_version_to_type

format = ExtensionFormat("application/ext-pix4d-polygonal-meshes+json")
version = VersionInfo(1, 0, "draft1")


class VertexMark(OpfObject):

    camera_uid: Uid64
    """ Camera id corresponding to this mark """
    position_px: np.ndarray
    """ The position of the mark inside the image """

    def __init__(self, camera_uid: Uid64, position_px: np.ndarray) -> None:
        super(VertexMark, self).__init__()
        self.camera_uid = camera_uid
        self.position_px = position_px

    @staticmethod
    def from_dict(obj: Any) -> "VertexMark":
        assert isinstance(obj, dict)
        camera_uid = from_uid(obj["camera_uid"])
        position_px = vector_from_list(obj["position_px"], 2, 2)
        result = VertexMark(camera_uid, position_px)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result: dict = {}
        result["camera_uid"] = self.camera_uid.int
        result["position_px"] = from_list(lambda x: x, self.position_px)
        return result


class Vertex(OpfObject):

    position: np.ndarray
    """ The vertex position """
    marks: Optional[List[VertexMark]]
    """ The image marks corresponding to this vertex """

    def __init__(
        self, position: np.ndarray, marks: Optional[List[VertexMark]] = None
    ) -> None:
        super(Vertex, self).__init__()
        self.position = position
        self.marks = marks

    @staticmethod
    def from_dict(obj: Any) -> "Vertex":
        assert isinstance(obj, dict)
        position = vector_from_list(obj["position"], 3, 3)
        marks = from_union(
            [lambda x: from_list(lambda x: VertexMark.from_dict(x), x), from_none],
            obj.get("marks"),
        )
        result = Vertex(position, marks)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result: dict = {}
        result["position"] = from_list(to_float, self.position)
        if self.marks is not None:
            result["marks"] = from_union(
                [lambda x: from_list(lambda x: to_class(VertexMark, x), x), from_none],
                self.marks,
            )
        return result


class EdgeMark(OpfObject):

    camera_uid: Uid64
    """ Camera id corresponding to this mark """
    segment_px: List[np.ndarray]
    """ The position of the mark inside the image """

    def __init__(self, camera_uid: Uid64, segment_px: List[np.ndarray]) -> None:
        super(EdgeMark, self).__init__()
        self.camera_uid = camera_uid
        self.segment_px = segment_px

    def __eq__(self, other):
        return (
            self.camera_uid == other.camera_uid and self.segment_px == other.segment_px
        )

    @staticmethod
    def from_dict(obj: Any) -> "EdgeMark":
        assert isinstance(obj, dict)
        camera_uid = from_uid(obj["camera_uid"])
        segment_px = from_list(lambda x: vector_from_list(x, 2, 2), obj["segment_px"])
        result = EdgeMark(camera_uid, segment_px)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result: dict = {}
        result["camera_uid"] = self.camera_uid.int
        result["segment_px"] = from_list(
            lambda x: from_list(lambda x: x, x), self.segment_px
        )
        return result


class Edge(OpfObject):

    vertex_indices: List[IntType]
    """ The indices of the two vertices connected by the edge """
    marks: Optional[List[EdgeMark]]
    """ List of edge image marks """

    def __init__(
        self, vertex_indices: List[IntType], marks: Optional[List[EdgeMark]] = None
    ) -> None:
        super(Edge, self).__init__()
        self.vertex_indices = vertex_indices
        self.marks = marks

    def __eq__(self, other):
        return (
            set(self.vertex_indices) == set(other.vertex_indices)
            and self.marks == other.marks
        )

    @staticmethod
    def from_dict(obj: Any) -> "Edge":
        assert isinstance(obj, dict)
        vertex_indices = from_list(int, obj["vertex_indices"])
        marks = from_union(
            [lambda x: from_list(lambda x: EdgeMark.from_dict(x), x), from_none],
            obj.get("marks"),
        )
        result = Edge(vertex_indices, marks)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result: dict = {}
        result["vertex_indices"] = from_list(from_int, self.vertex_indices)
        if self.marks is not None:
            result["marks"] = from_union(
                [lambda x: from_list(lambda x: to_class(EdgeMark, x), x), from_none],
                self.marks,
            )
        return result


class Face(OpfObject):

    outer_edge_indices: List[IntType]
    """ The indices of the (undirected) edges forming the polygonal face outer loop """
    inner_edge_indices: Optional[List[List[IntType]]]
    """ The indices of the (undirected) edges forming the polygonal face inner loops, if any """

    def __init__(self, outer_edge_indices, inner_edge_indices=None):
        super(Face, self).__init__()
        self.outer_edge_indices = outer_edge_indices
        self.inner_edge_indices = inner_edge_indices

    @staticmethod
    def from_dict(obj: Any) -> "Face":
        assert isinstance(obj, dict)
        outer_edge_indices = from_list(from_int, obj["outer_edge_indices"])
        inner_edge_indices = from_union(
            [lambda x: from_list(lambda x: from_list(from_int, x), x), from_none],
            obj.get("inner_edge_indices"),
        )
        result = Face(outer_edge_indices, inner_edge_indices)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result: dict = {}
        if self.inner_edge_indices is not None:
            result["inner_edge_indices"] = from_union(
                [lambda x: from_list(lambda x: from_list(from_int, x), x), from_none],
                self.inner_edge_indices,
            )
        result["outer_edge_indices"] = from_list(from_int, self.outer_edge_indices)
        return result


class PolygonalMesh(OpfObject):

    vertices: List[Vertex]
    """ List of vertices """
    edges: List[Edge]
    """ List of edges """
    faces: List[Face]
    """ List of faces """
    triangulation: Optional[List[np.ndarray]]
    """ Array of vertex indices triplets defining a triangulation of the polygonal mesh """

    def __init__(
        self,
        vertices: List[Vertex],
        edges: List[Edge],
        faces: List[Face],
        triangulation: Optional[List[np.ndarray]] = None,
    ) -> None:
        super(PolygonalMesh, self).__init__()
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.triangulation = triangulation

    @staticmethod
    def from_dict(obj: Any) -> "PolygonalMesh":
        assert isinstance(obj, dict)
        vertices = from_list(lambda x: Vertex.from_dict(x), obj["vertices"])
        edges = from_list(lambda x: Edge.from_dict(x), obj["edges"])
        faces = from_list(lambda x: Face.from_dict(x), obj["faces"])
        triangulation = from_union(
            [
                lambda x: from_list(lambda x: vector_from_list(x, 3, 3, int), x),
                from_none,
            ],
            obj.get("triangulation"),
        )

        result = PolygonalMesh(vertices, edges, faces, triangulation)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = {}
        result["vertices"] = from_list(lambda x: to_class(Vertex, x), self.vertices)
        result["edges"] = from_list(lambda x: to_class(Edge, x), self.edges)
        result["faces"] = from_list(lambda x: to_class(Face, x), self.faces)
        if self.triangulation is not None:
            result["triangulation"] = from_union(
                [lambda x: from_list(lambda x: from_list(from_int, x), x), from_none],
                self.triangulation,
            )
        return result


class Pix4DPolygonalMeshes(ExtensionItem):

    meshes: List[PolygonalMesh]
    """ A list of meshes """

    def __init__(
        self,
        meshes: List[PolygonalMesh],
        pformat: ExtensionFormat = format,
        version: VersionInfo = version,
    ) -> None:
        super(Pix4DPolygonalMeshes, self).__init__(format=pformat, version=version)
        assert self.format == format
        self.meshes = meshes

    @staticmethod
    def from_dict(obj: Any) -> "Pix4DPolygonalMeshes":
        base = ExtensionItem.from_dict(obj)
        meshes = from_list(PolygonalMesh.from_dict, obj["meshes"])
        result = Pix4DPolygonalMeshes(meshes, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Pix4DPolygonalMeshes, self).to_dict()
        result["meshes"] = from_list(lambda x: to_class(PolygonalMesh, x), self.meshes)
        return result


format_and_version_to_type[(format, version)] = Pix4DPolygonalMeshes
