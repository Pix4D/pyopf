from typing import Any, List, Optional

import numpy as np

from ..types import OpfObject
from ..util import (
    IntType,
    from_int,
    from_list,
    from_none,
    from_union,
    to_float,
    vector_from_list,
)


class Plane(OpfObject):
    """List of inner boundaries."""

    inner_boundaries: Optional[List[List[IntType]]]
    """Plane normal direction."""
    normal_vector: np.ndarray
    """List of indices in the 3D vertices array."""
    outer_boundary: List[IntType]
    """List of 3D vertices."""
    vertices3d: List[np.ndarray]

    def __init__(
        self,
        vertices3d: List[np.ndarray],
        normal_vector: np.ndarray,
        outer_boundary: List[IntType],
        inner_boundaries: Optional[List[List[IntType]]] = None,
    ) -> None:
        self.vertices3d = vertices3d
        self.normal_vector = normal_vector
        self.outer_boundary = outer_boundary
        self.inner_boundaries = inner_boundaries

    @staticmethod
    def from_dict(obj: Any) -> "Plane":
        assert isinstance(obj, dict)
        inner_boundaries = from_union(
            [lambda x: from_list(lambda x: from_list(from_int, x), x), from_none],
            obj.get("inner_boundaries"),
        )
        normal_vector = vector_from_list(obj.get("normal_vector"))
        outer_boundary = from_list(from_int, obj.get("outer_boundary"))
        vertices3d = from_list(lambda x: vector_from_list(x), obj.get("vertices3d"))
        return Plane(vertices3d, normal_vector, outer_boundary, inner_boundaries)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.inner_boundaries is not None:
            result["inner_boundaries"] = from_union(
                [lambda x: from_list(lambda x: from_list(from_int, x), x), from_none],
                self.inner_boundaries,
            )
        result["normal_vector"] = from_list(to_float, self.normal_vector)
        result["outer_boundary"] = from_list(from_int, self.outer_boundary)
        result["vertices3d"] = from_list(
            lambda x: from_list(to_float, x), self.vertices3d
        )
        return result
