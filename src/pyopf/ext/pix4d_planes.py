from typing import Any, List, Optional

import numpy as np

from ..formats import ExtensionFormat
from ..items import ExtensionItem
from ..uid64 import Uid64
from ..util import IntType, from_bool, from_list, to_class
from ..versions import VersionInfo, format_and_version_to_type
from .plane import Plane

format = ExtensionFormat("application/ext-pix4d-planes+json")
version = VersionInfo(1, 0, "draft2")


class ExtendedPlane(Plane):

    is_plane_oriented: bool
    """If True, indicates that the normal vector points towards the visible half-space defined by the plane.
       Otherwise, the normal can point in either direction."""

    viewing_cameras: List[Uid64]
    """List of camera ids from the input cameras which are known to view the plane or part of it."""

    def __init__(
        self,
        vertices3d: List[np.ndarray],
        normal_vector: np.ndarray,
        outer_boundary: List[IntType],
        is_plane_oriented: bool,
        viewing_cameras: List[Uid64],
        inner_boundaries: Optional[List[List[IntType]]] = None,
    ) -> None:
        super(ExtendedPlane, self).__init__(
            vertices3d, normal_vector, outer_boundary, inner_boundaries
        )
        self.is_plane_oriented = is_plane_oriented
        self.viewing_cameras = viewing_cameras

    @staticmethod
    def from_dict(obj: Any) -> "ExtendedPlane":
        assert isinstance(obj, dict)
        plane = Plane.from_dict(obj)
        is_plane_oriented = from_bool(obj.get("is_plane_oriented"))
        viewing_cameras = from_list(lambda x: Uid64(int=x), obj.get("viewing_cameras"))

        result = ExtendedPlane(
            plane.vertices3d,
            plane.normal_vector,
            plane.outer_boundary,
            is_plane_oriented,
            viewing_cameras,
            plane.inner_boundaries,
        )
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(ExtendedPlane, self).to_dict()
        result["is_plane_oriented"] = self.is_plane_oriented
        result["viewing_cameras"] = [int(x) for x in self.viewing_cameras]
        return result


class Pix4dPlanes(ExtensionItem):

    planes: List[ExtendedPlane]

    def __init__(
        self,
        planes: List[ExtendedPlane],
        pformat: ExtensionFormat = format,
        version: VersionInfo = version,
    ) -> None:
        super(Pix4dPlanes, self).__init__(format=pformat, version=version)
        assert self.format == format
        self.planes = planes

    @staticmethod
    def from_dict(obj: Any) -> "Pix4dPlanes":
        assert isinstance(obj, dict)
        base = ExtensionItem.from_dict(obj)
        planes = from_list(ExtendedPlane.from_dict, obj.get("planes"))
        result = Pix4dPlanes(planes, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result: dict = super(Pix4dPlanes, self).to_dict()

        result["planes"] = from_list(lambda x: to_class(ExtendedPlane, x), self.planes)

        return result


format_and_version_to_type[(format, version)] = Pix4dPlanes
