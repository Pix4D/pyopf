from typing import Any, Optional

from ..formats import ExtensionFormat
from ..items import ExtensionItem
from ..util import from_float, from_none, from_union, to_class, to_float
from ..versions import VersionInfo, format_and_version_to_type
from .plane import Plane

format = ExtensionFormat("application/ext-pix4d-region-of-interest+json")
version = VersionInfo(1, 0, "draft1")


class Pix4DRegionOfInterest(ExtensionItem):

    """Definition of a region of interest: a planar polygon with holes and an optional
    height, defined as a the distance from the plane in the normal direction. All the
    points on the hemispace where the normal lies that project inside the polygon and is at a
    distance less than the height of the ROI, is considered to be within.
    """

    plane: Plane
    """The height of the ROI volume, defined as a limit distance from the plane in the normal
    direction. If not specified, the height is assumed to be infinite.
    """
    height: Optional[float]

    def __init__(
        self,
        plane: Plane,
        height: Optional[float],
        format: ExtensionFormat = format,
        version: VersionInfo = version,
    ) -> None:
        super(Pix4DRegionOfInterest, self).__init__(format=format, version=version)

        assert self.format == format
        self.plane = plane
        self.height = height

    @staticmethod
    def from_dict(obj: Any) -> "Pix4DRegionOfInterest":
        base = ExtensionItem.from_dict(obj)
        plane = Plane.from_dict(obj.get("plane"))
        height = from_union([from_float, from_none], obj.get("height"))
        result = Pix4DRegionOfInterest(plane, height, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Pix4DRegionOfInterest, self).to_dict()
        result["plane"] = to_class(Plane, self.plane)
        if self.height is not None:
            result["height"] = from_union([to_float, from_none], self.height)
        return result


format_and_version_to_type[(format, version)] = Pix4DRegionOfInterest
