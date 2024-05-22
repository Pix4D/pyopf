from typing import Any, List

from ..cps import CalibratedControlPoint
from ..formats import ExtensionFormat
from ..items import ExtensionItem
from ..util import from_list, to_class
from ..versions import VersionInfo, format_and_version_to_type

format = ExtensionFormat(
    "application/ext-pix4d-calibrated-intersection-tie-points+json"
)
version = VersionInfo(1, 0, "draft2")


class Pix4DCalibratedIntersectionTiePoints(ExtensionItem):
    """Definition of calibrated intersection tie points, which are the optimised intersection
    tie points with coordinates expressed in the processing CRS.
    """

    """List of calibrated intersection tie points."""
    points: List[CalibratedControlPoint]

    def __init__(
        self,
        points: List[CalibratedControlPoint],
        format_: ExtensionFormat = format,
        version_: VersionInfo = version,
    ) -> None:
        super(Pix4DCalibratedIntersectionTiePoints, self).__init__(
            format=format_, version=version_
        )

        assert self.format == format
        self.points = points

    @staticmethod
    def from_dict(obj: Any) -> "Pix4DCalibratedIntersectionTiePoints":
        base = ExtensionItem.from_dict(obj)
        points = from_list(CalibratedControlPoint.from_dict, obj["points"])
        result = Pix4DCalibratedIntersectionTiePoints(points, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Pix4DCalibratedIntersectionTiePoints, self).to_dict()
        result["points"] = from_list(
            lambda x: to_class(CalibratedControlPoint, x), self.points
        )
        return result


format_and_version_to_type[(format, version)] = Pix4DCalibratedIntersectionTiePoints
