from typing import Any, List

import numpy as np

from ..formats import ExtensionFormat
from ..items import ExtensionItem
from ..types import OpfObject
from ..util import from_list, from_str, to_class, to_float, vector_from_list
from ..versions import VersionInfo, format_and_version_to_type
from .pix4d_input_intersection_tie_points import MarkWithSegments

format = ExtensionFormat(
    "application/ext-pix4d-calibrated-intersection-tie-points+json"
)
version = VersionInfo(1, 0, "draft3")


class CalibratedIntersectionTiePoint(OpfObject):
    id: str
    """A unique string that matches the input ITP."""
    coordinates: np.ndarray
    """Optimized 3D position in the processing CRS."""
    calibrated_marks: List[MarkWithSegments]
    """List of marks with line segments in the images that correspond to the projections of a 3D
    point.
    """

    def __init__(
        self, id: str, coordinates: np.ndarray, calibrated_marks: List[MarkWithSegments]
    ) -> None:
        self.id = id
        self.coordinates = coordinates
        self.calibrated_marks = calibrated_marks

    @staticmethod
    def from_dict(obj: Any) -> "CalibratedIntersectionTiePoint":
        assert isinstance(obj, dict)
        id = from_str(obj["id"])
        coordinates = vector_from_list(obj["coordinates"], 3, 3)
        calibrated_marks = (
            from_list(MarkWithSegments.from_dict, obj["calibrated_marks"])
            if "calibrated_marks" in obj
            else None
        )
        result = CalibratedIntersectionTiePoint(id, coordinates, calibrated_marks)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(CalibratedIntersectionTiePoint, self).to_dict()
        result["id"] = from_str(self.id)
        result["coordinates"] = from_list(to_float, self.coordinates)
        if self.calibrated_marks is not None:
            result["calibrated_marks"] = from_list(
                lambda x: to_class(MarkWithSegments, x), self.calibrated_marks
            )
        return result


class Pix4DCalibratedIntersectionTiePoints(ExtensionItem):
    """Definition of calibrated intersection tie points, which are the optimised intersection
    tie points with coordinates expressed in the processing CRS.
    """

    """List of calibrated intersection tie points."""
    points: List[CalibratedIntersectionTiePoint]

    def __init__(
        self,
        points: List[CalibratedIntersectionTiePoint],
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
        points = from_list(CalibratedIntersectionTiePoint.from_dict, obj["points"])
        result = Pix4DCalibratedIntersectionTiePoints(points, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Pix4DCalibratedIntersectionTiePoints, self).to_dict()
        result["points"] = from_list(
            lambda x: to_class(CalibratedIntersectionTiePoint, x), self.points
        )
        return result


format_and_version_to_type[(format, version)] = Pix4DCalibratedIntersectionTiePoints
# backward compatibility
format_and_version_to_type[
    (format, VersionInfo(1, 0, "draft2"))
] = Pix4DCalibratedIntersectionTiePoints
