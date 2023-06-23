from typing import Any, List

import numpy as np

from ..formats import CoreFormat
from ..items import CoreItem
from ..types import OpfObject, VersionInfo
from ..util import from_list, from_str, to_class, to_float, vector_from_list
from ..versions import FormatVersion, format_and_version_to_type


class CalibratedControlPoint(OpfObject):
    """Optimized 3D position in the processing CRS."""

    coordinates: np.ndarray
    id: str
    """A string identifier that matches the corresponding input control point."""

    def __init__(self, id: str, coordinates: np.ndarray) -> None:
        super(CalibratedControlPoint, self).__init__()
        self.id = id
        self.coordinates = coordinates

    @staticmethod
    def from_dict(obj: Any) -> "CalibratedControlPoint":
        assert isinstance(obj, dict)
        coordinates = vector_from_list(obj.get("coordinates"), 3, 3)
        id = from_str(obj.get("id"))
        result = CalibratedControlPoint(id, coordinates)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(CalibratedControlPoint, self).to_dict()
        result["id"] = str(self.id)
        result["coordinates"] = from_list(to_float, self.coordinates)
        return result


class CalibratedControlPoints(CoreItem):
    """Definition of calibrated control points, which are the optimised control points with
    coordinates expressed in the processing CRS.
    """

    points: List[CalibratedControlPoint]
    """List of calibrated control points."""

    def __init__(
        self,
        points: List[CalibratedControlPoint],
        format: CoreFormat = CoreFormat.CALIBRATED_CONTROL_POINTS,
        version: VersionInfo = FormatVersion.CALIBRATED_CONTROL_POINTS,
    ) -> None:
        super().__init__(format=format, version=version)

        assert self.format == CoreFormat.CALIBRATED_CONTROL_POINTS

        self.points = points

    @staticmethod
    def from_dict(obj: Any) -> "CalibratedControlPoints":
        base = CoreItem.from_dict(obj)
        points = from_list(CalibratedControlPoint.from_dict, obj.get("points"))
        result = CalibratedControlPoints(points, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(CalibratedControlPoints, self).to_dict()
        result["points"] = from_list(
            lambda x: to_class(CalibratedControlPoint, x), self.points
        )
        return result


format_and_version_to_type[
    (CoreFormat.CALIBRATED_CONTROL_POINTS, FormatVersion.CALIBRATED_CONTROL_POINTS)
] = CalibratedControlPoints
