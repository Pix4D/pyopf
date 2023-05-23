from typing import Any, Dict, List, Optional, Union

from ..items import ExtensionItem
from ..types import OpfPropertyExtObject
from ..uid64 import Uid64
from ..util import (
    from_float,
    from_none,
    from_str,
    from_union,
    from_version_info,
    to_class,
    to_float,
)
from ..versions import VersionInfo

_version = VersionInfo(1, 0, "draft2")


class DepthMapConfidence(OpfPropertyExtObject):
    """A confidence map indicates the level of confidence of the depth measurements. If present,
    it must be of the same dimension as the depth map. Valid confidence values range from a
    `min` (lowest confidence) to a `max` (highest confidence).
    """

    """The confidence map UID in the camera list."""
    id: Uid64
    """Maximum confidence value to consider a depth measurement valid."""
    max: float
    """Minimum confidence value to consider a depth measurement valid."""
    min: float

    def __init__(self, id: Uid64, max: float, min: float) -> None:
        super(DepthMapConfidence, self).__init__()
        self.id = id
        self.max = max
        self.min = min

    @staticmethod
    def from_dict(obj: Any) -> "DepthMapConfidence":
        assert isinstance(obj, dict)
        id = Uid64(int=obj.get("id"))
        max = from_float(obj.get("max"))
        min = from_float(obj.get("min"))
        result = DepthMapConfidence(id, max, min)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(DepthMapConfidence, self).to_dict()
        result["id"] = self.id.int
        result["max"] = to_float(self.max)
        result["min"] = to_float(self.min)
        return result


class Pix4dInputDepthMap(OpfPropertyExtObject):
    """Reference to depth information for input cameras, for example for RGB-D type sensors. In
    a depth map, each pixel represents the estimated distance from the device to its
    environment on the camera depth axis. A depth map image is aligned with an RGB image but
    it may have a different resolution. An optional confidence map may be provided as well.
    """

    """The depth map UID in the camera list."""
    id: Uid64
    """Multiply this scale factor with depth maps values in order to obtain values in meters.
    For example, if the depth map values represent millimeters the scale factor is 0.001
    (e.g. a value of 1000mm corresponds to 1m).  If not specified, defaults to 1.
    """
    unit_to_meters: Optional[float]
    confidence: Optional[DepthMapConfidence]
    extension_name = "PIX4D_input_depth_map"

    def __init__(
        self,
        id: Uid64,
        unit_to_meters: Optional[float],
        confidence: Optional[DepthMapConfidence],
        version=_version,
    ) -> None:
        self.id = id
        self.unit_to_meters = unit_to_meters
        self.confidence = confidence
        self.version = version

    @staticmethod
    def from_dict(obj: Any) -> "Pix4dInputDepthMap":
        assert isinstance(obj, dict)
        confidence = from_union(
            [DepthMapConfidence.from_dict, from_none], obj.get("confidence")
        )

        id = Uid64(obj.get("id"))
        unit_to_meters = from_union([from_float, from_none], obj.get("unit_to_meters"))
        version = from_union([from_version_info, VersionInfo.parse], obj.get("version"))
        result = Pix4dInputDepthMap(id, unit_to_meters, confidence, version)
        result._extract_unknown_properties_and_extensions(obj)

        return result

    def to_dict(self) -> dict:
        result: dict = {}
        if self.confidence is not None:
            result["confidence"] = from_union(
                [lambda x: to_class(DepthMapConfidence, x), from_none], self.confidence
            )
        result["id"] = self.id.int
        if self.unit_to_meters is not None:
            result["unit_to_meters"] = from_union(
                [to_float, from_none], self.unit_to_meters
            )
        result["version"] = str(self.version)
        return result
