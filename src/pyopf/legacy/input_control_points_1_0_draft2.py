from typing import Any, Dict, List, Optional

import numpy as np

from ..crs import Geolocation
from ..formats import CoreFormat
from ..items import BaseItem
from ..types import OpfObject, VersionInfo
from ..uid64 import Uid64
from ..util import (
    from_bool,
    from_float,
    from_list,
    from_none,
    from_str,
    from_union,
    to_class,
    to_float,
    vector_from_list,
)
from ..versions import format_and_version_to_type


class Mark(OpfObject):
    accuracy: float
    camera_id: Uid64
    position_px: np.ndarray  # vector of size 2

    def __init__(
        self,
        accuracy: float,
        camera_id: str,
        position_px: np.ndarray,
    ) -> None:
        super(Mark, self).__init__()
        self.accuracy = accuracy
        self.camera_id = Uid64(hex=camera_id)
        self.position_px = position_px

    @staticmethod
    def from_dict(obj: Any) -> "Mark":
        assert isinstance(obj, dict)
        accuracy = from_float(obj.get("accuracy"))
        camera_id = Uid64(hex=obj.get("camera_id"))
        position_px = vector_from_list(obj.get("position_px"), 2, 2)
        result = Mark(accuracy, camera_id.hex, position_px)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Mark, self).to_dict()
        result["accuracy"] = to_float(self.accuracy)
        result["camera_id"] = str(self.camera_id)
        result["position_px"] = from_list(to_float, self.position_px)
        return result


class Gcp(OpfObject):
    geolocation: Geolocation
    id: str
    is_checkpoint: bool
    marks: List[Mark]

    def __init__(
        self,
        id: str,
        geolocation: Geolocation,
        is_checkpoint: bool,
        marks: List[Mark],
    ) -> None:
        super(Gcp, self).__init__()
        self.geolocation = geolocation
        self.id = id
        self.is_checkpoint = is_checkpoint
        self.marks = marks

    @staticmethod
    def from_dict(obj: Any) -> "Gcp":
        assert isinstance(obj, dict)
        geolocation = Geolocation.from_dict(obj.get("geolocation"))
        id = from_str(obj.get("id"))
        is_checkpoint = from_bool(obj.get("is_checkpoint"))
        marks = from_list(Mark.from_dict, obj.get("marks"))
        result = Gcp(id, geolocation, is_checkpoint, marks)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Gcp, self).to_dict()
        result["geolocation"] = to_class(Geolocation, self.geolocation)
        result["id"] = from_str(self.id)
        result["is_checkpoint"] = from_bool(self.is_checkpoint)
        result["marks"] = from_list(lambda x: to_class(Mark, x), self.marks)
        return result


class Mtp(OpfObject):
    id: str
    is_checkpoint: bool
    marks: List[Mark]

    def __init__(
        self,
        id: str,
        is_checkpoint: bool,
        marks: List[Mark],
    ) -> None:
        super(Mtp, self).__init__()
        self.id = id
        self.is_checkpoint = is_checkpoint
        self.marks = marks

    @staticmethod
    def from_dict(obj: Any) -> "Mtp":
        assert isinstance(obj, dict)
        id = from_str(obj.get("id"))
        is_checkpoint = from_bool(obj.get("is_checkpoint"))
        marks = from_list(Mark.from_dict, obj.get("marks"))
        result = Mtp(id, is_checkpoint, marks)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Mtp, self).to_dict()
        result["id"] = from_str(self.id)
        result["is_checkpoint"] = from_bool(self.is_checkpoint)
        result["marks"] = from_list(lambda x: to_class(Mark, x), self.marks)
        return result


class InputControlPoints(BaseItem):
    gcps: List[Gcp]
    mtps: List[Mtp]

    def __init__(
        self,
        gcps: List[Gcp],
        mtps: List[Mtp],
        format: CoreFormat = CoreFormat.INPUT_CONTROL_POINTS,
        version: VersionInfo = VersionInfo(1, 0, "draft2"),
    ) -> None:
        super().__init__(format=format, version=version)

        assert self.format == CoreFormat.INPUT_CONTROL_POINTS
        self.gcps = gcps
        self.mtps = mtps

    @staticmethod
    def from_dict(obj: Any) -> "InputControlPoints":
        base = BaseItem.from_dict(obj)
        gcps = from_list(Gcp.from_dict, obj.get("gcps"))
        mtps = from_list(Mtp.from_dict, obj.get("mtps"))
        result = InputControlPoints(gcps, mtps, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(InputControlPoints, self).to_dict()
        result["gcps"] = from_list(lambda x: to_class(Gcp, x), self.gcps)
        result["mtps"] = from_list(lambda x: to_class(Mtp, x), self.mtps)
        return result


format_and_version_to_type[
    (CoreFormat.INPUT_CONTROL_POINTS, VersionInfo(1, 0, "draft2"))
] = InputControlPoints
