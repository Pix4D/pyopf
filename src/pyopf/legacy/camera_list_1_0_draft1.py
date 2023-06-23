from typing import Any, List

from ..formats import CoreFormat
from ..items import BaseItem
from ..types import OpfObject, VersionInfo
from ..uid64 import Uid64
from ..util import from_list, from_str, to_class
from ..versions import FormatVersion, format_and_version_to_type


class CameraData(OpfObject):
    id: Uid64
    uri: str

    def __init__(self, id: Uid64, uri: str) -> None:
        super(CameraData, self).__init__()
        self.id = id
        self.uri = uri

    @staticmethod
    def from_dict(obj: Any) -> "CameraData":
        assert isinstance(obj, dict)
        id = Uid64(hex=obj.get("id"))
        uri = from_str(obj.get("uri"))
        result = CameraData(id, uri)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(CameraData, self).to_dict()
        result["id"] = str(self.id)
        result["uri"] = from_str(self.uri)
        return result


class CameraList(BaseItem):
    cameras: List[CameraData]

    def __init__(
        self,
        cameras: List[CameraData],
        format: CoreFormat = CoreFormat.CAMERA_LIST,
        version: VersionInfo = VersionInfo(1, 0, "draft1"),
    ) -> None:
        super(CameraList, self).__init__(format=format, version=version)

        assert self.format == CoreFormat.CAMERA_LIST
        self.cameras = cameras

    @staticmethod
    def from_dict(obj: Any) -> "CameraList":
        base = BaseItem.from_dict(obj)
        cameras = from_list(CameraData.from_dict, obj.get("cameras"))
        result = CameraList(cameras, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(CameraList, self).to_dict()
        result["cameras"] = from_list(lambda x: to_class(CameraData, x), self.cameras)
        return result


format_and_version_to_type[
    (CoreFormat.CAMERA_LIST, VersionInfo(1, 0, "draft1"))
] = CameraList
