from typing import Any, Optional

from .formats import (
    CoreFormat,
    ExtensionFormat,
    format_from_str,
    format_to_str,
    from_format,
)
from .types import OpfObject, OpfPropertyExtObject
from .util import from_union, from_version_info
from .VersionInfo import VersionInfo


class BaseItem(OpfObject):
    _format: CoreFormat  # This type is meant to be used only with core items
    _version: VersionInfo
    metadata: Optional["Metadata"]  # noqa: F821 # type: ignore

    @property
    def format(self):
        return self._format

    @property
    def version(self):
        return self._version

    def __init__(self, format: CoreFormat, version: VersionInfo):
        super(BaseItem, self).__init__()
        self._format = format
        self._version = version

    def to_dict(self) -> dict:
        result = super(BaseItem, self).to_dict()
        result.update(
            {"format": format_to_str(self.format), "version": str(self._version)}
        )
        return result

    @staticmethod
    def from_dict(obj: Any) -> "BaseItem":
        assert isinstance(obj, dict)
        format = from_union([from_format, format_from_str], obj.get("format"))
        version = from_union([from_version_info, VersionInfo.parse], obj.get("version"))
        return BaseItem(format, version)

    def _extract_unknown_properties_and_extensions(
        self, obj: dict, ignore_keys=set()
    ) -> None:
        super(BaseItem, self)._extract_unknown_properties_and_extensions(
            obj, ignore_keys={"format", "version"}.union(ignore_keys)
        )


class ExtensionItem(OpfObject):
    _format: ExtensionFormat  # This type is meant to be used only with extension items
    _version: VersionInfo
    metadata: Optional["Metadata"]  # noqa: F821 # type: ignore

    @property
    def format(self):
        return self._format

    @property
    def version(self):
        return self._version

    def __init__(self, format: ExtensionFormat, version: VersionInfo):
        super(ExtensionItem, self).__init__()
        self._format = format
        self._version = version

    def to_dict(self) -> dict:
        result = super(ExtensionItem, self).to_dict()
        result.update(
            {"format": format_to_str(self.format), "version": str(self._version)}
        )
        return result

    @staticmethod
    def from_dict(obj: Any) -> "ExtensionItem":
        assert isinstance(obj, dict)
        format = from_union([from_format, format_from_str], obj.get("format"))
        version = from_union([from_version_info, VersionInfo.parse], obj.get("version"))
        return ExtensionItem(format, version)
