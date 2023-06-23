import copy
from abc import abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

from typing_extensions import Self

from .formats import (
    CoreFormat,
    ExtensionFormat,
    Extensions,
    Format,
    format_from_str,
    format_to_str,
    from_format,
)
from .uid64 import Uid64
from .util import from_extensions, from_uid, from_union, from_version_info
from .VersionInfo import VersionInfo


def _extract_unknown_properties(opf_object: Any, obj: dict, ignore_keys=set()) -> Any:
    result = {
        key: copy.deepcopy(val)
        for key, val in obj.items()
        if key not in opf_object.__dict__ and key not in ignore_keys
    }
    return None if len(result) == 0 else result


class OpfPropertyExtObject(object):
    """Base class for OPF extension property objects.
    This class is similar to OpfObject, but it doesn't contain any logic to handle
    extensions as extensions on extensions are not allowed."""

    unknown_properties: Optional[dict]
    extension_name: str = ""
    version: Optional[VersionInfo] = None

    def __init__(self, unknown_properties: Optional[dict] = None):
        self.unknown_properties = unknown_properties

    def to_dict(self) -> dict:
        result = (
            {}
            if self.unknown_properties is None
            else copy.deepcopy(self.unknown_properties)
        )
        return result

    @staticmethod
    @abstractmethod
    def from_dict(obj: Any) -> None:
        return None

    def _extract_unknown_properties_and_extensions(
        self, obj: dict, ignore_keys=set()
    ) -> "OpfPropertyExtObject":
        """This function is meant to be called from `from_dict` static methods to
        identify all unkonwn properties and store them in self.unown_properties.

        See OpfObject._extract_unknown_properties_and_extensions for details.
        """
        self.unknown_properties = _extract_unknown_properties(self, obj, ignore_keys)
        return self


class OpfObject:
    """Base class for any OPF object.
    This class contains the logic for making OPF objects extensible and preserve
    unknown properties during parsing and serialization."""

    extensions: Extensions
    unknown_properties: Optional[dict]

    def __init__(
        self,
        extensions: Optional[Extensions] = None,
        unknown_properties: Optional[dict] = None,
    ):
        self.extensions = extensions
        self.unknown_properties = unknown_properties

    def to_dict(self, *known_extensions) -> dict:
        result = (
            {}
            if self.unknown_properties is None
            else copy.deepcopy(self.unknown_properties)
        )
        if self.extensions is not None:
            extensions = from_extensions(self.extensions)
        extensions = {}
        for extension in known_extensions:
            if extension is not None:
                extensions[extension.extension_name] = extension.to_dict()
        if len(extensions) != 0:
            result["extensions"] = extensions
        return result

    def _extract_unknown_properties_and_extensions(
        self, obj: dict, ignore_keys=set(["format", "version"])
    ):
        """This function is meant to be called from `from_dict` static methods to
        retrieve the extensions and store them in self.extensions and identify
        all unkonwn properties and store them in self.unkown_properties.

        The implementation copies the input dict first and then removes all the entries
        whose key matches an attribute of the object. This uses self.dict(), which
        means it may not work if using slots. It also imples requires that class
        attributes must use the same name as the JSON attributes.

        The set of keys in ignore_keys are not considered unknown properties. This is
        used for example when a property is parsed to discern the type of an object
        but the property itself is not stored in the final object.
        """
        self.extensions = from_extensions(obj.get("extensions"))
        assert ignore_keys is not None
        self.unknown_properties = _extract_unknown_properties(self, obj, ignore_keys)

    T = TypeVar("T", bound="OpfPropertyExtObject")

    def _extract_known_extension(self, cls: Type[T]) -> Optional[T]:
        if self.extensions is None:
            return None
        try:
            extension = self.extensions[cls.extension_name]
        except KeyError:
            return None
        result = cls.from_dict(extension)
        del self.extensions[cls.extension_name]
        return result
