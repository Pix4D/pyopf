from enum import Enum
from types import DynamicClassAttribute
from typing import Any, Dict, Optional

from .util import from_union


class CoreFormat(str, Enum):
    @DynamicClassAttribute
    def name(self):
        return self.value

    CALIBRATED_CAMERAS = "application/opf-calibrated-cameras+json"
    CALIBRATED_CONTROL_POINTS = "application/opf-calibrated-control-points+json"
    CAMERA_LIST = "application/opf-camera-list+json"
    CONSTRAINTS = "application/opf-constraints+json"
    GLTF_MODEL = "model/gltf+json"
    GLTF_BUFFER = "application/gltf-buffer+bin"
    GPS_BIAS = "application/opf-gps-bias+json"
    INPUT_CAMERAS = "application/opf-input-cameras+json"
    INPUT_CONTROL_POINTS = "application/opf-input-control-points+json"
    PROJECTED_CONTROL_POINTS = "application/opf-projected-control-points+json"
    PROJECTED_INPUT_CAMERAS = "application/opf-projected-input-cameras+json"
    PROJECT = "application/opf-project+json"
    SCENE_REFERENCE_FRAME = "application/opf-scene-reference-frame+json"


class NamedFormat(str):
    name: str

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s("%s")' % (self.__class__.__name__, self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other: Any):
        if isinstance(other, self.__class__):
            return self.name == other.name
        else:
            return False

    @property
    def value(self):
        return self.name


class ExtensionFormat(NamedFormat):
    """A extension string formatted as "application/ext-vendor-extension_name+format"""

    def __init__(self, name: str):
        prefix = "application/ext-"
        assert name[: len(prefix)] == prefix
        super().__init__(name)


class UnknownFormat(NamedFormat):
    def __init__(self, name: str):
        super().__init__(name)


Format = CoreFormat | ExtensionFormat | UnknownFormat
Extensions = Optional[Dict[str, Dict[str, Any]]]


def format_from_str(x: Any) -> Format:
    return from_union([CoreFormat, ExtensionFormat, UnknownFormat], x)


def format_to_str(x: Format) -> str:
    if isinstance(x, CoreFormat):
        return x.value
    else:
        return x.name


def from_format(x: Format) -> Format:
    assert isinstance(x, Format)
    return x
