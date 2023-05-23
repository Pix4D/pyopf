from enum import Enum
from types import DynamicClassAttribute
from typing import Any, Union

from ..util import from_union


class CoreProjectItemType(Enum):
    """Project item type for items defined in the core OPF spec"""

    @DynamicClassAttribute
    def name(self) -> str:
        return self.value

    CALIBRATION = "calibration"
    CAMERA_LIST = "camera_list"
    CONSTRAINTS = "constraints"
    INPUT_CAMERAS = "input_cameras"
    INPUT_CONTROL_POINTS = "input_control_points"
    POINT_CLOUD = "point_cloud"
    PROJECTED_CONTROL_POINTS = "projected_control_points"
    PROJECTED_INPUT_CAMERAS = "projected_input_cameras"
    SCENE_REFERENCE_FRAME = "scene_reference_frame"


class NamedProjectItemType:
    name: str

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s("%s")' % (self.__class__.__name__, self.name)


class ExtensionProjectItemType(NamedProjectItemType):
    """Project item type for items defined in extensions."""

    """The item type name. Must begin with "ext_"."""

    def __init__(self, name: str):
        if not name.startswith("ext_"):
            raise ValueError(
                "Invalid name for extension project item type, it must start with ext_"
            )
        self.name = name

    def __eq__(self, other: "ExtensionProjectItemType"):
        return self.name == other.name


class UnknownProjectItemType(NamedProjectItemType):
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other: "UnknownProjectItemType"):
        return self.name == other.name


ProjectItemType = (
    CoreProjectItemType | ExtensionProjectItemType | UnknownProjectItemType
)


def project_item_type_from_str(x: Any) -> ProjectItemType:
    return from_union(
        [CoreProjectItemType, ExtensionProjectItemType, UnknownProjectItemType], x
    )


def from_project_item_type(x: ProjectItemType) -> ProjectItemType:
    assert isinstance(x, ProjectItemType)
    return x
