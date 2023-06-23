from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import dateutil.parser
import numpy as np

from ..crs import Geolocation
from ..formats import CoreFormat
from ..items import BaseItem
from ..types import OpfObject, VersionInfo
from ..uid64 import Uid64
from ..util import (
    from_float,
    from_int,
    from_list,
    from_none,
    from_str,
    from_union,
    to_class,
    to_enum,
    to_float,
    to_int,
    vector_from_list,
)
from ..versions import format_and_version_to_type
from .input_rig_relatives_1_0_draft11 import InputRigRelatives
from .pix4d_input_depth_map_1_0_draft2 import Pix4dInputDepthMap
from .sensor_internals_1_0_draft11 import (
    FisheyeInternals,
    Internals,
    PerspectiveInternals,
    SphericalInternals,
)


class ModelSource(str, Enum):
    DATABASE = "database"
    GENERIC = "generic"
    GENERIC_FROM_EXIF = "generic_from_exif"
    USER = "user"


class RigModelSource(Enum):
    DATABASE = "database"
    GENERIC = "generic"
    NOT_APPLICABLE = "not_applicable"
    USER = "user"


from .input_cameras_1_0_draft10 import BandInformation  # noqa: E402
from .input_cameras_1_0_draft10 import Camera  # noqa: E402
from .input_cameras_1_0_draft10 import Capture  # noqa: E402
from .input_cameras_1_0_draft10 import DynamicPixelRange  # noqa: E402
from .input_cameras_1_0_draft10 import OpkOrientation  # noqa: E402
from .input_cameras_1_0_draft10 import Orientation  # noqa: E402
from .input_cameras_1_0_draft10 import PixelRange  # noqa: E402
from .input_cameras_1_0_draft10 import PixelType  # noqa: E402
from .input_cameras_1_0_draft10 import Sensor  # noqa: E402
from .input_cameras_1_0_draft10 import ShutterType  # noqa: E402
from .input_cameras_1_0_draft10 import StaticPixelRange  # noqa: E402
from .input_cameras_1_0_draft10 import YprOrientation  # noqa: E402


class InputCameras(BaseItem):
    """Definition of the input cameras, i.e. the data as provided by the user and camera
    database.
    """

    """List of input captures."""
    captures: List[Capture]
    """List of input sensors."""
    sensors: List[Sensor]

    def __init__(
        self,
        captures: List[Capture],
        sensors: List[Sensor],
        format: CoreFormat = CoreFormat.INPUT_CAMERAS,
        version: VersionInfo = VersionInfo(1, 0, "draft11"),
    ) -> None:
        super(InputCameras, self).__init__(format=format, version=version)

        self.captures = captures
        self.sensors = sensors

    @staticmethod
    def from_dict(obj: Any) -> "InputCameras":
        base = BaseItem.from_dict(obj)

        captures = from_list(Capture.from_dict, obj.get("captures"))
        sensors = from_list(Sensor.from_dict, obj.get("sensors"))
        result = InputCameras(captures, sensors, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result: dict = super(InputCameras, self).to_dict()
        result["captures"] = from_list(lambda x: to_class(Capture, x), self.captures)
        result["sensors"] = from_list(lambda x: to_class(Sensor, x), self.sensors)
        return result


format_and_version_to_type[
    (CoreFormat.INPUT_CAMERAS, VersionInfo(1, 0, "draft11"))
] = InputCameras
