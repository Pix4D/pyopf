from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import dateutil.parser
import numpy as np

from ..crs import Geolocation
from ..ext.pix4d_input_depth_map import Pix4dInputDepthMap
from ..formats import CoreFormat
from ..items import CoreItem
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
from ..versions import FormatVersion, format_and_version_to_type
from .input_rig_relatives import InputRigRelatives
from .sensor_internals import (
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


class StaticPixelRange(OpfObject):
    """Defines the range of valid pixel values. Values &le; min are considered underexposed and
    &ge; max overexposed. Can be a [static range](#pixel-range) or a [dynamic
    range](#dynamic-pixel-range).

    Static pixel data range given by a minimum and maximum.
    """

    max: float
    """Maximum pixel value."""
    min: float
    """Minimum pixel value."""

    def __init__(
        self,
        min: float,
        max: float,
    ) -> None:
        super(StaticPixelRange, self).__init__()
        self.max = max
        self.min = min

    @staticmethod
    def from_dict(obj: Any) -> "StaticPixelRange":
        assert isinstance(obj, dict)
        max = from_float(obj.get("max"))
        min = from_float(obj.get("min"))
        result = StaticPixelRange(min, max)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(StaticPixelRange, self).to_dict()
        result["max"] = to_float(self.max)
        result["min"] = to_float(self.min)
        return result


class DynamicPixelRange(OpfObject):
    """Defines the range of valid pixel values. Values &le; min are considered underexposed and
    &ge; max overexposed. Can be a [static range](#pixel-range) or a [dynamic
    range](#dynamic-pixel-range).

    Dynamically inferred pixel range. The range needs to be derived from the data by looking
    at the image content, filtering extreme values at both ends with the given percentile.
    """

    percentile: float
    """Percentage of values ignored on both ends of the ordered list of values when computing
    the min/max. It must be a positive value and 0 means nothing is ignored.
    """

    def __init__(
        self,
        percentile: float,
    ) -> None:
        super(DynamicPixelRange, self).__init__()
        self.percentile = percentile

    @staticmethod
    def from_dict(obj: Any) -> "DynamicPixelRange":
        assert isinstance(obj, dict)
        percentile = from_float(obj.get("percentile"))
        result = DynamicPixelRange(percentile)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(DynamicPixelRange, self).to_dict()
        result["percentile"] = to_float(self.percentile)
        return result


PixelRange = StaticPixelRange | DynamicPixelRange


class PixelType(str, Enum):
    FLOAT = "float"
    UINT12 = "uint12"
    UINT16 = "uint16"
    UINT8 = "uint8"


class Camera(OpfObject):
    """One camera in a capture. It is associated to a sensor via a sensor identifier."""

    id: Uid64
    image_orientation: Optional[int]
    """Specifies the image orientation following [EXIF, page
    37](https://www.jeita.or.jp/japanese/standard/book/CP-3451E_E/#target/page_no=38). 1: no
    rotation, no mirror, 2: mirror horizontal, 3: rotate 180 degrees, 4: mirror vertical, 5:
    mirror horizontal and rotate 270 degrees CW, 6: rotate 90 degrees CW, 7: mirror
    horizontal and rotate 90 degrees CW, 8: rotate 270 degrees CW.
    """
    model_source: ModelSource
    pixel_range: PixelRange
    """Defines the range of valid pixel values. Values &le; min are considered underexposed and
    &ge; max overexposed. Can be a StaticPixelRange or a DynamicPixelRange.
    """
    pixel_type: PixelType
    sensor_id: Uid64
    input_depth_map: Optional[Pix4dInputDepthMap]

    def __init__(
        self,
        id: Uid64,
        model_source: ModelSource,
        pixel_range: PixelRange,
        pixel_type: PixelType,
        sensor_id: Uid64,
        image_orientation: Optional[int] = None,
        input_depth_map: Optional[Pix4dInputDepthMap] = None,
    ) -> None:
        super(Camera, self).__init__()
        self.id = id
        self.image_orientation = image_orientation
        self.model_source = model_source
        self.pixel_range = pixel_range
        self.pixel_type = pixel_type
        self.sensor_id = sensor_id
        self.input_depth_map = input_depth_map

    @staticmethod
    def from_dict(obj: Any) -> "Camera":
        assert isinstance(obj, dict)
        id = Uid64(int=int(obj.get("id")))
        image_orientation = from_union(
            [from_int, from_none], obj.get("image_orientation")
        )
        model_source = ModelSource(obj.get("model_source"))
        pixel_range = from_union(
            [StaticPixelRange.from_dict, DynamicPixelRange.from_dict],
            obj.get("pixel_range"),
        )
        pixel_type = PixelType(obj.get("pixel_type"))
        sensor_id = Uid64(int=int(obj.get("sensor_id")))
        result = Camera(
            id,
            model_source,
            pixel_range,
            pixel_type,
            sensor_id,
            image_orientation,
        )
        result._extract_unknown_properties_and_extensions(obj)
        result.input_depth_map = result._extract_known_extension(Pix4dInputDepthMap)
        return result

    def to_dict(self) -> dict:
        result = super(Camera, self).to_dict(self.input_depth_map)
        result["id"] = self.id.int
        if self.image_orientation is not None:
            result["image_orientation"] = from_union(
                [from_int, from_none], self.image_orientation
            )
        result["model_source"] = to_enum(ModelSource, self.model_source)
        result["pixel_range"] = to_class(PixelRange, self.pixel_range)
        result["pixel_type"] = to_enum(PixelType, self.pixel_type)
        result["sensor_id"] = self.sensor_id.int
        return result


class YprOrientation(OpfObject):
    """Camera orientation as Yaw-Pitch-Roll

    Yaw-Pitch-Roll angles represent a rotation R_z(yaw)R_y(pitch)R_x(roll) from the image CS
    to navigation CRS base change, where the image CS is right-top-back in image space, the
    navigation CRS is East-North-Down and angles (0, 0, 0) represent the identity
    transformation.
    """

    angles_deg: np.ndarray
    """Yaw, pitch, roll angles in degrees."""
    sigmas_deg: np.ndarray
    """Error estimation (standard deviation) in degrees."""
    type = "yaw_pitch_roll"

    def __init__(
        self,
        angles_deg: np.ndarray,
        sigmas_deg: np.ndarray,
    ) -> None:
        super(YprOrientation, self).__init__()
        self.angles_deg = angles_deg
        self.sigmas_deg = sigmas_deg

    @staticmethod
    def from_dict(obj: Any) -> "YprOrientation":
        assert isinstance(obj, dict)
        assert obj.get("type") == YprOrientation.type

        angles_deg = vector_from_list(obj.get("angles_deg"), 3, 3)
        sigmas_deg = vector_from_list(obj.get("sigmas_deg"), 3, 3)
        result = YprOrientation(angles_deg, sigmas_deg)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(YprOrientation, self).to_dict()
        result["angles_deg"] = from_list(to_float, self.angles_deg)
        result["sigmas_deg"] = from_list(to_float, self.sigmas_deg)
        result["type"] = from_str(self.type)
        return result


class OpkOrientation(OpfObject):
    """Camera orientation as Omega-Phi-Kappa.

    Omega-Phi-Kappa represent a rotation R_x(ω)R_y(ϕ)R_z(κ) from the image CS to a separately
    defined Cartesian CRS, where the image CS is right-top-back in image space.
    """

    angles_deg: np.ndarray
    """Omega, phi, kappa angles in degrees."""
    sigmas_deg: np.ndarray
    """Error estimation (standard deviation) in degrees."""
    crs: str
    """The target CRS of the rotation. A Cartesian horizontal CRS as WKT2 string or `"Auth:code"`."""

    type = "omega_phi_kappa"

    def __init__(
        self,
        angles_deg: np.ndarray,
        sigmas_deg: np.ndarray,
        crs: str,
    ) -> None:
        super(OpkOrientation, self).__init__()
        self.angles_deg = angles_deg
        self.sigmas_deg = sigmas_deg
        self.crs = crs

    @staticmethod
    def from_dict(obj: Any) -> "OpkOrientation":
        assert isinstance(obj, dict)
        assert obj.get("type") == OpkOrientation.type

        angles_deg = vector_from_list(obj.get("angles_deg"), 3, 3)
        sigmas_deg = vector_from_list(obj.get("sigmas_deg"), 3, 3)
        crs = from_str(obj.get("crs"))

        result = OpkOrientation(angles_deg, sigmas_deg, crs)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(OpkOrientation, self).to_dict()
        result["angles_deg"] = from_list(to_float, self.angles_deg)
        result["sigmas_deg"] = from_list(to_float, self.sigmas_deg)
        result["type"] = from_str(self.type)
        result["crs"] = from_str(self.crs)
        return result


Orientation = YprOrientation | OpkOrientation


class RigModelSource(Enum):
    DATABASE = "database"
    GENERIC = "generic"
    NOT_APPLICABLE = "not_applicable"
    USER = "user"


class Capture(OpfObject):
    """A collection of cameras and spatio-temporal information of an image acquisition event."""

    cameras: List[Camera]
    """List of cameras in the capture."""
    geolocation: Optional[Geolocation]
    height_above_takeoff_m: Optional[float]
    """Height above the take-off place in meters."""
    id: Uid64
    orientation: Optional[Orientation]
    """One of YprOrientation or OpkOrientation"""
    reference_camera_id: Uid64
    """ID of the reference camera in a rig. Required also for single camera capture."""
    rig_model_source: RigModelSource
    time: datetime
    """The time of image acquisition formatted as [ISO
    8601](https://en.wikipedia.org/wiki/ISO_8601). If the timezone is known then the time
    should be specified as UTC, if no timezone is given then it is unknown.
    """

    def __init__(
        self,
        id: Uid64,
        cameras: List[Camera],
        height_above_takeoff_m: Optional[float],
        reference_camera_id: Uid64,
        rig_model_source: RigModelSource,
        time: datetime,
        geolocation: Optional[Geolocation] = None,
        orientation: Optional[Orientation] = None,
    ) -> None:
        super(Capture, self).__init__()
        self.cameras = cameras
        self.geolocation = geolocation
        self.height_above_takeoff_m = height_above_takeoff_m
        self.id = id
        self.orientation = orientation
        self.reference_camera_id = reference_camera_id
        self.rig_model_source = rig_model_source
        self.time = time

    @staticmethod
    def from_dict(obj: Any) -> "Capture":
        assert isinstance(obj, dict)
        cameras = from_list(Camera.from_dict, obj.get("cameras"))
        geolocation = from_union(
            [Geolocation.from_dict, from_none], obj.get("geolocation")
        )
        height_above_takeoff_m = from_union(
            [from_float, from_none], obj.get("height_above_takeoff_m")
        )
        id = Uid64(int=int(obj.get("id")))
        orientation = from_union(
            [YprOrientation.from_dict, OpkOrientation.from_dict, from_none],
            obj.get("orientation"),
        )
        reference_camera_id = Uid64(int=int(obj.get("reference_camera_id")))
        rig_model_source = RigModelSource(obj.get("rig_model_source"))
        time = dateutil.parser.isoparse(str(obj.get("time")))
        result = Capture(
            id,
            cameras,
            height_above_takeoff_m,
            reference_camera_id,
            rig_model_source,
            time,
            geolocation,
            orientation,
        )
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Capture, self).to_dict()
        result["cameras"] = from_list(lambda x: to_class(Camera, x), self.cameras)
        if self.geolocation is not None:
            result["geolocation"] = from_union(
                [lambda x: to_class(Geolocation, x), from_none], self.geolocation
            )
        if self.height_above_takeoff_m is not None:
            result["height_above_takeoff_m"] = from_union(
                [to_float, from_none], self.height_above_takeoff_m
            )
        result["id"] = self.id.int
        if self.orientation is not None:
            result["orientation"] = from_union(
                [lambda x: to_class(Orientation, x), from_none], self.orientation
            )
        result["reference_camera_id"] = self.reference_camera_id.int
        result["rig_model_source"] = to_enum(RigModelSource, self.rig_model_source)
        result["time"] = self.time.isoformat()
        return result


class BandInformation(OpfObject):
    """Information about a band"""

    name: Optional[str]
    weight: float
    """Weights to compute a luminance representation of the image"""

    def __init__(
        self,
        weight: float,
        name: Optional[str] = None,
    ) -> None:
        super(BandInformation, self).__init__()
        self.name = name
        assert 0 <= weight <= 1
        self.weight = weight

    @staticmethod
    def from_dict(obj: Any) -> "BandInformation":
        assert isinstance(obj, dict)
        name = from_union([from_str, from_none], obj.get("name"))
        weight = from_float(obj.get("weight"))
        result = BandInformation(weight, name)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(BandInformation, self).to_dict()
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        result["weight"] = to_float(self.weight)
        return result


class ShutterType(Enum):
    GLOBAL = "global"
    ROLLING = "rolling"


class Sensor(OpfObject):
    """Specifies one sensor model of a physical camera unit, described by lens type, general
    parameters and internal optical parameters.
    """

    """Image band properties. The number of items must be equal to the channel count. For
    example, an RGB image has the three bands `"Red", "Green", "Blue"`. The weights of all
    bands must be greater than or equal to 0 and sum to 1.
    """
    bands: List[BandInformation]
    id: Uid64
    image_size_px: np.ndarray  # 2D vector
    """Image width and height in pixels."""
    internals: Internals
    name: str
    """Sensor name."""
    pixel_size_um: float
    """Pixel size in micrometers."""
    rig_relatives: Optional[InputRigRelatives]
    shutter_type: ShutterType

    def __init__(
        self,
        id: Uid64,
        name: str,
        bands: List[BandInformation],
        image_size_px: np.ndarray,
        internals: Internals,
        pixel_size_um: float,
        shutter_type: ShutterType,
        rig_relatives: Optional[InputRigRelatives] = None,
    ) -> None:
        super(Sensor, self).__init__()
        self.bands = bands
        self.id = id
        self.image_size_px = image_size_px
        self.internals = internals
        self.name = name
        self.pixel_size_um = pixel_size_um
        self.rig_relatives = rig_relatives
        self.shutter_type = shutter_type

    @staticmethod
    def from_dict(obj: Any) -> "Sensor":
        assert isinstance(obj, dict)
        bands = from_list(BandInformation.from_dict, obj.get("bands"))
        id = Uid64(int=int(obj.get("id")))
        image_size_px = vector_from_list(obj.get("image_size_px"), 2, 2, dtype=int)
        internals = from_union(
            [
                SphericalInternals.from_dict,
                PerspectiveInternals.from_dict,
                FisheyeInternals.from_dict,
            ],
            obj.get("internals"),
        )
        name = from_str(obj.get("name"))
        pixel_size_um = from_float(obj.get("pixel_size_um"))
        rig_relatives = from_union(
            [InputRigRelatives.from_dict, from_none], obj.get("rig_relatives")
        )
        shutter_type = ShutterType(obj.get("shutter_type"))
        result = Sensor(
            id,
            name,
            bands,
            image_size_px,
            internals,
            pixel_size_um,
            shutter_type,
            rig_relatives,
        )
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Sensor, self).to_dict()
        result["bands"] = from_list(lambda x: to_class(BandInformation, x), self.bands)
        result["id"] = self.id.int
        result["image_size_px"] = from_list(to_int, self.image_size_px)
        result["internals"] = to_class(Internals, self.internals)
        result["name"] = from_str(self.name)
        result["pixel_size_um"] = to_float(self.pixel_size_um)
        if self.rig_relatives is not None:
            result["rig_relatives"] = from_union(
                [lambda x: to_class(InputRigRelatives, x), from_none],
                self.rig_relatives,
            )
        result["shutter_type"] = to_enum(ShutterType, self.shutter_type)
        return result


class InputCameras(CoreItem):
    """Definition of the input cameras, i.e. the data as provided by the user and camera
    database.
    """

    captures: List[Capture]
    """List of input captures."""
    sensors: List[Sensor]
    """List of input sensors."""

    def __init__(
        self,
        captures: List[Capture],
        sensors: List[Sensor],
        format: CoreFormat = CoreFormat.INPUT_CAMERAS,
        version: VersionInfo = FormatVersion.INPUT_CAMERAS,
    ) -> None:
        super(InputCameras, self).__init__(format=format, version=version)

        self.captures = captures
        self.sensors = sensors

    @staticmethod
    def from_dict(obj: Any) -> "InputCameras":
        base = CoreItem.from_dict(obj)

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
    (CoreFormat.INPUT_CAMERAS, FormatVersion.INPUT_CAMERAS)
] = InputCameras
