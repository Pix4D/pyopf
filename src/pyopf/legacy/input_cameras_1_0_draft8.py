from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

import numpy as np

from ..crs import Geolocation
from ..formats import CoreFormat
from ..items import BaseItem
from ..types import OpfObject, VersionInfo
from ..uid64 import Uid64
from ..util import (
    IntType,
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
from .input_rig_relatives_1_0_draft8 import InputRigRelatives
from .sensor_internals_1_0_draft8 import (
    FisheyeInternals,
    Internals,
    PerspectiveInternals,
    SphericalInternals,
)


class ModelSource(str, Enum):
    DATABASE = "database"
    GENERIC = "generic"
    GENERIC_FROM_EXIF = "generic_from_exif"


class DepthMap(OpfObject):
    """Reference to depth information, for example for RGB-D type sensors"""

    """The location of the depth map image given as a URI-reference, pointing to a single
    channel image file. Each pixel represents the estimated distance from the device to its
    environment on the camera depth axis. Multiply by `unit_to_meters` to obtain values in
    meters. The depth map is aligned with the camera image but may be of different
    resolution. Pixel values of `0`, `NaN` and negative infinity indicate invalid or missing
    data. Values of +infinity indicate measurements that are above the sensing range.
    """
    uri: str
    """The location of the confidence map image given as a URI-reference, pointing to a single
    channel image file. If present, it must be of the same dimension as the depth map and
    `min_confidence`, in addition `max_confidence` must be defined. It indicates the level of
    confidence of the depth measurements. Valid confidence values range from `min_confidence`
    (lowest confidence) to `max_confidence` (highest confidence).
    """
    confidence_uri: Optional[str]
    """Maximum confidence value to consider a depth measurement valid. It must be present if
    `confidence_uri` is present.
    """
    max_confidence: Optional[float]
    """Minimum confidence value to consider a depth measurement valid. It must be present if
    `confidence_uri` is present.
    """
    min_confidence: Optional[float]
    """Multiply this scale factor with depth maps values in order to obtain values in meters.
    For example, if the depth map values represent millimeters the scale factor is 1/1000
    (e.g. a value of 1000mm corresponds to 1m).  If not specified, defaults to 1.
    """
    unit_to_meters: Optional[float]

    def __init__(
        self,
        uri: str,
        confidence_uri: Optional[str] = None,
        max_confidence: Optional[float] = None,
        min_confidence: Optional[float] = None,
        unit_to_meters: Optional[float] = None,
    ) -> None:
        super(DepthMap, self).__init__()
        self.confidence_uri = confidence_uri
        self.max_confidence = max_confidence
        self.min_confidence = min_confidence
        self.unit_to_meters = unit_to_meters
        self.uri = uri

    @staticmethod
    def from_dict(obj: Any) -> "DepthMap":
        assert isinstance(obj, dict)
        uri = from_str(obj.get("uri"))
        confidence_uri = from_union([from_str, from_none], obj.get("confidence_uri"))
        max_confidence = from_union([from_float, from_none], obj.get("max_confidence"))
        min_confidence = from_union([from_float, from_none], obj.get("min_confidence"))
        unit_to_meters = from_union([from_float, from_none], obj.get("unit_to_meters"))
        result = DepthMap(
            uri,
            confidence_uri,
            max_confidence,
            min_confidence,
            unit_to_meters,
        )
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(DepthMap, self).to_dict()
        if self.confidence_uri is not None:
            result["confidence_uri"] = from_union(
                [from_str, from_none], self.confidence_uri
            )
        if self.max_confidence is not None:
            result["max_confidence"] = from_union(
                [to_float, from_none], self.max_confidence
            )
        if self.min_confidence is not None:
            result["min_confidence"] = from_union(
                [to_float, from_none], self.min_confidence
            )
        if self.unit_to_meters is not None:
            result["unit_to_meters"] = from_union(
                [to_float, from_none], self.unit_to_meters
            )
        result["uri"] = from_str(self.uri)
        return result


class StaticPixelRange(OpfObject):
    """Defines the range of valid pixel values. Values &le; min are considered underexposed and
    &ge; max overexposed. Can be a [static range](#pixel-range) or a [dynamic
    range](#dynamic-pixel-range).

    Static pixel data range given by a minimum and maximum.
    """

    """Maximum pixel value."""
    max: float
    """Minimum pixel value."""
    min: float

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

    """Percentage of values ignored on both ends of the ordered list of values when computing
    the min/max. It must be a positive value and 0 means nothing is ignored.
    """
    percentile: float

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

    depth: Optional[DepthMap]
    id: Uid64
    """Specifies the image orientation following [EXIF, page
    37](https://www.jeita.or.jp/japanese/standard/book/CP-3451E_E/#target/page_no=38). 1: no
    rotation, no mirror, 2: mirror horizontal, 3: rotate 180 degrees, 4: mirror vertical, 5:
    mirror horizontal and rotate 270 degrees CW, 6: rotate 90 degrees CW, 7: mirror
    horizontal and rotate 90 degrees CW, 8: rotate 270 degrees CW.
    """
    image_orientation: Optional[int]
    """The location of the image file given as a URI-reference."""
    image_uri: str
    model_source: ModelSource
    """Specifies the page in a multi-image file."""
    page_index: IntType
    """Defines the range of valid pixel values. Values &le; min are considered underexposed and
    &ge; max overexposed. Can be a StaticPixelRange or a DynamicPixelRange.
    """
    pixel_range: PixelRange
    pixel_type: PixelType
    sensor_id: Uid64

    def __init__(
        self,
        id: Uid64,
        image_uri: str,
        model_source: ModelSource,
        page_index: IntType,
        pixel_range: PixelRange,
        pixel_type: PixelType,
        sensor_id: Uid64,
        depth: Optional[DepthMap] = None,
        image_orientation: Optional[int] = None,
    ) -> None:
        super(Camera, self).__init__()
        self.depth = depth
        self.id = id
        self.image_orientation = image_orientation
        self.image_uri = image_uri
        self.model_source = model_source
        self.page_index = page_index
        self.pixel_range = pixel_range
        self.pixel_type = pixel_type
        self.sensor_id = sensor_id

    @staticmethod
    def from_dict(obj: Any) -> "Camera":
        assert isinstance(obj, dict)
        depth = from_union([DepthMap.from_dict, from_none], obj.get("depth"))
        id = Uid64(hex=obj.get("id"))
        image_orientation = from_union(
            [from_int, from_none], obj.get("image_orientation")
        )
        image_uri = from_str(obj.get("image_uri"))
        model_source = ModelSource(obj.get("model_source"))
        page_index = from_int(obj.get("page_index"))
        pixel_range = from_union(
            [StaticPixelRange.from_dict, DynamicPixelRange.from_dict],
            obj.get("pixel_range"),
        )
        pixel_type = PixelType(obj.get("pixel_type"))
        sensor_id = Uid64(hex=from_str(obj.get("sensor_id")))
        result = Camera(
            id,
            image_uri,
            model_source,
            page_index,
            pixel_range,
            pixel_type,
            sensor_id,
            depth,
            image_orientation,
        )
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Camera, self).to_dict()
        if self.depth is not None:
            result["depth"] = from_union(
                [lambda x: to_class(DepthMap, x), from_none], self.depth
            )
        result["id"] = str(self.id)
        if self.image_orientation is not None:
            result["image_orientation"] = from_union(
                [from_int, from_none], self.image_orientation
            )
        result["image_uri"] = from_str(self.image_uri)
        result["model_source"] = to_enum(ModelSource, self.model_source)
        result["page_index"] = from_int(self.page_index)
        result["pixel_range"] = to_class(PixelRange, self.pixel_range)
        result["pixel_type"] = to_enum(PixelType, self.pixel_type)
        result["sensor_id"] = from_str(self.sensor_id.hex)
        return result


class YprOrientation(OpfObject):
    """Camera orientation as Yaw-Pitch-Roll

    Yaw-Pitch-Roll angles represent a rotation R_z(yaw)R_y(pitch)R_x(roll) from the image CS
    to navigation CRS base change, where the image CS is right-top-back in image space, the
    navigation CRS is East-North-Down and angles (0, 0, 0) represent the identity
    transformation.
    """

    """Yaw, pitch, roll angles in degrees."""
    angles_deg: np.ndarray
    """Error estimation (standard deviation) in degrees."""
    sigmas_deg: np.ndarray
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

    """Omega, phi, kappa angles in degrees."""
    angles_deg: np.ndarray
    """Error estimation (standard deviation) in degrees."""
    sigmas_deg: np.ndarray
    """The target CRS of the rotation. A Cartesian horizontal CRS as WKT2 string or `"Auth:code"`."""
    crs: str

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


class Capture(OpfObject):
    """A collection of cameras and spatio-temporal information of an image acquisition event."""

    """List of cameras in the capture."""
    cameras: List[Camera]
    geolocation: Optional[Geolocation]
    """Height above the take-off place in meters."""
    height_above_takeoff_m: Optional[float]
    id: Uid64
    """One of YprOrientation or OpkOrientation"""
    orientation: Optional[Orientation]
    """ID of the reference camera in a rig. Required also for single camera capture."""
    reference_camera_id: Uid64
    rig_model_source: RigModelSource
    """The time of image acquisition formatted as [ISO
    8601](https://en.wikipedia.org/wiki/ISO_8601). If the timezone is known then the time
    should be specified as UTC, if no timezone is given then it is unknown.
    """
    time: datetime

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
        id = Uid64(hex=obj.get("id"))
        orientation = from_union(
            [YprOrientation.from_dict, OpkOrientation.from_dict, from_none],
            obj.get("orientation"),
        )
        reference_camera_id = Uid64(hex=obj.get("reference_camera_id"))
        rig_model_source = RigModelSource(obj.get("rig_model_source"))
        time = datetime.fromisoformat(obj["time"])
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
        result["id"] = str(self.id)
        if self.orientation is not None:
            result["orientation"] = from_union(
                [lambda x: to_class(Orientation, x), from_none], self.orientation
            )
        result["reference_camera_id"] = str(self.reference_camera_id)
        result["rig_model_source"] = to_enum(RigModelSource, self.rig_model_source)
        result["time"] = str(self.time)
        return result


class BandInformation(OpfObject):
    """Information about a band"""

    name: Optional[str]
    """Weights to compute a luminance representation of the image"""
    weight: float

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
    """Image width and height in pixels."""
    image_size_px: np.ndarray  # 2D vector
    internals: Internals
    """Sensor name."""
    name: str
    """Pixel size in micrometers."""
    pixel_size_um: float
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
        id = Uid64(hex=obj.get("id"))
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
        result["id"] = str(self.id)
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
        version: VersionInfo = VersionInfo(1, 0, "draft8"),
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
    (CoreFormat.INPUT_CAMERAS, VersionInfo(1, 0, "draft8"))
] = InputCameras
