from typing import Any, Dict, List, Optional

import numpy as np

from ..formats import CoreFormat
from ..items import CoreItem
from ..types import OpfObject, VersionInfo
from ..uid64 import Uid64
from ..util import (
    from_float,
    from_list,
    from_none,
    from_str,
    from_union,
    to_class,
    to_float,
    vector_from_list,
)
from ..versions import FormatVersion, format_and_version_to_type
from .sensor_internals import (
    FisheyeInternals,
    Internals,
    PerspectiveInternals,
    SphericalInternals,
)


class CalibratedCamera(OpfObject):
    id: Uid64
    """Unique ID of the camera, it must appear in the input cameras."""
    orientation_deg: np.ndarray
    """Calibrated Omega-Phi-Kappa angles in degree representing a rotation R_x(ω)R_y(ϕ)R_z(κ)
    from the image CS to the processing CRS.
    """
    position: np.ndarray
    """Calibrated coordinates in the processing CRS."""
    rolling_shutter: Optional[np.ndarray]
    """Refer to [this
    document](https://s3.amazonaws.com/mics.pix4d.com/KB/documents/isprs_rolling_shutter_paper_final_2016.pdf).
    """
    sensor_id: Uid64
    """Unique ID of the sensor used by this camera."""

    def __init__(
        self,
        id: Uid64,
        sensor_id: Uid64,
        orientation_deg: np.ndarray,
        position: np.ndarray,
        rolling_shutter: Optional[np.ndarray] = None,
    ) -> None:
        super(CalibratedCamera, self).__init__()
        self.id = id
        self.orientation_deg = orientation_deg
        self.position = position
        self.rolling_shutter = rolling_shutter
        self.sensor_id = sensor_id

    @staticmethod
    def from_dict(obj: Any) -> "CalibratedCamera":
        assert isinstance(obj, dict)
        id = Uid64(int=int(obj.get("id")))
        orientation_deg = vector_from_list(obj.get("orientation_deg"), 3, 3)
        position = vector_from_list(obj.get("position"), 3, 3)
        rolling_shutter = from_union(
            [lambda x: vector_from_list(x, 3, 3), from_none], obj.get("rolling_shutter")
        )
        sensor_id = Uid64(int=int(obj.get("sensor_id")))
        result = CalibratedCamera(
            id, sensor_id, orientation_deg, position, rolling_shutter
        )
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(CalibratedCamera, self).to_dict()
        result["id"] = self.id.int
        result["orientation_deg"] = from_list(to_float, self.orientation_deg)
        result["position"] = from_list(to_float, self.position)
        if self.rolling_shutter is not None:
            result["rolling_shutter"] = from_union(
                [lambda x: from_list(to_float, x), from_none], self.rolling_shutter
            )
        result["sensor_id"] = self.sensor_id.int
        return result


class CalibratedRigRelatives(OpfObject):
    """Calibrated rig relatives contain the optimised relative translations and rotations in
    processing CRS units.
    """

    """Euler angles in degree (see convention [here](auxiliary_objects.md#rig-relatives))"""
    rotation_angles_deg: np.ndarray
    """Relative translation in processing CRS units"""
    translation: np.ndarray

    def __init__(
        self,
        rotation_angles_deg: np.ndarray,
        translation: np.ndarray,
    ) -> None:
        super(CalibratedRigRelatives, self).__init__()
        self.rotation_angles_deg = rotation_angles_deg
        self.translation = translation

    @staticmethod
    def from_dict(obj: Any) -> "CalibratedRigRelatives":
        assert isinstance(obj, dict)
        rotation_angles_deg = vector_from_list(obj.get("rotation_angles_deg"), 3, 3)
        translation = vector_from_list(obj.get("translation"), 3, 3)
        result = CalibratedRigRelatives(rotation_angles_deg, translation)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(CalibratedRigRelatives, self).to_dict()
        result["rotation_angles_deg"] = from_list(to_float, self.rotation_angles_deg)
        result["translation"] = from_list(to_float, self.translation)
        return result


class CalibratedSensor(OpfObject):
    """Unique ID of the sensor, it must appear in the input cameras."""

    id: Uid64
    """Calibrated sensor internal parameters."""
    internals: Internals
    rig_relatives: Optional[CalibratedRigRelatives]

    def __init__(
        self,
        id: Uid64,
        internals: Internals,
        rig_relatives: Optional[CalibratedRigRelatives] = None,
    ) -> None:
        super(CalibratedSensor, self).__init__()
        self.id = id
        self.internals = internals
        self.rig_relatives = rig_relatives

    @staticmethod
    def from_dict(obj: Any) -> "CalibratedSensor":
        assert isinstance(obj, dict)
        id = Uid64(int=int(obj.get("id")))
        internals = from_union(
            [
                SphericalInternals.from_dict,
                PerspectiveInternals.from_dict,
                FisheyeInternals.from_dict,
            ],
            obj.get("internals"),
        )
        rig_relatives = from_union(
            [CalibratedRigRelatives.from_dict, from_none], obj.get("rig_relatives")
        )
        result = CalibratedSensor(id, internals, rig_relatives)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(CalibratedSensor, self).to_dict()
        result["id"] = self.id.int
        result["internals"] = to_class(Internals, self.internals)
        if self.rig_relatives is not None:
            result["rig_relatives"] = from_union(
                [lambda x: to_class(CalibratedRigRelatives, x), from_none],
                self.rig_relatives,
            )
        return result


class CalibratedCameras(CoreItem):
    """Definition of Calibrated Camera Parameters"""

    """Calibrated camera parameters."""
    cameras: List[CalibratedCamera]
    """Calibrated sensor parameters."""
    sensors: List[CalibratedSensor]

    def __init__(
        self,
        cameras: List[CalibratedCamera],
        sensors: List[CalibratedSensor],
        format: CoreFormat = CoreFormat.CALIBRATED_CAMERAS,
        version: VersionInfo = FormatVersion.CALIBRATED_CAMERAS,
    ) -> None:
        super(CalibratedCameras, self).__init__(format=format, version=version)

        self.cameras = cameras
        self.sensors = sensors

    @staticmethod
    def from_dict(obj: Any) -> "CalibratedCameras":
        base = CoreItem.from_dict(obj)
        cameras = from_list(CalibratedCamera.from_dict, obj.get("cameras"))
        sensors = from_list(CalibratedSensor.from_dict, obj.get("sensors"))
        result = CalibratedCameras(cameras, sensors, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result: dict = super(CalibratedCameras, self).to_dict()
        result["cameras"] = from_list(
            lambda x: to_class(CalibratedCamera, x), self.cameras
        )
        result["sensors"] = from_list(
            lambda x: to_class(CalibratedSensor, x), self.sensors
        )
        return result


format_and_version_to_type[
    (CoreFormat.CALIBRATED_CAMERAS, FormatVersion.CALIBRATED_CAMERAS)
] = CalibratedCameras
