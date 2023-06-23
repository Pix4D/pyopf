from typing import Any, List, Optional

import numpy as np

from ..formats import CoreFormat
from ..items import CoreItem
from ..types import OpfObject, VersionInfo
from ..uid64 import Uid64
from ..util import (
    from_list,
    from_none,
    from_union,
    to_class,
    to_float,
    vector_from_list,
)
from ..versions import FormatVersion, format_and_version_to_type


class ProjectedGeolocation(OpfObject):
    """Input geolocation in the processing CRS axes and units."""

    position: np.ndarray
    """Coordinates in the processing CRS."""
    sigmas: np.ndarray
    """Standard deviation in the processing CRS units."""

    def __init__(
        self,
        position: np.ndarray,
        sigmas: np.ndarray,
    ) -> None:
        super(ProjectedGeolocation, self).__init__()
        self.position = position
        self.sigmas = sigmas

    @staticmethod
    def from_dict(obj: Any) -> "ProjectedGeolocation":
        assert isinstance(obj, dict)
        position = vector_from_list(obj.get("position"), 3, 3)
        sigmas = vector_from_list(obj.get("sigmas"), 3, 3)
        result = ProjectedGeolocation(position, sigmas)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(ProjectedGeolocation, self).to_dict()
        result["position"] = from_list(to_float, self.position)
        result["sigmas"] = from_list(to_float, self.sigmas)
        return result


class ProjectedOrientation(OpfObject):
    """Input orientation in the processing CRS axes."""

    angles_deg: np.ndarray
    """Omega-Phi-Kappa angles in degree representing a rotation R_x(ω)R_y(ϕ)R_z(κ) from the
    image CS to the processing CRS.
    """
    sigmas_deg: np.ndarray
    """Standard deviation of Omega-Phi-Kappa angles in degree."""

    def __init__(
        self,
        angles_deg: np.ndarray,
        sigmas_deg: np.ndarray,
    ) -> None:
        super(ProjectedOrientation, self).__init__()
        self.angles_deg = angles_deg
        self.sigmas_deg = sigmas_deg

    @staticmethod
    def from_dict(obj: Any) -> "ProjectedOrientation":
        assert isinstance(obj, dict)
        angles_deg = vector_from_list(obj.get("angles_deg"), 3, 3)
        sigmas_deg = vector_from_list(obj.get("sigmas_deg"), 3, 3)
        result = ProjectedOrientation(angles_deg, sigmas_deg)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(ProjectedOrientation, self).to_dict()
        result["angles_deg"] = from_list(to_float, self.angles_deg)
        result["sigmas_deg"] = from_list(to_float, self.sigmas_deg)
        return result


class ProjectedCapture(OpfObject):
    """Processing CRS dependent parameters of a capture sensor."""

    geolocation: Optional[ProjectedGeolocation]
    """Unique identifier pointing to a capture element in the input cameras."""
    id: Uid64
    orientation: Optional[ProjectedOrientation]

    def __init__(
        self,
        id: Uid64,
        geolocation: Optional[ProjectedGeolocation] = None,
        orientation: Optional[ProjectedOrientation] = None,
    ) -> None:
        super(ProjectedCapture, self).__init__()
        self.geolocation = geolocation
        self.id = id
        self.orientation = orientation

    @staticmethod
    def from_dict(obj: Any) -> "ProjectedCapture":
        assert isinstance(obj, dict)
        geolocation = from_union(
            [ProjectedGeolocation.from_dict, from_none], obj.get("geolocation")
        )
        id = Uid64(int=int(obj.get("id")))
        orientation = from_union(
            [ProjectedOrientation.from_dict, from_none], obj.get("orientation")
        )
        result = ProjectedCapture(id, geolocation, orientation)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(ProjectedCapture, self).to_dict()
        if self.geolocation is not None:
            result["geolocation"] = from_union(
                [lambda x: to_class(ProjectedGeolocation, x), from_none],
                self.geolocation,
            )
        result["id"] = self.id.int
        if self.orientation is not None:
            result["orientation"] = from_union(
                [lambda x: to_class(ProjectedOrientation, x), from_none],
                self.orientation,
            )
        return result


class ProjectedRigTranslation(OpfObject):
    """Projected rig relatives only contain the relative translation as the relative rotation
    stays the same as the input. The difference between the projected rig translation and
    input rig translation is that the projected translation uses units of the processing CRS.
    """

    sigmas: np.ndarray
    """Measurement error (standard deviation) in processing CRS units."""
    values: np.ndarray
    """Relative translation in processing CRS units."""

    def __init__(
        self,
        sigmas: np.ndarray,
        values: np.ndarray,
    ) -> None:
        super(ProjectedRigTranslation, self).__init__()
        self.sigmas = sigmas
        self.values = values

    @staticmethod
    def from_dict(obj: Any) -> "ProjectedRigTranslation":
        assert isinstance(obj, dict)
        sigmas = vector_from_list(obj.get("sigmas"), 3, 3)
        values = vector_from_list(obj.get("values"), 3, 3)
        result = ProjectedRigTranslation(sigmas, values)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(ProjectedRigTranslation, self).to_dict()
        result["sigmas"] = from_list(to_float, self.sigmas)
        result["values"] = from_list(to_float, self.values)
        return result


class ProjectedSensor(OpfObject):
    """Processing CRS dependent parameters of an input sensor."""

    id: Uid64
    """Unique identifier pointing to a sensor element in the input cameras."""
    rig_translation: Optional[ProjectedRigTranslation]

    def __init__(
        self,
        id: Uid64,
        rig_translation: Optional[ProjectedRigTranslation] = None,
    ) -> None:
        super(ProjectedSensor, self).__init__()
        self.id = id
        self.rig_translation = rig_translation

    @staticmethod
    def from_dict(obj: Any) -> "ProjectedSensor":
        assert isinstance(obj, dict)
        id = Uid64(int=int(obj.get("id")))
        rig_translation = from_union(
            [ProjectedRigTranslation.from_dict, from_none], obj.get("rig_translation")
        )
        result = ProjectedSensor(id, rig_translation)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(ProjectedSensor, self).to_dict()
        result["id"] = self.id.int
        if self.rig_translation is not None:
            result["rig_translation"] = from_union(
                [lambda x: to_class(ProjectedRigTranslation, x), from_none],
                self.rig_translation,
            )
        return result


class ProjectedInputCameras(CoreItem):
    """Definition of the input cameras data in the processing CRS, which is a projected
    right-handed isometric CS.
    """

    captures: List[ProjectedCapture]
    """Captures for which there are processing CRS dependent parameters."""
    sensors: List[ProjectedSensor]
    """Sensors for which there are processing CRS dependent parameters, for example rigs. May
    contain fewer elements than the sensor list from the corresponding input cameras (or none
    if there are no rigs).
    """

    def __init__(
        self,
        captures: List[ProjectedCapture],
        sensors: List[ProjectedSensor],
        format: CoreFormat = CoreFormat.PROJECTED_INPUT_CAMERAS,
        version: VersionInfo = FormatVersion.PROJECTED_INPUT_CAMERAS,
    ) -> None:
        super(ProjectedInputCameras, self).__init__(format=format, version=version)

        self.captures = captures
        self.sensors = sensors

    @staticmethod
    def from_dict(obj: Any) -> "ProjectedInputCameras":
        base = CoreItem.from_dict(obj)

        captures = from_list(ProjectedCapture.from_dict, obj.get("captures"))
        sensors = from_list(ProjectedSensor.from_dict, obj.get("sensors"))
        result = ProjectedInputCameras(captures, sensors, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result: dict = super(ProjectedInputCameras, self).to_dict()

        result["captures"] = from_list(
            lambda x: to_class(ProjectedCapture, x), self.captures
        )
        result["sensors"] = from_list(
            lambda x: to_class(ProjectedSensor, x), self.sensors
        )
        return result


format_and_version_to_type[
    (CoreFormat.PROJECTED_INPUT_CAMERAS, FormatVersion.PROJECTED_INPUT_CAMERAS)
] = ProjectedInputCameras
