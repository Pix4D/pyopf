from typing import Any

import numpy as np

from ..types import OpfObject
from ..util import from_list, to_class, to_float, vector_from_list


class RigRelativeRotation(OpfObject):
    """Input camera rig rotation relative to the reference camera."""

    angles_deg: np.ndarray  # 3D vector
    """Euler angles in degree (see convention [here](auxiliary_objects.md#rig-relatives))."""
    sigmas_deg: np.ndarray  # 3D vector
    """Measurement error (standard deviation) in degrees."""

    def __init__(
        self,
        angles_deg: np.ndarray,
        sigmas_deg: np.ndarray,
    ) -> None:
        super(RigRelativeRotation, self).__init__()
        self.angles_deg = angles_deg
        self.sigmas_deg = sigmas_deg

    @staticmethod
    def from_dict(obj: Any) -> "RigRelativeRotation":
        assert isinstance(obj, dict)
        angles_deg = vector_from_list(obj.get("angles_deg"), 3, 3)
        sigmas_deg = vector_from_list(obj.get("sigmas_deg"), 3, 3)
        result = RigRelativeRotation(angles_deg, sigmas_deg)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(RigRelativeRotation, self).to_dict()
        result["angles_deg"] = from_list(to_float, self.angles_deg)
        result["sigmas_deg"] = from_list(to_float, self.sigmas_deg)
        return result


class RigRelativeTranslation(OpfObject):
    """Input camera rig translation relative to the reference camera."""

    sigmas_m: np.ndarray
    """Measurement error (standard deviation) in meters."""
    values_m: np.ndarray
    """Relative translation of the secondary sensor in the image CS of the reference sensor in
    meters.
    """

    def __init__(self, sigmas_m: np.ndarray, values_m: np.ndarray) -> None:
        super(RigRelativeTranslation, self).__init__()
        self.sigmas_m = sigmas_m
        self.values_m = values_m

    @staticmethod
    def from_dict(obj: Any) -> "RigRelativeTranslation":
        assert isinstance(obj, dict)
        sigmas_m = vector_from_list(obj.get("sigmas_m"), 3, 3)
        values_m = vector_from_list(obj.get("values_m"), 3, 3)
        result = RigRelativeTranslation(sigmas_m, values_m)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(RigRelativeTranslation, self).to_dict()
        result["sigmas_m"] = from_list(to_float, self.sigmas_m)
        result["values_m"] = from_list(to_float, self.values_m)
        return result


class InputRigRelatives(OpfObject):
    """Input rig relatives contain the a priori knowledge about the relative translation and
    rotation of secondary cameras. Since these values are supposedly coming from a sensor
    database, the units are always meters and degrees.
    """

    rotation: RigRelativeRotation
    translation: RigRelativeTranslation

    def __init__(
        self,
        rotation: RigRelativeRotation,
        translation: RigRelativeTranslation,
    ) -> None:
        super(InputRigRelatives, self).__init__()
        self.rotation = rotation
        self.translation = translation

    @staticmethod
    def from_dict(obj: Any) -> "InputRigRelatives":
        assert isinstance(obj, dict)
        rotation = RigRelativeRotation.from_dict(obj.get("rotation"))
        translation = RigRelativeTranslation.from_dict(obj.get("translation"))
        result = InputRigRelatives(rotation, translation)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(InputRigRelatives, self).to_dict()
        result["rotation"] = to_class(RigRelativeRotation, self.rotation)
        result["translation"] = to_class(RigRelativeTranslation, self.translation)
        return result
