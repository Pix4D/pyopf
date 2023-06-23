from typing import Any, Dict, List, Optional

import numpy as np

from ..formats import CoreFormat
from ..items import CoreItem
from ..types import OpfObject, VersionInfo
from ..uid64 import Uid64
from ..util import from_float, from_list, to_class, to_float, vector_from_list
from ..versions import FormatVersion, format_and_version_to_type


class RigidTransformationWithScaling(OpfObject):
    """Rigid transform

    Definition of a rigid transformation with rotation, translation, and scaling. Transforms
    input points p to output points p' via p' = scale * rotation * p + translation.
    """

    rotation_deg: np.ndarray
    """Rotation as Euler angles in degree (see convention for camera rig-relative rotations)"""
    scale: float
    """Scale"""
    translation: np.ndarray
    """Translation in units of the processing CRS."""

    def __init__(
        self, rotation_deg: np.ndarray, scale: float, translation: np.ndarray
    ) -> None:
        self.rotation_deg = rotation_deg
        self.scale = scale
        self.translation = translation

    @staticmethod
    def from_dict(obj: Any) -> "RigidTransformationWithScaling":
        assert isinstance(obj, dict)
        rotation_deg = vector_from_list(obj.get("rotation_deg"), 3, 3)
        scale = from_float(obj.get("scale"))
        translation = vector_from_list(obj.get("translation"), 3, 3)
        result = RigidTransformationWithScaling(rotation_deg, scale, translation)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(RigidTransformationWithScaling, self).to_dict()
        result["rotation_deg"] = from_list(to_float, self.rotation_deg)
        result["scale"] = to_float(self.scale)
        result["translation"] = from_list(to_float, self.translation)
        return result


class GpsBias(CoreItem):
    """For projects processed with both camera GPS and GCPs, the GPS bias describes a transform
    from the (GCP-adjusted) camera output positions to the prior camera GPS positions. For an
    output camera point p, a camera GPS point p' is computed as p' = RigidTransformation(p).
    Note that both the GPS and camera positions are in the processing CRS. A GPS bias is a
    rigid transformation with rotation, translation, and scaling.
    """

    transform: RigidTransformationWithScaling
    """Rigid transform"""

    def __init__(
        self,
        transform: RigidTransformationWithScaling,
        format: CoreFormat = CoreFormat.GPS_BIAS,
        version: VersionInfo = FormatVersion.GPS_BIAS,
    ) -> None:
        super(GpsBias, self).__init__(format=format, version=version)
        self.transform = transform

    @staticmethod
    def from_dict(obj: Any) -> "GpsBias":
        base = CoreItem.from_dict(obj)
        transform = RigidTransformationWithScaling.from_dict(obj.get("transform"))
        result = GpsBias(transform, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result: dict = super(GpsBias, self).to_dict()
        result["transform"] = to_class(RigidTransformationWithScaling, self.transform)
        return result


format_and_version_to_type[(CoreFormat.GPS_BIAS, FormatVersion.GPS_BIAS)] = GpsBias
