from typing import Any

import numpy as np

from ..types import OpfObject
from ..util import (
    from_bool,
    from_float,
    from_list,
    from_str,
    to_float,
    vector_from_list,
)


class SphericalInternals(OpfObject):
    """Parameters of the spherical camera model are described in Pix4D [knowledge base](https://support.pix4d.com/hc/en-us/articles/202559089)."""

    type = "spherical"

    principal_point_px: np.ndarray  # 2D vector
    """Principal point with respect to the top left corner in pixels given as `[number, number]`."""

    def __init__(
        self,
        principal_point_px: np.ndarray,
    ) -> None:
        super(SphericalInternals, self).__init__()
        self.principal_point_px = principal_point_px

    @staticmethod
    def from_dict(obj: Any) -> "SphericalInternals":
        assert isinstance(obj, dict)
        assert obj.get("type") == SphericalInternals.type

        principal_point_px = vector_from_list(obj.get("principal_point_px"), 2, 2)
        result = SphericalInternals(principal_point_px)
        result._extract_unknown_properties_and_extensions(obj, ["type"])
        return result

    def to_dict(self) -> dict:
        result = super(SphericalInternals, self).to_dict()
        result["principal_point_px"] = from_list(to_float, self.principal_point_px)
        result["type"] = from_str(self.type)
        return result


class PerspectiveInternals(OpfObject):
    """Parameters of the perspective camera model as described in Pix4D [knowledge base](https://support.pix4d.com/hc/en-us/articles/202559089#label1)."""

    principal_point_px: np.ndarray  # 2D vector
    """Principal point with respect to the top left corner in pixels given as `[number, number]`."""
    focal_length_px: float
    """Focal length in pixels."""
    radial_distortion: np.ndarray  # 3D vector
    """The radial distortion coefficients (R1, R2, R3)."""
    tangential_distortion: np.ndarray  # 2D vector
    """The tangential distortion coefficients (T1, T2)."""

    type = "perspective"

    def __init__(
        self,
        principal_point_px: np.ndarray,
        focal_length_px: float,
        radial_distortion: np.ndarray,
        tangential_distortion: np.ndarray,
    ) -> None:
        super(PerspectiveInternals, self).__init__()
        self.focal_length_px = focal_length_px
        self.principal_point_px = principal_point_px
        self.radial_distortion = radial_distortion
        self.tangential_distortion = tangential_distortion

    @staticmethod
    def from_dict(obj: Any) -> "PerspectiveInternals":
        assert isinstance(obj, dict)
        assert obj.get("type") == PerspectiveInternals.type

        focal_length_px = from_float(obj.get("focal_length_px"))
        principal_point_px = vector_from_list(obj.get("principal_point_px"), 2, 2)
        radial_distortion = vector_from_list(obj.get("radial_distortion"), 3, 3)
        tangential_distortion = vector_from_list(obj.get("tangential_distortion"), 2, 2)

        result = PerspectiveInternals(
            principal_point_px,
            focal_length_px,
            radial_distortion,
            tangential_distortion,
        )
        result._extract_unknown_properties_and_extensions(obj, ["type"])
        return result

    def to_dict(self) -> dict:
        result = super(PerspectiveInternals, self).to_dict()
        result["focal_length_px"] = from_float(self.focal_length_px)
        result["principal_point_px"] = from_list(to_float, self.principal_point_px)
        result["radial_distortion"] = from_list(to_float, self.radial_distortion)
        result["tangential_distortion"] = from_list(
            to_float, self.tangential_distortion
        )
        result["type"] = from_str(self.type)
        return result


class FisheyeInternals(OpfObject):
    """Parameters of the fisheye camera model as described in Pix4D [knowledge base](https://support.pix4d.com/hc/en-us/articles/202559089#label2)."""

    principal_point_px: np.ndarray  # 2D vector
    """Principal point with respect to the top left corner in pixels given as `[number, number]`."""
    type = "fisheye"
    affine: np.ndarray  # 4D vector
    """Affine transformation parameters as [ c d; e f ]"""
    is_p0_zero: bool
    """If true, it is prior knowledge that the first polynomial coefficient is equal to zero and
    should be kept zero.
    """
    is_symmetric_affine: bool
    """If true, it is prior knowledge that the affine matrix is symmetric (that is, c=f and
    d=e=0) and should be kept symmetric.
    """
    polynomial: np.ndarray
    """The coefficients of the distortion polynomial."""

    def __init__(
        self,
        principal_point_px: np.ndarray,
        affine: np.ndarray,
        is_p0_zero: bool,
        is_symmetric_affine: bool,
        polynomial: np.ndarray,
    ) -> None:
        super(FisheyeInternals, self).__init__()
        self.principal_point_px = principal_point_px
        self.affine = affine
        self.is_p0_zero = is_p0_zero
        self.is_symmetric_affine = is_symmetric_affine
        self.polynomial = polynomial

    @staticmethod
    def from_dict(obj: Any) -> "FisheyeInternals":
        assert isinstance(obj, dict)
        assert obj.get("type") == FisheyeInternals.type

        principal_point_px = vector_from_list(obj.get("principal_point_px"), 2, 2)
        affine = vector_from_list(obj.get("affine"), 4, 4)
        is_p0_zero = from_bool(obj.get("is_p0_zero"))
        is_symmetric_affine = from_bool(obj.get("is_symmetric_affine"))
        polynomial = np.array(from_list(from_float, obj.get("polynomial")))

        result = FisheyeInternals(
            principal_point_px,
            affine,
            is_p0_zero,
            is_symmetric_affine,
            polynomial,
        )
        result._extract_unknown_properties_and_extensions(obj, ["type"])
        return result

    def to_dict(self) -> dict:
        result = super(FisheyeInternals, self).to_dict()
        result["principal_point_px"] = from_list(to_float, self.principal_point_px)
        result["affine"] = from_list(to_float, self.affine)
        result["is_p0_zero"] = from_bool(self.is_p0_zero)
        result["is_symmetric_affine"] = from_bool(self.is_symmetric_affine)
        result["polynomial"] = from_list(to_float, self.polynomial)
        result["type"] = from_str(self.type)
        return result


Internals = FisheyeInternals | PerspectiveInternals | SphericalInternals
