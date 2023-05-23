from typing import Any, Dict, List, Optional

import numpy as np

from ..types import OpfObject
from ..util import (
    from_float,
    from_list,
    from_none,
    from_union,
    to_class,
    to_float,
    vector_from_list,
)
from .crs import Crs


class Geolocation(OpfObject):
    """Geolocation information"""

    coordinates: np.ndarray  # 3D vector
    """3D coordinates of a point using the same axis convention as declared by the CRS, i.e.,
    the X, Y axes are **not** always Easting-Northing.
    """
    crs: Crs
    sigmas: np.ndarray  # 3D vector
    """Standard deviation of a measured position. For geographic CRSs, all units are meters. For
    Cartesian CRSs, the units are given by the 3D promoted definition of the axes (see the
    specification of the coordinate reference system above for the definition of the
    promotion).
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        crs: Crs,
        sigmas: np.ndarray,
    ) -> None:
        super(Geolocation, self).__init__()
        self.coordinates = coordinates
        self.crs = crs
        self.sigmas = sigmas

    @staticmethod
    def from_dict(obj: Any) -> "Geolocation":
        assert isinstance(obj, dict)
        coordinates = vector_from_list(obj.get("coordinates"), 3, 3)
        crs = Crs.from_dict(obj.get("crs"))
        sigmas = vector_from_list(obj.get("sigmas"), 3, 3)
        result = Geolocation(coordinates, crs, sigmas)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result: dict = super(Geolocation, self).to_dict()
        result["coordinates"] = from_list(to_float, self.coordinates)
        result["crs"] = to_class(Crs, self.crs)
        result["sigmas"] = from_list(to_float, self.sigmas)
        return result
