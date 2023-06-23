from typing import Any, List

import numpy as np

from ..formats import CoreFormat
from ..items import CoreItem
from ..types import OpfObject, VersionInfo
from ..util import (
    from_float,
    from_list,
    from_str,
    to_class,
    to_float,
    vector_from_list,
)
from ..versions import FormatVersion, format_and_version_to_type


class OrientationConstraint(OpfObject):
    """A unique string that identifies the constraint."""

    id_from: str
    id: str
    """A string identifier that matches the correspondent input control point."""
    id_to: str
    """A string identifier that matches the correspondent input control point."""
    sigma_deg: float
    """Accuracy of the alignment expressed as the angle between the unit_vector and the to-from
    vector in degrees.
    """
    unit_vector: np.ndarray  # 3D vector
    """Direction in which the to-from vector has to point given as a unit vector in the
    processing CRS.
    """

    def __init__(
        self,
        id: str,
        id_from: str,
        id_to: str,
        unit_vector: np.ndarray,
        sigma_deg: float,
    ) -> None:
        super(OrientationConstraint, self).__init__()
        self.id = id
        self.id_from = id_from
        self.id_to = id_to
        self.sigma_deg = sigma_deg
        self.unit_vector = unit_vector

    @staticmethod
    def from_dict(obj: Any) -> "OrientationConstraint":
        assert isinstance(obj, dict)
        id = from_str(obj.get("id"))
        id_from = from_str(obj.get("id_from"))
        id_to = from_str(obj.get("id_to"))
        sigma_deg = from_float(obj.get("sigma_deg"))
        unit_vector = vector_from_list(obj.get("unit_vector"), 3, 3)
        result = OrientationConstraint(id, id_from, id_to, unit_vector, sigma_deg)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result: dict = super(OrientationConstraint, self).to_dict()
        result["id"] = from_str(self.id)
        result["id_from"] = from_str(self.id_from)
        result["id_to"] = from_str(self.id_to)
        result["sigma_deg"] = to_float(self.sigma_deg)
        result["unit_vector"] = from_list(to_float, self.unit_vector)
        return result


class ScaleConstraint(OpfObject):
    """Distance between the two control points in the processing CRS."""

    id: str
    distance: float
    """A unique string that identifies the constraint."""
    id_from: str
    """A string identifier that matches the correspondent input control point."""
    id_to: str
    """A string identifier that matches the correspondent input control point."""
    sigma: float
    """Distance accuracy in the processing CRS."""

    def __init__(
        self,
        id: str,
        id_from: str,
        id_to: str,
        distance: float,
        sigma: float,
    ) -> None:
        super(ScaleConstraint, self).__init__()
        self.distance = distance
        self.id = id
        self.id_from = id_from
        self.id_to = id_to
        self.sigma = sigma

    @staticmethod
    def from_dict(obj: Any) -> "ScaleConstraint":
        assert isinstance(obj, dict)
        distance = from_float(obj.get("distance"))
        id = from_str(obj.get("id"))
        id_from = from_str(obj.get("id_from"))
        id_to = from_str(obj.get("id_to"))
        sigma = from_float(obj.get("sigma"))
        result = ScaleConstraint(id, id_from, id_to, distance, sigma)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(ScaleConstraint, self).to_dict()
        result["distance"] = to_float(self.distance)
        result["id"] = from_str(self.id)
        result["id_from"] = from_str(self.id_from)
        result["id_to"] = from_str(self.id_to)
        result["sigma"] = to_float(self.sigma)
        return result


class Constraints(CoreItem):
    """Scale and orientation constraints"""

    orientation_constraints: List[OrientationConstraint]
    """List of orientation constraints."""
    scale_constraints: List[ScaleConstraint]
    """List of scale constraints."""

    def __init__(
        self,
        orientation_constraints: List[OrientationConstraint],
        scale_constraints: List[ScaleConstraint],
        format: CoreFormat = CoreFormat.CONSTRAINTS,
        version: VersionInfo = FormatVersion.CONSTRAINTS,
    ) -> None:
        super(Constraints, self).__init__(format=format, version=version)

        assert self.format == CoreFormat.CONSTRAINTS
        self.orientation_constraints = orientation_constraints
        self.scale_constraints = scale_constraints

    @staticmethod
    def from_dict(obj: Any) -> "Constraints":
        base = CoreItem.from_dict(obj)

        orientation_constraints = from_list(
            OrientationConstraint.from_dict, obj.get("orientation_constraints")
        )
        scale_constraints = from_list(
            ScaleConstraint.from_dict, obj.get("scale_constraints")
        )
        result = Constraints(
            orientation_constraints, scale_constraints, base.format, base.version
        )
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Constraints, self).to_dict()
        result["orientation_constraints"] = from_list(
            lambda x: to_class(OrientationConstraint, x), self.orientation_constraints
        )
        result["scale_constraints"] = from_list(
            lambda x: to_class(ScaleConstraint, x), self.scale_constraints
        )
        return result


format_and_version_to_type[
    (CoreFormat.CONSTRAINTS, FormatVersion.CONSTRAINTS)
] = Constraints
