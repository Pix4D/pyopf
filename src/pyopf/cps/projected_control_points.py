from typing import Any, Dict, List, Optional

import numpy as np

from ..formats import CoreFormat
from ..items import CoreItem
from ..types import OpfObject, VersionInfo
from ..util import (
    from_bool,
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


class ProjectedGcp(OpfObject):
    """3D position in the processing CRS."""

    coordinates: np.ndarray
    id: str
    """A string identifier that matches the correspondent input GCP."""
    sigmas: np.ndarray
    """Standard deviation of the 3D position in processing CRS units."""

    def __init__(
        self,
        id: str,
        coordinates: np.ndarray,
        sigmas: np.ndarray,
    ) -> None:
        super(ProjectedGcp, self).__init__()
        self.id = id
        self.coordinates = coordinates
        self.sigmas = sigmas

    @staticmethod
    def from_dict(obj: Any) -> "ProjectedGcp":
        assert isinstance(obj, dict)

        coordinates = vector_from_list(obj.get("coordinates"), 3, 3)
        sigmas = vector_from_list(obj.get("sigmas"), 3, 3)
        id = from_str(obj.get("id"))

        result = ProjectedGcp(id, coordinates, sigmas)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(ProjectedGcp, self).to_dict()
        result["id"] = str(self.id)
        result["coordinates"] = from_list(to_float, self.coordinates)
        result["sigmas"] = from_list(to_float, self.sigmas)

        return result


class ProjectedControlPoints(CoreItem):
    """Definition of projected control points, which are the input control points with
    coordinates expressed in the processing CRS
    """

    projected_gcps: List[ProjectedGcp]
    """List of projected GCPs."""

    def __init__(
        self,
        projected_gcps: List[ProjectedGcp],
        format: CoreFormat = CoreFormat.PROJECTED_CONTROL_POINTS,
        version: VersionInfo = FormatVersion.PROJECTED_CONTROL_POINTS,
    ) -> None:
        super().__init__(format=format, version=version)

        assert self.format == CoreFormat.PROJECTED_CONTROL_POINTS

        self.projected_gcps = projected_gcps

    @staticmethod
    def from_dict(obj: Any) -> "ProjectedControlPoints":
        base = CoreItem.from_dict(obj)
        projected_gcps = from_list(ProjectedGcp.from_dict, obj.get("projected_gcps"))
        result = ProjectedControlPoints(projected_gcps, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(ProjectedControlPoints, self).to_dict()
        result["projected_gcps"] = from_list(
            lambda x: to_class(ProjectedGcp, x), self.projected_gcps
        )
        return result


format_and_version_to_type[
    (CoreFormat.PROJECTED_CONTROL_POINTS, FormatVersion.PROJECTED_CONTROL_POINTS)
] = ProjectedControlPoints
