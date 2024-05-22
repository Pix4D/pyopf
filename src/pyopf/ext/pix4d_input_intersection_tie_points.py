from enum import Enum
from typing import Any, List, Optional

import numpy as np

from ..formats import ExtensionFormat
from ..items import ExtensionItem
from ..types import OpfObject, VersionInfo
from ..uid64 import Uid64
from ..util import (
    from_bool,
    from_float,
    from_list,
    from_none,
    from_str,
    from_union,
    to_class,
    to_enum,
    to_float,
    vector_from_list,
)
from ..versions import format_and_version_to_type

format = ExtensionFormat("application/ext-pix4d-input-intersection-tie-points+json")
version = VersionInfo(1, 0, "draft3")


class CreationMethodType(Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class CreationMethod(OpfObject):
    """The method that was used to create the mark. A mark that is edited by a user should be
    defined as manual.
    """

    type: Optional[CreationMethodType]

    def __init__(self, type: Optional[CreationMethodType]) -> None:
        self.type = type

    @staticmethod
    def from_dict(obj: Any) -> "CreationMethod":
        assert isinstance(obj, dict)
        type = from_union([CreationMethodType, from_none], obj.get("type"))
        result = CreationMethod(type)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(CreationMethod, self).to_dict()
        if self.type is not None:
            result["type"] = to_enum(CreationMethodType, self.type)
        return result


class MarkWithSegments(OpfObject):
    """2D image mark, defined as the intersection of a set of line segments."""

    camera_id: Uid64
    """Camera ID for the image on which the mark is defined."""
    common_endpoint_px: np.ndarray  # vector of size 2
    """Pixel location of the common endpoint of all intersecting segments marked on this image."""
    creation_method: CreationMethod
    """The method that was used to create the mark. A mark that is edited by a user should be
    defined as manual.
    """
    other_endpoints_px: List[np.ndarray]  # list of vectors of size 2
    """Array of pixel locations, each of these endpoints and common_endpoint_px defines a
    segment.
    """
    accuracy: Optional[float]
    """A number representing the accuracy of the mark, used by the calibration algorithm to
    estimate the position error of the mark.
    """

    def __init__(
        self,
        camera_id: Uid64,
        common_endpoint_px: np.ndarray,
        creation_method: CreationMethod,
        other_endpoints_px: List[np.ndarray],
        accuracy: Optional[float],
    ) -> None:
        super(MarkWithSegments, self).__init__()
        self.camera_id = camera_id
        self.common_endpoint_px = common_endpoint_px
        self.creation_method = creation_method
        self.other_endpoints_px = other_endpoints_px
        self.accuracy = accuracy

    @staticmethod
    def from_dict(obj: Any) -> "MarkWithSegments":
        assert isinstance(obj, dict)
        camera_id = Uid64(int=int(obj["camera_id"]))
        common_endpoint_px = vector_from_list(obj["common_endpoint_px"], 2, 2)
        creation_method = CreationMethod.from_dict(obj["creation_method"])
        other_endpoints_px = from_list(
            lambda x: vector_from_list(x, 2, 2), obj["other_endpoints_px"]
        )
        accuracy = from_union([from_float, from_none], obj.get("accuracy"))
        result = MarkWithSegments(
            camera_id, common_endpoint_px, creation_method, other_endpoints_px, accuracy
        )
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(MarkWithSegments, self).to_dict()
        result["camera_id"] = self.camera_id.int
        result["common_endpoint_px"] = from_list(to_float, self.common_endpoint_px)
        result["creation_method"] = to_class(CreationMethod, self.creation_method)
        result["other_endpoints_px"] = from_list(
            lambda x: from_list(to_float, x), self.other_endpoints_px
        )
        if self.accuracy is not None:
            result["accuracy"] = from_union([to_float, from_none], self.accuracy)
        return result


class IntersectionTiePoint(OpfObject):

    id: str
    """A unique string that identifies the ITP amongst all control points."""
    marks: List[MarkWithSegments]
    """List of marks with line segments in the images that correspond to the projections of a 3D
    point.
    """
    modified_by_user: bool
    """If true, indicates that the ITP was modified by the user."""

    def __init__(
        self, id: str, marks: List[MarkWithSegments], modified_by_user: bool
    ) -> None:
        self.id = id
        self.marks = marks
        self.modified_by_user = modified_by_user

    @staticmethod
    def from_dict(obj: Any) -> "IntersectionTiePoint":
        assert isinstance(obj, dict)
        id = from_str(obj["id"])
        marks = from_list(MarkWithSegments.from_dict, obj["marks"])
        modified_by_user = from_bool(obj["modified_by_user"])
        result = IntersectionTiePoint(id, marks, modified_by_user)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(IntersectionTiePoint, self).to_dict()
        result["id"] = from_str(self.id)
        result["marks"] = from_list(lambda x: to_class(MarkWithSegments, x), self.marks)
        result["modified_by_user"] = from_bool(self.modified_by_user)
        return result


class Pix4DInputIntersectionTiePoints(ExtensionItem):
    """Definition of Intersection Tie Points"""

    itps: List[IntersectionTiePoint]
    """List of input ITPs."""

    def __init__(
        self,
        itps: List[IntersectionTiePoint],
        format_: ExtensionFormat = format,
        version_: VersionInfo = version,
    ) -> None:
        super(Pix4DInputIntersectionTiePoints, self).__init__(
            format=format_, version=version_
        )

        assert self.format == format
        self.itps = itps

    @staticmethod
    def from_dict(obj: Any) -> "Pix4DInputIntersectionTiePoints":
        base = ExtensionItem.from_dict(obj)
        itps = from_list(IntersectionTiePoint.from_dict, obj["itps"])
        result = Pix4DInputIntersectionTiePoints(itps, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Pix4DInputIntersectionTiePoints, self).to_dict()
        result["itps"] = from_list(
            lambda x: to_class(IntersectionTiePoint, x), self.itps
        )
        return result


format_and_version_to_type[(format, version)] = Pix4DInputIntersectionTiePoints
