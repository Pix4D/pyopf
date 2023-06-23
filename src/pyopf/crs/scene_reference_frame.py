from typing import Any

import numpy as np

from ..formats import CoreFormat
from ..items import CoreItem
from ..types import OpfObject, VersionInfo
from ..util import from_bool, from_list, to_class, to_float, vector_from_list
from ..versions import FormatVersion, format_and_version_to_type
from .crs import Crs


class BaseToTranslatedCanonicalCrsTransform(OpfObject):
    """Per axis scaling factors to make the base CRS isometric."""

    scale: np.ndarray  # array of size 3
    shift: np.ndarray  # array of size 3
    """Translation from the canonical CRS to a recentered reference frame suitable for
    processing and visualization.
    """
    swap_xy: bool
    """true if and only if the base CRS is left-handed."""

    def __init__(
        self,
        scale: np.ndarray,
        shift: np.ndarray,
        swap_xy: bool,
    ) -> None:
        self.scale = scale
        super(BaseToTranslatedCanonicalCrsTransform, self).__init__()
        self.shift = shift
        self.swap_xy = swap_xy

    @staticmethod
    def from_dict(obj: Any) -> "BaseToTranslatedCanonicalCrsTransform":
        assert isinstance(obj, dict)
        scale = vector_from_list(obj.get("scale"), 3, 3)
        shift = vector_from_list(obj.get("shift"), 3, 3)
        swap_xy = from_bool(obj.get("swap_xy"))
        result = BaseToTranslatedCanonicalCrsTransform(scale, shift, swap_xy)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(BaseToTranslatedCanonicalCrsTransform, self).to_dict()
        result["scale"] = from_list(to_float, self.scale)
        result["shift"] = from_list(to_float, self.shift)
        result["swap_xy"] = from_bool(self.swap_xy)
        return result


class SceneReferenceFrame(CoreItem):
    """An object that specifies a base Cartesian CRS and the transformation parameters to a
    translated canonical form suitable for processing and visualization.
    """

    base_to_canonical: BaseToTranslatedCanonicalCrsTransform
    crs: Crs

    def __init__(
        self,
        base_to_canonical: BaseToTranslatedCanonicalCrsTransform,
        crs: Crs,
        format: CoreFormat = CoreFormat.SCENE_REFERENCE_FRAME,
        version: VersionInfo = FormatVersion.SCENE_REFERENCE_FRAME,
    ) -> None:
        super(SceneReferenceFrame, self).__init__(format=format, version=version)

        assert self.format == CoreFormat.SCENE_REFERENCE_FRAME
        self.base_to_canonical = base_to_canonical
        self.crs = crs

    @staticmethod
    def from_dict(obj: Any) -> "SceneReferenceFrame":
        base = CoreItem.from_dict(obj)
        base_to_canonical = BaseToTranslatedCanonicalCrsTransform.from_dict(
            obj.get("base_to_canonical")
        )
        crs = Crs.from_dict(obj.get("crs"))
        result = SceneReferenceFrame(base_to_canonical, crs, base.format, base.version)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(SceneReferenceFrame, self).to_dict()
        result["base_to_canonical"] = to_class(
            BaseToTranslatedCanonicalCrsTransform, self.base_to_canonical
        )
        result["crs"] = to_class(Crs, self.crs)
        return result


format_and_version_to_type[
    (CoreFormat.SCENE_REFERENCE_FRAME, FormatVersion.SCENE_REFERENCE_FRAME)
] = SceneReferenceFrame
