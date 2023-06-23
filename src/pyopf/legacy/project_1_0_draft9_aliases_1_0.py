from ..cameras import (
    CalibratedCameras,
    CameraList,
    GpsBias,
    InputCameras,
    ProjectedInputCameras,
)
from ..cps import (
    CalibratedControlPoints,
    Constraints,
    InputControlPoints,
    ProjectedControlPoints,
)
from ..crs import SceneReferenceFrame
from ..formats import CoreFormat
from ..types import VersionInfo
from ..versions import FormatVersion, format_and_version_to_type

format_and_version_to_type[
    (CoreFormat.SCENE_REFERENCE_FRAME, VersionInfo(1, 0, "draft2"))
] = SceneReferenceFrame

format_and_version_to_type[
    (CoreFormat.CALIBRATED_CAMERAS, VersionInfo(1, 0, "draft3"))
] = CalibratedCameras

format_and_version_to_type[
    (CoreFormat.CAMERA_LIST, VersionInfo(1, 0, "draft2"))
] = CameraList

format_and_version_to_type[
    (CoreFormat.INPUT_CAMERAS, VersionInfo(1, 0, "draft11"))
] = InputCameras

format_and_version_to_type[
    (CoreFormat.PROJECTED_INPUT_CAMERAS, VersionInfo(1, 0, "draft6"))
] = ProjectedInputCameras

format_and_version_to_type[(CoreFormat.GPS_BIAS, VersionInfo(1, 0, "draft1"))] = GpsBias

format_and_version_to_type[
    (CoreFormat.CALIBRATED_CONTROL_POINTS, VersionInfo(1, 0, "draft3"))
] = CalibratedControlPoints

format_and_version_to_type[
    (CoreFormat.CONSTRAINTS, VersionInfo(1, 0, "draft2"))
] = Constraints

format_and_version_to_type[
    (CoreFormat.INPUT_CONTROL_POINTS, VersionInfo(1, 0, "draft3"))
] = InputControlPoints

format_and_version_to_type[
    (CoreFormat.PROJECTED_CONTROL_POINTS, VersionInfo(1, 0, "draft2"))
] = InputControlPoints
