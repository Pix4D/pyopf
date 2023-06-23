from pyopf.project import Project
from pyopf.types import CoreFormat, VersionInfo
from pyopf.versions import format_and_version_to_type

from . import (
    calibrated_cameras_1_0_draft2,
    camera_list_1_0_draft1,
    input_cameras_1_0_draft8,
    input_cameras_1_0_draft9,
    input_cameras_1_0_draft10,
    input_cameras_1_0_draft11,
    input_control_points_1_0_draft2,
    project_1_0_draft9_aliases_1_0,
    projected_input_cameras_1_0_draft5,
    tracks,
)

format_and_version_to_type[(CoreFormat.PROJECT, VersionInfo(1, 0, "draft6"))] = Project
format_and_version_to_type[(CoreFormat.PROJECT, VersionInfo(1, 0, "draft7"))] = Project
format_and_version_to_type[(CoreFormat.PROJECT, VersionInfo(1, 0, "draft8"))] = Project
format_and_version_to_type[(CoreFormat.PROJECT, VersionInfo(1, 0, "draft9"))] = Project
