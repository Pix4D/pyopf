import copy
from pathlib import Path
from typing import Any

import pyopf.legacy as legacy
from pyopf.cameras import InputCameras
from pyopf.project import ProjectObjects
from pyopf.types import VersionInfo


def convert(obj: Any, new_class: type, version: str) -> Any:

    d = obj.to_dict()

    d["version"] = version
    new_obj = new_class.from_dict(d)

    if hasattr(obj, "metadata"):
        new_obj.metadata = obj.metadata

    return new_obj


def convert_list(obj: list[Any], convert_object: Any) -> list[Any]:

    if isinstance(obj, list):
        return [convert_object(o) for o in obj]


def convert_input_cameras(
    obj: InputCameras | legacy.input_cameras_1_0_draft11.InputCameras,
) -> InputCameras | legacy.input_cameras_1_0_draft11.InputCameras:

    if isinstance(obj, legacy.input_cameras_1_0_draft11.InputCameras):
        return convert(obj, InputCameras, "1.0")
    if isinstance(obj, InputCameras):
        return convert(
            obj, legacy.input_cameras_1_0_draft11.InputCameras, "1.0-draft11"
        )
    assert False, "unreachable"


def v1_0_draft9_to_latest(project: ProjectObjects, base_dir: Path) -> ProjectObjects:

    project = copy.deepcopy(project)
    project.input_cameras_objs = convert_list(
        project.input_cameras_objs, convert_input_cameras
    )
    project.metadata.version = VersionInfo(1, 0)
    return project


def latest_to_v1_0_draft9(project: ProjectObjects, base_dir: Path) -> ProjectObjects:

    project = copy.deepcopy(project)
    project.input_cameras_objs = convert_list(
        project.input_cameras_objs, convert_input_cameras
    )
    project.metadata.version = VersionInfo(1, 0, "draft9")
    return project
