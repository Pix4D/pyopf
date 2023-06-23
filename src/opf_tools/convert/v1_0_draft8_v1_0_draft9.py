import copy
import warnings
from pathlib import Path
from typing import Any

import pyopf.legacy as legacy
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
    obj: legacy.input_cameras_1_0_draft11.InputCameras
    | legacy.input_cameras_1_0_draft10.InputCameras,
) -> legacy.input_cameras_1_0_draft11.InputCameras | legacy.input_cameras_1_0_draft10.InputCameras:

    if isinstance(obj, legacy.input_cameras_1_0_draft11.InputCameras):
        for capture in obj.captures:
            if str(capture.rig_model_source) == str(
                legacy.input_cameras_1_0_draft11.RigModelSource.USER
            ):
                raise ValueError(
                    '"user" is not a valid value for "rig_model_source" in <= 1.0-draft8'
                )
            for camera in capture.cameras:
                if str(camera.model_source) == str(
                    legacy.input_cameras_1_0_draft11.ModelSource.USER
                ):
                    raise ValueError(
                        '"user" is not a valid value for "model_source" in <= 1.0-draft8'
                    )

        return convert(
            obj, legacy.input_cameras_1_0_draft10.InputCameras, "1.0-draft10"
        )
    if isinstance(obj, legacy.input_cameras_1_0_draft10.InputCameras):
        return convert(
            obj, legacy.input_cameras_1_0_draft11.InputCameras, "1.0-draft11"
        )
    assert False, "unreachable"


def v1_0_draft9_to_v1_0_draft8(
    project: ProjectObjects, base_dir: Path
) -> ProjectObjects:

    project = copy.deepcopy(project)

    project.input_cameras_objs = convert_list(
        project.input_cameras_objs, convert_input_cameras
    )

    # The GPS bias resource doesn't exist in draft8
    warned_about_gps_bias = False
    for calibration in project.calibration_objs:
        if calibration.gps_bias is not None and not warned_about_gps_bias:
            warnings.warn("GPS bias is not a valid resource in <=1.0-draft8, dropping")
            warned_about_gps_bias = True
        calibration.gps_bias = None

    project.metadata.version = VersionInfo(1, 0, "draft8")
    return project


def v1_0_draft8_to_v1_0_draft9(
    project: ProjectObjects, base_dir: Path
) -> ProjectObjects:

    project = copy.deepcopy(project)

    project.input_cameras_objs = convert_list(
        project.input_cameras_objs, convert_input_cameras
    )

    project.metadata.version = VersionInfo(1, 0, "draft9")
    return project
