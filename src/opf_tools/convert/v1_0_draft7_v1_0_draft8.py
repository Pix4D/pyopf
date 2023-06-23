import copy
import re
from pathlib import Path
from typing import Any

import pyopf.legacy as legacy
from pyopf.cameras import (
    CalibratedCameras,
    CameraList,
    InputCameras,
    ProjectedInputCameras,
)
from pyopf.cps import InputControlPoints
from pyopf.pointcloud import GlTFPointCloud
from pyopf.project import Metadata, ProjectObjects
from pyopf.types import VersionInfo

camera_id_regex = re.compile("^0x[0-9a-fA-F]{16}$")
camera_id_keys = {"id", "sensor_id", "capture_id", "reference_camera_id", "camera_id"}


def switch_ids(obj: dict | list):
    if isinstance(obj, list):
        for o in obj:
            switch_ids(o)
    elif isinstance(obj, dict):
        for k, value in obj.items():
            if isinstance(value, dict) or isinstance(value, list):
                switch_ids(value)
            elif k in camera_id_keys:
                if type(value) == int and value >= 0 and value < 2**64:
                    obj[k] = "0x%016X" % value
                elif type(value) == str and camera_id_regex.match(value):
                    obj[k] = int(value, 16)


def convert(obj: Any, new_class: type, version: str) -> Any:

    d = obj.to_dict()

    switch_ids(d)

    d["version"] = version
    new_obj = new_class.from_dict(d)

    if hasattr(obj, "metadata"):
        new_obj.metadata = obj.metadata

    return new_obj


def convert_list(obj: list[Any], convert_object: Any) -> list[Any]:

    if isinstance(obj, list):
        return [convert_object(o) for o in obj]


def convert_camera_list(
    obj: CameraList | legacy.camera_list_1_0_draft1.CameraList,
) -> CameraList | legacy.camera_list_1_0_draft1.CameraList:
    if isinstance(obj, CameraList):
        return convert(obj, legacy.camera_list_1_0_draft1.CameraList, "1.0-draft1")
    if isinstance(obj, legacy.camera_list_1_0_draft1.CameraList):
        return convert(obj, CameraList, "1.0-draft2")


def convert_input_cameras(
    obj: legacy.input_cameras_1_0_draft10.InputCameras
    | legacy.input_cameras_1_0_draft9.InputCameras,
) -> legacy.input_cameras_1_0_draft10.InputCameras | legacy.input_cameras_1_0_draft9.InputCameras:
    if isinstance(obj, legacy.input_cameras_1_0_draft10.InputCameras):
        for capture in obj.captures:
            for camera in capture.cameras:
                if camera.input_depth_map is not None:
                    camera.input_depth_map.version = VersionInfo(1, 0, "draft1")
        return convert(obj, legacy.input_cameras_1_0_draft9.InputCameras, "1.0-draft9")
    if isinstance(obj, legacy.input_cameras_1_0_draft9.InputCameras):
        for capture in obj.captures:
            for camera in capture.cameras:
                if camera.input_depth_map is not None:
                    camera.input_depth_map.version = VersionInfo(1, 0, "draft2")
        return convert(
            obj, legacy.input_cameras_1_0_draft10.InputCameras, "1.0-draft10"
        )


def convert_projected_input_cameras(
    obj: ProjectedInputCameras
    | legacy.projected_input_cameras_1_0_draft5.ProjectedInputCameras,
) -> ProjectedInputCameras | legacy.projected_input_cameras_1_0_draft5.ProjectedInputCameras:
    if isinstance(obj, ProjectedInputCameras):
        return convert(
            obj,
            legacy.projected_input_cameras_1_0_draft5.ProjectedInputCameras,
            "1.0-draft5",
        )
    if isinstance(obj, legacy.projected_input_cameras_1_0_draft5.ProjectedInputCameras):
        return convert(obj, ProjectedInputCameras, "1.0-draft6")


def convert_calibrated_cameras(
    obj: CalibratedCameras | legacy.calibrated_cameras_1_0_draft2.CalibratedCameras,
) -> CalibratedCameras | legacy.calibrated_cameras_1_0_draft2.CalibratedCameras:
    if isinstance(obj, CalibratedCameras):
        return convert(
            obj, legacy.calibrated_cameras_1_0_draft2.CalibratedCameras, "1.0-draft2"
        )
    if isinstance(obj, legacy.calibrated_cameras_1_0_draft2.CalibratedCameras):
        return convert(obj, CalibratedCameras, "1.0-draft3")


def convert_input_control_points(
    obj: InputControlPoints | legacy.input_control_points_1_0_draft2.InputControlPoints,
) -> InputControlPoints | legacy.input_control_points_1_0_draft2.InputControlPoints:
    if isinstance(obj, InputControlPoints):
        return convert(
            obj, legacy.input_control_points_1_0_draft2.InputControlPoints, "1.0-draft2"
        )
    if isinstance(obj, legacy.input_control_points_1_0_draft2.InputControlPoints):
        return convert(obj, InputControlPoints, "1.0-draft3")

    return obj


def convert_point_cloud(obj: GlTFPointCloud) -> GlTFPointCloud:
    # From draft7 to draft8 and viceverse we only need to swap the versions, the rest
    # is handled by GlTFPointCloud
    draft7 = VersionInfo(1, 0, "draft7")
    draft8 = VersionInfo(1, 0, "draft8")
    if obj._version == draft8:
        obj._version = draft7
    else:
        obj._version = draft8
    return obj


def convert_project(project: ProjectObjects) -> ProjectObjects:

    project.input_control_points_objs = convert_list(
        project.input_control_points_objs, convert_input_control_points
    )
    project.camera_list_objs = convert_list(
        project.camera_list_objs, convert_camera_list
    )
    project.input_cameras_objs = convert_list(
        project.input_cameras_objs, convert_input_cameras
    )
    project.projected_input_cameras_objs = convert_list(
        project.projected_input_cameras_objs, convert_projected_input_cameras
    )

    for calibration in project.calibration_objs:
        calibration.calibrated_cameras_objs = convert_list(
            calibration.calibrated_cameras_objs, convert_calibrated_cameras
        )
        calibration.point_cloud_objs = convert_list(
            calibration.point_cloud_objs, convert_point_cloud
        )

    project.point_cloud_objs = convert_list(
        project.point_cloud_objs, convert_point_cloud
    )

    return project


def v1_0_draft8_to_v1_0_draft7(
    project: ProjectObjects, base_dir: Path
) -> ProjectObjects:

    project = convert_project(copy.deepcopy(project))
    project.metadata.version = VersionInfo(1, 0, "draft7")
    return project


def v1_0_draft7_to_v1_0_draft8(
    project: ProjectObjects, base_dir: Path
) -> ProjectObjects:

    project = convert_project(copy.deepcopy(project))
    project.metadata.version = VersionInfo(1, 0, "draft8")
    return project
