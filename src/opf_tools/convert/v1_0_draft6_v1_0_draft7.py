import copy
import re
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag

import pyopf.legacy as legacy
from pyopf.project import (
    CoreProjectItemType,
    Metadata,
    ProjectObjects,
    ProjectSource,
)
from pyopf.types import VersionInfo
from pyopf.uid64 import uid64

PIX4D_input_depth_map = "PIX4D_input_depth_map"


def input_cameras_v1_0_draft6_to_v1_0_draft7(
    original_input_cameras: legacy.input_cameras_1_0_draft8.InputCameras,
    camera_list: list[legacy.camera_list_1_0_draft1.CameraData],
) -> legacy.input_cameras_1_0_draft9.InputCameras:

    input_cameras = original_input_cameras.to_dict()

    for capture in input_cameras["captures"]:
        for camera in capture["cameras"]:

            uri = camera.pop("image_uri")
            try:
                page_index = camera.pop("page_index")
                if page_index != 0:
                    uri += f"#{page_index}"
            except KeyError:
                pass

            camera_list.append(
                legacy.camera_list_1_0_draft1.CameraData(id=camera["id"], uri=uri)
            )

            try:
                depth = camera["depth"]

                depth_id = uid64()
                camera_list.append(
                    legacy.camera_list_1_0_draft1.CameraData(
                        id=depth_id, uri=depth["uri"]
                    )
                )
                extension: dict[str, Any] = {
                    "version": "1.0-draft1",
                    "id": str(depth_id),
                }

                try:
                    confidence = depth["confidence_uri"]
                    confidence_id = uid64()
                    min_confidence = depth["min_confidence"]
                    max_confidence = depth["max_confidence"]
                    extension["confidence"] = {
                        "id": str(confidence_id),
                        "min": min_confidence,
                        "max": max_confidence,
                    }
                    camera.setdefault("extensions", {})
                    camera["extensions"][PIX4D_input_depth_map] = extension
                    camera_list.append(
                        legacy.camera_list_1_0_draft1.CameraData(
                            id=confidence_id, uri=confidence
                        )
                    )
                except KeyError:
                    # We don't carea about malformed inputs where confidence_uri is present but either
                    # min_confidence or max_confidence is missing
                    pass

                del camera["depth"]
            except KeyError:
                pass

    input_cameras["version"] = "1.0-draft9"
    input_cameras = legacy.input_cameras_1_0_draft9.InputCameras.from_dict(
        input_cameras
    )
    input_cameras.metadata = copy.deepcopy(original_input_cameras.metadata)
    return input_cameras


def input_cameras_v1_0_draft7_to_v1_0_draft6(
    original_input_cameras: legacy.input_cameras_1_0_draft9.InputCameras,
    id_to_uri: dict[str, str],
) -> legacy.input_cameras_1_0_draft8.InputCameras:

    warned_about_threshold = False

    input_cameras = original_input_cameras.to_dict()
    for capture in input_cameras["captures"]:
        for camera in capture["cameras"]:

            uri = urldefrag(id_to_uri[camera["id"]])
            page_index_match = re.match(".*page=([0-9]+)\\D*", uri.fragment)
            camera["image_uri"] = uri.url
            camera["page_index"] = (
                int(page_index_match[1]) if page_index_match is not None else 0
            )

            try:
                depth = camera["extensions"][PIX4D_input_depth_map]
                camera["depth"] = depth

                depth["uri"] = id_to_uri[depth["id"]]
                del depth["id"]
                del depth["version"]

                try:
                    confidence = depth["confidence"]
                    depth["confidence_uri"] = id_to_uri[confidence["id"]]
                    depth["max_confidence"] = confidence["max"]
                    depth["min_confidence"] = confidence["min"]
                    if "threshold" in confidence and not warned_about_threshold:
                        warnings.warn(
                            "Depth map confidence threshold can't be represented in 1.0-draft6, dropping"
                        )
                        warned_about_threshold = True
                    del depth["confidence"]
                except KeyError:
                    pass

                del camera["extensions"][PIX4D_input_depth_map]
                if len(camera["extensions"]) == 0:
                    del camera["extensions"]
            except KeyError:
                pass

    input_cameras["version"] = "1.0-draft8"
    input_cameras = legacy.input_cameras_1_0_draft8.InputCameras.from_dict(
        input_cameras
    )

    if original_input_cameras.metadata is None:
        raise ValueError("InputCameras metadata is missing")

    input_cameras.metadata = copy.deepcopy(original_input_cameras.metadata)

    if isinstance(input_cameras.metadata.sources, list):
        input_cameras.metadata.sources = [
            source
            for source in input_cameras.metadata.sources
            if source.type != CoreProjectItemType.CAMERA_LIST
        ]
    else:
        del input_cameras.metadata.sources.camera_list
    return input_cameras


def v1_0_draft6_to_v1_0_draft7(
    project: ProjectObjects, base_dir: Path
) -> ProjectObjects:

    project = copy.deepcopy(project)

    if len(project.input_cameras_objs) != 0:

        camera_list = []
        project.input_cameras_objs = [
            input_cameras_v1_0_draft6_to_v1_0_draft7(cameras, camera_list)
            for cameras in project.input_cameras_objs
        ]
        camera_list = legacy.camera_list_1_0_draft1.CameraList(camera_list)
        camera_list.metadata = Metadata(type=CoreProjectItemType.CAMERA_LIST)

        for input_cameras in project.input_cameras_objs:
            if input_cameras.metadata is None:
                raise ValueError("InputCameras metadata is missing")

            sources = input_cameras.metadata.sources
            if isinstance(sources, list):
                sources.append(
                    ProjectSource(
                        id=camera_list.metadata.id, type=camera_list.metadata.type
                    )
                )
            else:
                sources.camera_list = camera_list

        project.camera_list_objs = [camera_list]

    if len(project.calibration_objs) != 0:
        for calibration in project.calibration_objs:
            for tracks in calibration.track_positions_objs:
                calibration.point_cloud_objs.append(
                    legacy.tracks.tracks_to_gltf(tracks)
                )
            del calibration.track_positions_objs

    project.metadata.version = VersionInfo(1, 0, "draft7")

    return project


def v1_0_draft7_to_v1_0_draft6(
    project: ProjectObjects, base_dir: Path
) -> ProjectObjects:

    project = copy.deepcopy(project)

    if len(project.camera_list_objs) != 0:

        if project.input_cameras is None:
            warnings.warn(
                "Input project lacks input_camears, camera_list data will be lost"
            )
            project.camera_list_objs = []

        else:
            id_to_uri = {
                str(camera.id): camera.uri
                for camera_list in project.camera_list_objs
                for camera in camera_list.cameras
            }

            project.input_cameras_objs = [
                input_cameras_v1_0_draft7_to_v1_0_draft6(cameras, id_to_uri)
                for cameras in project.input_cameras_objs
            ]
            project.camera_list_objs = []

    if len(project.calibration_objs) != 0:
        for calibration in project.calibration_objs:
            setattr(calibration, "track_positions_objs", [])
            for point_cloud in calibration.point_cloud_objs:
                calibration.track_positions_objs.append(
                    legacy.tracks.gltf_to_tracks(point_cloud)
                )
            calibration.point_cloud_objs = []

    project.metadata.version = VersionInfo(1, 0, "draft6")

    return project
