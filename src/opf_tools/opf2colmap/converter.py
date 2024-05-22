import argparse
import os
import shutil
from urllib.parse import ParseResult, urlparse
from urllib.request import url2pathname

import numpy as np
from PIL import Image
from tqdm import tqdm

from pyopf.io import load
from pyopf.project import ProjectObjects
from pyopf.resolve import resolve


def compile_tracks(project: ProjectObjects) -> tuple[dict, list]:
    """Compile tracks to have data easily usable to generate colmap files."""
    project_track = project.calibration.tracks.nodes[0]

    # create list of camera uid corresponding to each match
    hex_matched_uid = np.asarray([uid.hex for uid in project_track.matches.camera_uids])
    matches_cam_uid = hex_matched_uid[project_track.matches.camera_ids.flatten()]
    # create lists of projected tracks (2d coordinates) per image
    argsort_matches_cam_uid = np.argsort(matches_cam_uid)
    camd_uids, uid_to_idx = np.unique(
        matches_cam_uid[argsort_matches_cam_uid], return_index=True
    )
    coords_2d = np.split(
        project_track.matches.image_points.pixelCoordinates[argsort_matches_cam_uid],
        uid_to_idx[1:],
    )
    # create lists of 3d point IDs corresponding to the projected tracks
    idx_range = np.asarray(project_track.matches.point_index_ranges)
    argsort_matches = np.argsort(idx_range[:, 0])
    point3D_id = np.split(
        np.repeat(argsort_matches, idx_range[:, 1][argsort_matches])[
            argsort_matches_cam_uid
        ],
        uid_to_idx[1:],
    )
    point3D_id = np.array(point3D_id, dtype=object)
    # create dictionary: camera_uid -> (list of projected tracks, list of point3D_id)
    tracks_2Dcoords = dict(zip(camd_uids, zip(coords_2d, point3D_id)))

    # create lists of camera uids per track, and corresponding index
    track_cam_uids = np.empty(len(project_track), dtype=object)
    track_cam_uids[...] = [[] for _ in range(track_cam_uids.shape[0])]
    track_indices = np.empty(len(project_track), dtype=object)
    track_indices[...] = [[] for _ in range(track_indices.shape[0])]
    # fill in from the already generated point3D_id
    for cam_uid, points3D in tqdm(zip(camd_uids, point3D_id), total=len(camd_uids)):
        for track_index, track in enumerate(points3D):
            track_cam_uids[track].append(cam_uid)
            track_indices[track].append(track_index)
    # create track list where each row index is a point3d_id: (list of camera uids, list of indices)
    tracks_attributes = list(zip(track_cam_uids, track_indices))

    return tracks_2Dcoords, tracks_attributes


def orient_pos_computation(
    orient_deg: np.ndarray, position: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform orientation and position of OPF cameras to colmap format."""
    orient_rad = np.deg2rad(orient_deg)

    cosO = np.cos(orient_rad[0])
    sinO = np.sin(orient_rad[0])
    cosP = np.cos(orient_rad[1])
    sinP = np.sin(orient_rad[1])
    cosK = np.cos(orient_rad[2])
    sinK = np.sin(orient_rad[2])

    rot = np.zeros((3, 3), dtype=np.float64)
    rot[0][0] = cosP * cosK
    rot[0][1] = cosO * sinK + sinO * sinP * cosK
    rot[0][2] = sinO * sinK - cosO * sinP * cosK
    rot[1][0] = cosP * sinK
    rot[1][1] = -cosO * cosK + sinO * sinP * sinK
    rot[1][2] = -sinO * cosK - cosO * sinP * sinK
    rot[2][0] = -sinP
    rot[2][1] = sinO * cosP
    rot[2][2] = -cosO * cosP

    quat = np.zeros(4)
    quat[0] = 0.5 * np.sqrt(1 + rot[0][0] + rot[1][1] + rot[2][2])
    quat[1] = (rot[2][1] - rot[1][2]) / (4 * quat[0])
    quat[2] = (rot[0][2] - rot[2][0]) / (4 * quat[0])
    quat[3] = (rot[1][0] - rot[0][1]) / (4 * quat[0])

    pos = np.matmul(-rot, position)
    return quat, pos


def generate_cameras(
    project: ProjectObjects, sensors_uid_to_id: dict, user_config: dict
) -> None:
    """Generate the colmap file 'cameras.txt', storing sensors information."""
    sensors = project.calibration.calibrated_cameras.sensors

    file_header = "Camera list with one line of data per camera:\n"
    file_header += "  CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
    file_header += "  FULL_OPENCV model: CAMERA_ID FULL_OPENCV WIDTH HEIGHT fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6\n"
    file_header += "Number of cameras: " + str(len(sensors))

    cameras_data = np.zeros(
        len(sensors),
        dtype=[
            ("id", np.int32),
            ("model", "<U11"),
            ("width_px", np.int32),
            ("height_px", np.int32),
            ("fx", float),
            ("fy", float),
            ("ppa_x", float),
            ("ppa_y", float),
            ("k1", float),
            ("k2", float),
            ("p1", float),
            ("p2", float),
            ("k3", float),
            ("k4", float),
            ("k5", float),
            ("k6", float),
        ],
    )
    cameras_data["model"] = "FULL_OPENCV"

    for idx, sensor in enumerate(sensors):

        input_sensor = next(
            input_sensor
            for input_sensor in project.input_cameras.sensors
            if input_sensor.id == sensor.id
        )

        cameras_data["id"][idx] = sensors_uid_to_id[sensor.id.hex]
        cameras_data[["width_px", "height_px"]][idx] = tuple(input_sensor.image_size_px)
        cameras_data[["fx", "fy"]][idx] = sensor.internals.focal_length_px
        cameras_data[["ppa_x", "ppa_y"]][idx] = tuple(
            sensor.internals.principal_point_px
        )
        cameras_data[["k1", "k2", "k3"]][idx] = tuple(
            sensor.internals.radial_distortion
        )
        cameras_data[["p1", "p2"]][idx] = tuple(sensor.internals.tangential_distortion)

    output_file = os.path.join(user_config["out_dir"], "cameras.txt")
    np.savetxt(output_file, cameras_data, fmt="%s", header=file_header)


def generate_images(
    project: ProjectObjects,
    sensors_uid_to_id: dict,
    cameras_uid_to_id: dict,
    tracks_2Dcoords: dict,
    cam_paths_dict: dict,
    user_config: dict,
) -> None:
    """Generate the colmap file 'images.txt', storing cameras information."""
    cameras = project.calibration.calibrated_cameras.cameras

    tracks_count_per_camera = [
        len(point3D_ids) for _, point3D_ids in tracks_2Dcoords.values()
    ]

    file_header = "Image list with two lines of data per image:\n"
    file_header += "  IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
    file_header += "  POINTS2D[] as (X, Y, POINT3D_ID)\n"
    file_header += (
        "Number of images: "
        + str(len(cameras))
        + " , mean observations per image: "
        + str(sum(tracks_count_per_camera) / len(tracks_count_per_camera))
    )

    images_data = np.full(2 * len(cameras), "", dtype=object)

    for row, camera in enumerate(cameras):
        image_filename = cam_paths_dict[camera.id]["output_relative"]
        orient, pos = orient_pos_computation(camera.orientation_deg, camera.position)

        track_list = [0] * (3 * len(tracks_2Dcoords[camera.id.hex][1]))
        track_list[::3] = tracks_2Dcoords[camera.id.hex][0][:, 0]
        track_list[1::3] = tracks_2Dcoords[camera.id.hex][0][:, 1]
        track_list[2::3] = tracks_2Dcoords[camera.id.hex][1][:]

        images_data[2 * row] = f"{cameras_uid_to_id[camera.id.hex]}"
        images_data[
            2 * row
        ] += f" {' '.join(map(str, orient))} {' '.join(map(str, pos))}"
        images_data[
            2 * row
        ] += f" {sensors_uid_to_id[camera.sensor_id.hex]} {image_filename}"
        images_data[2 * row + 1] = " ".join(map(str, track_list))

    output_file = os.path.join(user_config["out_dir"], "images.txt")
    np.savetxt(output_file, images_data, fmt="%s", header=file_header)


def generate_points3D(
    project: ProjectObjects,
    camera_uid_to_id: dict,
    tracks_attributes: list,
    user_config: dict,
) -> None:
    """Generate the colmap file 'points3D.txt', storing tracks information."""
    tracks = project.calibration.tracks.nodes[0]

    cameras_per_track = list(map(lambda track: len(track[0]), tracks_attributes))

    tracks_color = np.zeros((len(tracks), 3), dtype=np.uint8)
    if project.calibration.tracks.nodes[0].color is not None:
        tracks_color = project.calibration.tracks.nodes[0].color[:, 0:3]

    file_header = "3D point list with one line of data per point:\n"
    file_header += (
        "  POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
    )
    file_header += (
        "Number of points: "
        + str(len(tracks))
        + ", mean track length: "
        + str(sum(cameras_per_track) / len(cameras_per_track))
    )

    track_image_data = np.full(len(tracks), "", dtype=object)

    # correction because gltf assumes Y is up, colmap assumes Z is up
    rotation_matrix = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    # apply rotation correction and gltf node transform
    positions_aug = np.c_[tracks.position.astype(np.float64), np.ones(len(tracks))]
    positions_aug = (
        rotation_matrix @ (tracks.matrix.astype(np.float64) @ positions_aug.T)
    ).T
    tracks_position = positions_aug[:, 0:3]

    for point3D_id in range(len(tracks)):
        image_point2d_list = [0] * (2 * len(tracks_attributes[point3D_id][0]))
        image_point2d_list[::2] = map(
            camera_uid_to_id.get, tracks_attributes[point3D_id][0]
        )
        image_point2d_list[1::2] = tracks_attributes[point3D_id][1]
        track_image_data[point3D_id] = " ".join(map(str, image_point2d_list))

    points3D_data = np.column_stack(
        (
            np.arange(len(tracks)),
            tracks_position,
            tracks_color,
            np.zeros(len(tracks)),
            track_image_data,
        )
    )

    output_file = os.path.join(user_config["out_dir"], "points3D.txt")
    np.savetxt(output_file, points3D_data, fmt="%s", header=file_header)


def generate_colmap(
    project: ProjectObjects,
    user_config: dict,
    cam_paths_dict: dict,
) -> None:
    """Generate the colmap files 'cameras.txt', 'images.txt', and 'points3D.txt' from the OPF project."""
    list_sensor_hex = [
        sensor.id.hex for sensor in project.calibration.calibrated_cameras.sensors
    ]
    sensors_uid_to_id = dict(zip(list_sensor_hex, np.arange(len(list_sensor_hex)) + 1))

    list_cam_hex = [
        camera.id.hex for camera in project.calibration.calibrated_cameras.cameras
    ]
    cameras_uid_to_id = dict(zip(list_cam_hex, np.arange(len(list_cam_hex)) + 1))

    tracks_2Dcoords, tracks_attributes = compile_tracks(project)

    os.makedirs(user_config["out_dir"], exist_ok=True)
    generate_cameras(project, sensors_uid_to_id, user_config)
    generate_images(
        project,
        sensors_uid_to_id,
        cameras_uid_to_id,
        tracks_2Dcoords,
        cam_paths_dict,
        user_config,
    )
    generate_points3D(project, cameras_uid_to_id, tracks_attributes, user_config)


def cam_input_output_paths(
    project: ProjectObjects, opf_project_folder: str, user_config: dict
) -> dict:
    """Compute/store the input and output paths (in absolute) of images. If output
    paths are different from input, copy the tree structure of input images.
    """
    cam_in_out_paths_dict = {}

    for camera in project.calibration.calibrated_cameras.cameras:
        camera_uri = [
            temp_camera.uri
            for temp_camera in project.camera_list.cameras
            if temp_camera.id == camera.id
        ][0]
        cam_input_path = url2pathname(urlparse(camera_uri).path)
        if not os.path.isabs(cam_input_path):
            # if path relative to opf_project_folder, make absolute
            cam_input_path = os.path.abspath(
                os.path.join(opf_project_folder, cam_input_path)
            )

        cam_in_out_paths_dict[camera.id] = {"input": cam_input_path}

    common_input_prefix = os.path.commonpath(
        [paths["input"] for paths in cam_in_out_paths_dict.values()]
    )

    for camera in project.calibration.calibrated_cameras.cameras:
        cam_output_path = cam_in_out_paths_dict[camera.id]["input"]

        cam_in_out_paths_dict[camera.id]["output_relative"] = os.path.relpath(
            cam_output_path, common_input_prefix
        )

        if user_config["out_img_dir"] is not None:
            # remove from paths common path, and add copy dir
            cam_output_path = os.path.join(
                os.path.abspath(user_config["out_img_dir"]),
                os.path.relpath(cam_output_path, common_input_prefix),
            )
        cam_in_out_paths_dict[camera.id]["output"] = cam_output_path

    return cam_in_out_paths_dict


def file_is_local(url: ParseResult) -> bool:
    """Check if file is stored locally."""
    return (url.hostname is None or url.hostname == "localhost") and (
        url.scheme == "file" or url.scheme == ""
    )


def check_project_supported(project: ProjectObjects) -> None:
    """Check that the project is supported by this converter tool."""
    # check project is calibrated
    if (
        (project.input_cameras is None)
        or (project.camera_list is None)
        or (project.calibration is None)
        or (project.calibration.calibrated_cameras is None)
    ):
        raise ValueError("Only calibrated projects can be converted")

    # check all sensors are perspective
    for sensor in project.calibration.calibrated_cameras.sensors:
        if sensor.internals.type != "perspective":
            raise ValueError("Only perspective sensors are supported.")

    # check if all cameras are stored locally
    list_camera_url = [urlparse(camera.uri) for camera in project.camera_list.cameras]
    if not all(file_is_local(url) for url in list_camera_url):
        raise ValueError("Remote files are not supported.")


def copy_images(project: ProjectObjects, cam_paths_dict: dict) -> None:
    """Copy images to output image directory."""
    pbar = tqdm(project.calibration.calibrated_cameras.cameras)
    for camera in pbar:
        pbar.set_postfix_str(
            os.path.basename(cam_paths_dict[camera.id]["input"])
            + "->"
            + cam_paths_dict[camera.id]["output"]
        )

        os.makedirs(os.path.dirname(cam_paths_dict[camera.id]["output"]), exist_ok=True)

        shutil.copyfile(
            cam_paths_dict[camera.id]["input"], cam_paths_dict[camera.id]["output"]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an OPF project to a sparse colmap model, consisting of three files: cameras, images and points3D.\
        These will contain information about the intrinsic and extrinsic parameters of the cameras, as well as the tracks. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "opf_project",
        type=str,
        help="An OPF project file",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output folder of colmap model.",
    )

    parser.add_argument(
        "--out-img-dir",
        type=str,
        default=None,
        help="If specified, the images will be copied to this directory.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    user_config = {
        "out_dir": args.out_dir,
        "out_img_dir": args.out_img_dir,
    }

    opf_project = args.opf_project
    opf_project_folder = os.path.dirname(opf_project)
    project = load(opf_project)
    project = resolve(project)
    cam_paths_dict = cam_input_output_paths(project, opf_project_folder, user_config)

    check_project_supported(project)

    print("Generating colmap files:")
    generate_colmap(project, user_config, cam_paths_dict)

    if args.out_img_dir is not None:
        print("Copying images:")
        copy_images(project, cam_paths_dict)
