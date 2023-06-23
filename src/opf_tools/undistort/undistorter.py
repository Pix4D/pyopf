import argparse
import functools
import os
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
from PIL import Image

from pyopf.cameras import CalibratedSensor, PerspectiveInternals
from pyopf.io import load
from pyopf.project import ProjectObjects
from pyopf.resolve import resolve


def camera_supported(
    camera_uri: str, sensor: list[CalibratedSensor]
) -> tuple[bool, str]:
    """Check if camera is supported, and create warning message accordingly if not."""
    supported = True
    warning_message = "Warning! Image " + camera_uri

    url = urlparse(camera_uri)
    if (url.hostname is not None and url.hostname != "localhost") or (
        url.scheme != "file" and url.scheme != ""
    ):
        # check if camera uri is supported
        warning_message += " has an unsupported URI. Only relative URI references or absolute URIs referring to the localhost are supported. \
        Also only 'file' or '' url scheme are supported."
        supported = False
    elif len(sensor) == 0:
        # check if camera has calibrated sensor
        warning_message += " has no calibrated sensor."
        supported = False
    elif sensor[0].internals.type != "perspective":
        # check if camera is perspective
        warning_message += (
            " uses a unsupported camera type, only perspective cameras are supported."
        )
        supported = False
    warning_message += " It will be skipped."

    return supported, warning_message


def load_image(opf_project_folder: str, img_path: str) -> np.ndarray:
    """Load the original image."""
    if not os.path.isabs(img_path):
        # if relative path, make absolute
        img_path = os.path.join(opf_project_folder, img_path)
    img = np.asarray(Image.open(img_path))

    return img


def save_image(image: np.ndarray, save_path: str) -> None:
    """Save the undistorted image."""
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    Image.fromarray(image, "RGB").save(save_path, quality=95)


@functools.lru_cache
def compute_undistort_map(
    h: int, w: int, sensor_internals: PerspectiveInternals
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the undistortion mapping.

    Compute the uv-xy mapping. A given sensor will always follow the same mapping, so we use memoization to not recompute the same output multiple times.
    The output is used in billinear interpolation, so for each pixel of undistorted image we need 4 pixel from original. These are returned uv_mapping of shape (4, h, w).
    We also need the corresponding coefficients, these are returned in coeffs, also of shape (4, h, w).
    OpenCV implementation:  https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    """
    # get sensor internals
    focal_length = sensor_internals.focal_length_px
    principal_point = sensor_internals.principal_point_px
    ks = np.zeros(6)
    ks[0 : len(sensor_internals.radial_distortion)] = sensor_internals.radial_distortion
    ps = np.zeros(2)
    ps[
        0 : len(sensor_internals.tangential_distortion)
    ] = sensor_internals.tangential_distortion

    # normalized coordinates
    norm_x = np.tile(np.arange(w), (h, 1))
    norm_y = np.transpose(np.tile(np.arange(h), (w, 1)))
    xy = (
        np.array([norm_x - principal_point[0], norm_y - principal_point[1]])
        / focal_length
    )
    # radius squared
    r2 = xy[0] ** 2 + xy[1] ** 2
    # distortion model
    radial_distort = (1 + ks[0] * r2 + ks[1] * r2**2 + ks[2] * r2**3) / (
        1 + ks[3] * r2 + ks[4] * r2**2 + ks[5] * r2**3
    )
    source_xy = xy * radial_distort
    source_xy[0] += 2 * ps[0] * xy[0] * xy[1] + ps[1] * (r2 + 2 * xy[0] ** 2)
    source_xy[1] += 2 * ps[1] * xy[0] * xy[1] + ps[0] * (r2 + 2 * xy[1] ** 2)

    # in uv space
    uv = source_xy * focal_length + principal_point.reshape(2, 1, 1)
    # crop to avoid mapping to outside image
    uv = np.maximum(0.5, uv)
    uv[0] = np.minimum(w - 1.5, uv[0])
    uv[1] = np.minimum(h - 1.5, uv[1])
    # fixing to integer, for each uv coordinate there will be 4 close pixels, we keep their coordinates in uv_mapping
    uv_max = np.floor(uv + 1).astype(int)
    uv_max = np.maximum(0, uv_max)
    uv_max[0] = np.minimum(w - 1, uv_max[0])
    uv_max[1] = np.minimum(h - 1, uv_max[1])
    uv_min = np.floor(uv).astype(int)
    uv_min = np.maximum(0, uv_min)
    uv_min[0] = np.minimum(w - 1, uv_min[0])
    uv_min[1] = np.minimum(h - 1, uv_min[1])
    uv_mapping = np.concatenate((uv_max, uv_min))

    # coefficients for interpollation of each coordinate
    d_uvmax = uv_max - uv
    d_uvmin = uv - uv_min
    coeffs = np.concatenate((d_uvmax, d_uvmin))

    return coeffs, uv_mapping


def bilinear_interpolation(
    img: np.ndarray, coeffs: np.ndarray, uv_mapping: np.ndarray
) -> np.ndarray:
    """Compute the undistorted image using bilinear interpolation."""
    h, w = img.shape[:2]

    fxy1 = np.multiply(
        coeffs[0].reshape(h, w, 1), img[uv_mapping[3], uv_mapping[2]]
    ) + np.multiply(coeffs[2].reshape(h, w, 1), img[uv_mapping[3], uv_mapping[0]])
    fxy2 = np.multiply(
        coeffs[0].reshape(h, w, 1), img[uv_mapping[1], uv_mapping[2]]
    ) + np.multiply(coeffs[2].reshape(h, w, 1), img[uv_mapping[1], uv_mapping[0]])
    img = (
        np.multiply(coeffs[1].reshape(h, w, 1), fxy1)
        + np.multiply(coeffs[3].reshape(h, w, 1), fxy2)
    ).astype("uint8")

    return img


def undistort(project: ProjectObjects, opf_project_folder: str) -> None:
    """Undistort all images of the project for which a calibrated sensor exists."""
    if (
        (project.input_cameras is None)
        or (project.camera_list is None)
        or (project.calibration is None)
        or (project.calibration.calibrated_cameras is None)
    ):
        print("Project doesn't have calibrated cameras. Quitting.")
        return

    for capture in project.input_cameras.captures:
        for camera in capture.cameras:
            # get camera's image uri
            camera_uri = [
                temp_camera.uri
                for temp_camera in project.camera_list.cameras
                if temp_camera.id == camera.id
            ][0]

            sensor = [
                sensor
                for sensor in project.calibration.calibrated_cameras.sensors
                if sensor.id == camera.sensor_id
            ]

            supported, warning_message = camera_supported(camera_uri, sensor)
            if not supported:
                print(warning_message)
                continue
            else:
                sensor = sensor[0]
                camera_url = url2pathname(urlparse(camera_uri).path)

            # load camera image
            print("Input image: ", camera_uri)
            img = load_image(opf_project_folder, camera_url)

            # get sampling map (where to sample original image)
            h, w = img.shape[:2]
            coeffs, uv_mapping = compute_undistort_map(h, w, sensor.internals)

            # bilinear interpolation
            undist_img = bilinear_interpolation(img, coeffs, uv_mapping)

            # puts them in an 'undistort' directory in their original location
            save_path = os.path.join(
                os.path.dirname(camera_url), "undistort", os.path.basename(camera_url)
            )
            if not os.path.isabs(save_path):
                # if relative path, make absolute
                save_path = os.path.join(opf_project_folder, save_path)
            print(
                "Output image: ",
                save_path,
            )
            save_image(undist_img, save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Undistorts the images of an OPF project. Only perspective cameras with a calibrated sensor will be undistorted. \
        The undistorted images will be stored in their original place, but in an 'undistort' directory."
    )
    parser.add_argument(
        "input",
        type=str,
        help="An OPF project file",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    opf_project = args.input
    opf_project_folder = os.path.dirname(opf_project)

    project = load(opf_project)
    project = resolve(project)

    undistort(project, opf_project_folder)
