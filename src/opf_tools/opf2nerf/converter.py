import argparse
import json
import os
from urllib.parse import ParseResult, urlparse
from urllib.request import url2pathname

import numpy as np
from PIL import Image
from tqdm import tqdm

from pyopf.cameras import CalibratedCamera, CalibratedSensor
from pyopf.io import load
from pyopf.project import ProjectObjects
from pyopf.resolve import resolve


def load_image(
    opf_project_folder: str, img_path: str, grayscale: bool = False
) -> np.ndarray:
    """Load the original image."""
    if not os.path.isabs(img_path):
        # if relative to opf project folder, make absolute
        img_path = os.path.join(opf_project_folder, img_path)
    if grayscale:
        return np.asarray(Image.open(img_path).convert("L"))
    else:
        return np.asarray(Image.open(img_path))


def apply_laplacian_kernel(array: np.ndarray) -> np.ndarray:
    """Apply laplacian kernel on an image (with valid convolution) as described by openCV. Reference:
    https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6
    """
    new_array = (
        -4 * array
        + np.roll(array, 1, axis=0)
        + np.roll(array, -1, axis=0)
        + np.roll(array, 1, axis=1)
        + np.roll(array, -1, axis=1)
    )
    return new_array[1 : new_array.shape[0] - 1, 1 : new_array.shape[1] - 1]


def get_sharpness(img: np.ndarray) -> float:
    """Compute the sharpness of an image."""
    return apply_laplacian_kernel(img).var()


def sensor_dict(
    project: ProjectObjects, sensor: CalibratedSensor, user_config: dict
) -> dict:
    """Describe sensor intrinsics in NeRF format."""
    transforms = {}
    input_sensor = [
        input_sensor
        for input_sensor in project.input_cameras.sensors
        if input_sensor.id == sensor.id
    ][0]
    transforms["camera_angle_x"] = 2 * np.arctan(
        input_sensor.image_size_px[0] / (2 * sensor.internals.focal_length_px)
    )
    transforms["camera_angle_y"] = 2 * np.arctan(
        input_sensor.image_size_px[1] / (2 * sensor.internals.focal_length_px)
    )
    transforms["fl_x"] = sensor.internals.focal_length_px
    transforms["fl_y"] = sensor.internals.focal_length_px
    transforms["k1"] = sensor.internals.radial_distortion[0]
    transforms["k2"] = sensor.internals.radial_distortion[1]
    transforms["k3"] = sensor.internals.radial_distortion[2]
    transforms["k4"] = 0
    transforms["p1"] = sensor.internals.tangential_distortion[0]
    transforms["p2"] = sensor.internals.tangential_distortion[1]
    transforms["is_fisheye"] = False
    transforms["cx"] = sensor.internals.principal_point_px[0]
    transforms["cy"] = sensor.internals.principal_point_px[1]
    transforms["w"] = int(input_sensor.image_size_px[0])
    transforms["h"] = int(input_sensor.image_size_px[1])
    transforms["aabb_scale"] = user_config["aabb_scale"]
    return transforms


def rotation_matrix_from_opk(cam_orient_deg: np.ndarray) -> np.ndarray:
    """Compute rotation matrix described by OPF omega-phi-kappa angles."""
    omega = np.radians(cam_orient_deg[0])
    phi = np.radians(cam_orient_deg[1])
    kappa = np.radians(cam_orient_deg[2])

    cosO = np.cos(omega)
    sinO = np.sin(omega)
    cosP = np.cos(phi)
    sinP = np.sin(phi)
    cosK = np.cos(kappa)
    sinK = np.sin(kappa)

    rot = np.zeros((3, 3), dtype=np.float64)
    rot[0][0] = cosP * cosK
    rot[1][0] = cosO * sinK + sinO * sinP * cosK
    rot[2][0] = sinO * sinK - cosO * sinP * cosK
    rot[0][1] = cosP * sinK
    rot[1][1] = -cosO * cosK + sinO * sinP * sinK
    rot[2][1] = -sinO * cosK - cosO * sinP * sinK
    rot[0][2] = -sinP
    rot[1][2] = sinO * cosP
    rot[2][2] = -cosO * cosP

    return rot


def get_file_path(camera: CalibratedCamera, paths_dict: dict, user_config: dict) -> str:
    """Get image path depending on user configuration of outputs."""

    if user_config["abs_img_path"]:
        file_path = paths_dict[camera.id]["output"]
    else:
        # get path relative to where the transforms file will be generated
        file_path = os.path.relpath(
            paths_dict[camera.id]["output"], user_config["out_dir"]
        )

    if not user_config["output_extension"]:
        # remove the extension of the file
        file_path = os.path.splitext(file_path)[0]

    return file_path


def get_transform_matrix(camera: CalibratedCamera, cam_flip: bool) -> np.ndarray:
    """Describe camera extrinsics as a 4x4 transform matrix."""
    cam_transform_matrix = np.zeros((4, 4))
    cam_transform_matrix[0:3, 0:3] = rotation_matrix_from_opk(camera.orientation_deg)
    cam_transform_matrix[0:3, 3] = camera.position
    cam_transform_matrix[3, 3] = 1

    if cam_flip:
        # some NeRFs require to 'flip' the camera
        flip_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        cam_transform_matrix = np.matmul(cam_transform_matrix, flip_mat)

    return cam_transform_matrix


def get_main_sensor(project: ProjectObjects) -> CalibratedSensor:
    """Determine the most used sensor model in the project."""
    cam_sensors = [
        camera.sensor_id.hex
        for camera in project.calibration.calibrated_cameras.cameras
    ]
    sensor_hex, sensor_counts = np.unique(np.asarray(cam_sensors), return_counts=True)
    max_sensor_arg = np.argmax(sensor_counts)
    main_sensor = [
        sensor
        for sensor in project.calibration.calibrated_cameras.sensors
        if sensor.id.hex == sensor_hex[max_sensor_arg]
    ][0]
    return main_sensor


def split_train_test(project: ProjectObjects, user_config: dict) -> np.ndarray:
    """Assign each camera randomly to train or test files, based on the --train-frac parameter."""
    nb_calib_cam = len(project.calibration.calibrated_cameras.cameras)

    attributed_train = np.zeros(nb_calib_cam, dtype=bool)
    attributed_train[0 : int(nb_calib_cam * user_config["train_fraction"])] = True
    np.random.shuffle(attributed_train)
    return attributed_train


def rescale_position(
    transforms_train: dict,
    transforms_test: dict,
    avg_pos: np.ndarray,
    max_pos: np.ndarray,
    min_pos: np.ndarray,
    cam_aabb_size: float,
) -> None:
    """Rescale and center so all cameras fit in a cube bounding box of edge length cam_aabb_size centered in [0, 0, 0]."""
    scale = np.max(max_pos - min_pos)
    scale /= cam_aabb_size

    for transforms in [transforms_train, transforms_test]:
        for frame in transforms["frames"]:
            transform_matrix = np.asarray(frame["transform_matrix"])
            transform_matrix[0:3, 3] = (transform_matrix[0:3, 3] - avg_pos) / scale
            frame["transform_matrix"] = transform_matrix.tolist()


def generate_nerf_transforms(
    project: ProjectObjects,
    opf_project_folder: str,
    user_config: dict,
    paths_dict: dict,
) -> tuple[dict, dict]:
    """Convert the OPF format to the transforms files used by NeRF."""
    main_sensor = get_main_sensor(project)

    transforms_train = sensor_dict(project, main_sensor, user_config)
    transforms_train["frames"] = []
    transforms_test = sensor_dict(project, main_sensor, user_config)
    transforms_test["frames"] = []

    attributed_train = split_train_test(project, user_config)

    avg_pos = np.zeros(3)
    max_pos = -np.ones(3) * np.inf
    min_pos = np.ones(3) * np.inf
    for idx, camera in enumerate(
        pbar := tqdm(project.calibration.calibrated_cameras.cameras)
    ):
        image_path = paths_dict[camera.id]["input"]

        # get camera properties
        img = load_image(opf_project_folder, image_path, grayscale=True)
        image_sharpness = get_sharpness(img)
        file_path = get_file_path(camera, paths_dict, user_config)
        cam_transform_matrix = get_transform_matrix(camera, user_config["camera_flip"])
        position = cam_transform_matrix[0:3, 3]
        avg_pos += position
        max_pos = np.maximum(max_pos, position)
        min_pos = np.minimum(min_pos, position)

        frame_properties = {
            "file_path": file_path,
            "sharpness": image_sharpness,
            "transform_matrix": cam_transform_matrix.tolist(),
        }

        if camera.sensor_id != main_sensor.id:
            # if the current sensor used is not the main one, add the intrinsics of the current sensor
            frame_sensor = [
                sensor
                for sensor in project.calibration.calibrated_cameras.sensors
                if sensor.id == camera.sensor_id
            ][0]
            frame_properties.update(sensor_dict(project, frame_sensor, user_config))

        # write in transform files, depending on train/test assignement
        if attributed_train[idx]:
            transforms_train["frames"].append(frame_properties)
            assigned_msg = "train"
        else:
            transforms_test["frames"].append(frame_properties)
            assigned_msg = "test "

        # describe progress, where images are assigned
        pbar.set_postfix_str(os.path.basename(image_path) + "->" + assigned_msg)

    avg_pos /= len(project.calibration.calibrated_cameras.cameras)
    rescale_position(
        transforms_train,
        transforms_test,
        avg_pos,
        max_pos,
        min_pos,
        user_config["camera_aabb_size"],
    )

    return transforms_train, transforms_test


def copy_convert_images(
    project: ProjectObjects, opf_project_folder: str, cam_in_out_paths_dict: dict
) -> None:
    """Copy and convert images."""
    pbar = tqdm(project.calibration.calibrated_cameras.cameras)
    for camera in pbar:
        pbar.set_postfix_str(
            os.path.basename(cam_in_out_paths_dict[camera.id]["input"])
            + "->"
            + cam_in_out_paths_dict[camera.id]["output"]
        )

        os.makedirs(
            os.path.dirname(cam_in_out_paths_dict[camera.id]["output"]), exist_ok=True
        )

        Image.fromarray(
            load_image(
                opf_project_folder,
                cam_in_out_paths_dict[camera.id]["input"],
                grayscale=False,
            )
        ).save(cam_in_out_paths_dict[camera.id]["output"], quality=95)


def cam_input_output_paths(
    project: ProjectObjects, opf_project_folder: str, user_config: dict
) -> dict:
    """Compute/store the input and output paths (in absolute) of images. If output
    paths is different from output, copies the tree structure of input images.
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
        [cam["input"] for cam in cam_in_out_paths_dict.values()]
    )

    for camera in project.calibration.calibrated_cameras.cameras:
        cam_output_path = cam_in_out_paths_dict[camera.id]["input"]
        if user_config["out_img_format"] is not None:
            cam_output_path = (
                os.path.splitext(cam_output_path)[0]
                + "."
                + user_config["out_img_format"]
            )

        if user_config["out_img_dir"] is not None:
            # remove from paths common path, and add copy dir
            cam_output_path = os.path.join(
                os.path.abspath(user_config["out_img_dir"]),
                os.path.relpath(cam_output_path, common_input_prefix),
            )
        cam_in_out_paths_dict[camera.id]["output"] = cam_output_path

    return cam_in_out_paths_dict


def save_transforms(
    transforms_train: dict, transforms_test: dict, user_config: dict
) -> None:
    """Save the transforms files as created by the converter."""
    os.makedirs(user_config["out_dir"], exist_ok=True)
    json.dump(
        transforms_train,
        open(os.path.join(user_config["out_dir"], "transforms_train.json"), "w"),
        indent=2,
    )
    if user_config["train_fraction"] != 1.0:
        json.dump(
            transforms_test,
            open(os.path.join(user_config["out_dir"], "transforms_test.json"), "w"),
            indent=2,
        )


def file_is_local(url: ParseResult) -> bool:
    """Check if file is stored locally."""
    return (url.hostname is None or url.hostname == "localhost") and (
        url.scheme == "file" or url.scheme == ""
    )


def check_project_supported(project: ProjectObjects, cam_paths_dict: dict) -> None:
    """Check that the project is supported by this converter tool."""
    # check project is calibrated
    if (
        (project.input_cameras is None)
        or (project.camera_list is None)
        or (project.calibration is None)
        or (project.calibration.calibrated_cameras is None)
    ):
        raise Exception("Only calibrated projects can be converted")

    # check all sensors are perspective
    for sensor in project.calibration.calibrated_cameras.sensors:
        if sensor.internals.type != "perspective":
            raise Exception("Only perspective sensors are supported.")

    # check if all cameras are stored locally
    list_camera_url = [urlparse(camera.uri) for camera in project.camera_list.cameras]
    if not all(file_is_local(url) for url in list_camera_url):
        raise Exception("Remote files are not supported.")

    # raise warnings if the output is not supported by all nerfs.
    # check if all images are (or will be if converted) in PNG format
    if not all(
        [
            os.path.splitext(cam_path["output"])[1].lower() == ".png"
            for cam_path in cam_paths_dict.values()
        ]
    ):
        print(
            "Warning: A lot of NeRFs don't specify the image format and use PNG. Consider using the --out-img-format parameter if you encounter problems."
        )

    # check if only one sensor used
    if len(project.calibration.calibrated_cameras.sensors) > 1:
        print(
            "Warning: More than one sensor model found in the input. This might not be supported by all NeRFs."
        )


def check_valid_aabb_scale(
    namespace: argparse.Namespace, var_name: str, aabb_scale: int, arg: str
) -> None:
    """Check validity of user given aabb_scale."""
    if not (1 <= aabb_scale <= 128) or not np.log2(aabb_scale).is_integer():
        raise Exception(arg + " should be a power of 2 in the [1,128] range.")
    setattr(namespace, var_name, aabb_scale)


def check_valid_train_frac(
    namespace: argparse.Namespace, var_name: str, train_frac: float, arg: str
) -> None:
    """Check validity of user given train fraction."""
    if not (0.0 <= train_frac <= 1.0):
        raise Exception(arg + " should in the [0.0, 1.0] range.")
    setattr(namespace, var_name, train_frac)


def check_valid_out_img_format(
    namespace: argparse.Namespace, var_name: str, img_format: str, arg: str
) -> None:
    """Check if user given image format is supported by PIL."""
    supported_extensions = [ext[1:] for ext in Image.registered_extensions()]
    if img_format is not None and img_format not in supported_extensions:
        raise Exception(arg + " " + img_format + " is not supported.")
    setattr(namespace, var_name, img_format)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an OPF project to NeRF transforms file(s). It can be split in train and tests. \
        Includes an image converter, as some NeRFs require specific format.",
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
        help="Output folder of transforms file(s).",
    )

    parser.add_argument(
        "--train-frac",
        type=float,
        default=1.0,
        metavar="[0.0, 1.0]",
        action=type(
            "",
            (argparse.Action,),
            dict(
                __call__=lambda var, _, namespace, val, arg: check_valid_train_frac(
                    namespace, var.dest, val, arg
                )
            ),
        ),
        help="Fraction of images to be used for training, the remaining are used for tests. \
            If set to 1.0, transforms_test.json is not generated.",
    )

    parser.add_argument(
        "--aabb-scale",
        type=int,
        default=16,
        metavar="[1, 128]",
        action=type(
            "",
            (argparse.Action,),
            dict(
                __call__=lambda var, _, namespace, val, arg: check_valid_aabb_scale(
                    namespace, var.dest, val, arg
                )
            ),
        ),
        help="aabb-scale is a NeRF parameter that specifies the extent (bounding box) of the scene. A value of 2 \
            means the scene bounding box edge length equals twice the average camera distance to the origin. Must be a power of 2.",
    )

    parser.add_argument(
        "--camera-aabb-size",
        type=float,
        default=8.0,
        help="Scales the camera positions to fit in a cube bounding box of edge length camera-aabb-size.",
    )

    parser.add_argument(
        "--camera-flip",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, the direction the cameras are facing is flipped with respect to OPF. \
            This is because some NeRFs use a different convention from OPF, while other use the same.",
    )

    parser.add_argument(
        "--abs-img-path",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, the image paths in transforms file(s) will be absolute.",
    )

    parser.add_argument(
        "--out-img-format",
        type=str,
        default=None,
        action=type(
            "",
            (argparse.Action,),
            dict(
                __call__=lambda var, _, namespace, val, arg: check_valid_out_img_format(
                    namespace, var.dest, val, arg
                )
            ),
        ),
        help="If given, convert the input images to this format (png, jpeg, jpg, ...). \
            The output images are written to --out-img-dir if given, otherwise they are \
            written to the same directory as the input images",
    )

    parser.add_argument(
        "--output-extension",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, images paths in the transforms file(s) will include the image extension. \
            Note that a lot of NeRFs don't expect an extension and assume they use PNG. \
            For these, you can use add the '--out-img-format png' parameter.",
    )

    parser.add_argument(
        "--out-img-dir",
        type=str,
        default=None,
        help="If specified, the images will be copied to this \
            directory (in the new format if --out-img-format is specified).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    user_config = {
        "train_fraction": args.train_frac,
        "aabb_scale": args.aabb_scale,
        "camera_aabb_size": args.camera_aabb_size,
        "camera_flip": args.camera_flip,
        "out_dir": args.out_dir,
        "output_extension": args.output_extension,
        "abs_img_path": args.abs_img_path,
        "out_img_format": args.out_img_format,
        "out_img_dir": args.out_img_dir,
    }

    opf_project = args.opf_project
    opf_project_folder = os.path.dirname(opf_project)
    project = load(opf_project)
    project = resolve(project)
    cam_paths_dict = cam_input_output_paths(project, opf_project_folder, user_config)

    check_project_supported(project, cam_paths_dict)

    print("Generating transforms file(s):")
    transforms_train, transforms_test = generate_nerf_transforms(
        project, opf_project_folder, user_config, cam_paths_dict
    )

    save_transforms(transforms_train, transforms_test, user_config)

    if (args.out_img_dir is not None) or (args.out_img_format is not None):
        print("Copying/Converting images:")
        copy_convert_images(project, opf_project_folder, cam_paths_dict)
