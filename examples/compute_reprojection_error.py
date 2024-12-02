#!/usr/bin/env python

"""
this script computes the reprojection error of input GCPs in calibrated cameras
"""

import argparse

import numpy as np

import pyopf.cameras
import pyopf.io
import pyopf.resolve

# == define some helper functions


def find_object_with_given_id(objects: list, id):
    """
    Returns the first object in the list that matches the given id or None if not found
    """

    return next((obj for obj in objects if obj.id == id), None)


def make_basis_change_matrix_from_opk_in_degrees(omega_phi_kappa: np.array):
    """
    Computes a basis change matrix from angles (in degrees) expressed in the omega-phi-kappa convention.
    This matrix transforms points from a right-top-back camera coordinate reference frame to the scene one

    Please see the definition of the omega-phi-kappa angles in the OPF specifications:
    https://pix4d.github.io/opf-spec/specification/input_cameras.html#omega-phi-kappa-orientation
    More details are provided in:
    https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
    """

    omega_rad, phi_rad, kappa_rad = np.deg2rad(omega_phi_kappa)

    r_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(omega_rad), -np.sin(omega_rad)],
            [0.0, np.sin(omega_rad), np.cos(omega_rad)],
        ],
        dtype=np.float64,
    )

    r_y = np.array(
        [
            [np.cos(phi_rad), 0.0, np.sin(phi_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(phi_rad), 0.0, np.cos(phi_rad)],
        ],
        dtype=np.float64,
    )

    r_z = np.array(
        [
            [np.cos(kappa_rad), -np.sin(kappa_rad), 0.0],
            [np.sin(kappa_rad), np.cos(kappa_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return r_x @ r_y @ r_z


def invert_transformation(transformation: np.ndarray):
    """
    Computes the inverse of a given 4x4 transformation
    """

    inverse_transformation = np.identity(4, dtype=np.float64)

    inverse_transformation[0:3, 0:3] = transformation[0:3, 0:3].transpose()
    inverse_transformation[0:3, 3] = (
        -transformation[0:3, 0:3].transpose() @ transformation[0:3, 3]
    )

    return inverse_transformation


def make_camera_intrinsic_matrix(
    internals: pyopf.cameras.sensor_internals.PerspectiveInternals,
):
    """
    makes a 3x3 camera intrinsic matrix form the OPF internals
    """

    intrinsic_matrix = np.zeros((3, 3), dtype=np.float64)

    intrinsic_matrix[0, 0] = internals.focal_length_px
    intrinsic_matrix[1, 1] = internals.focal_length_px

    intrinsic_matrix[0:2, 2] = internals.principal_point_px

    intrinsic_matrix[2, 2] = 1.0

    return intrinsic_matrix


def apply_distortion_model(
    ux: float, uy: float, internals: pyopf.cameras.sensor_internals.PerspectiveInternals
):
    """
    applies the distortion model to undistorted image coordinates
    """

    k1, k2, k3 = internals.radial_distortion
    t1, t2 = internals.tangential_distortion
    cpx, cpy = internals.principal_point_px
    f = internals.focal_length_px

    ux = (ux - cpx) / f
    uy = (uy - cpy) / f
    r = ux * ux + uy * uy
    dr = 1.0 + r * k1 + r**2 * k2 + r**3 * k3
    dtx = 2.0 * t1 * ux * uy + t2 * (r + 2.0 * ux * ux)
    dty = 2.0 * t2 * ux * uy + t1 * (r + 2.0 * uy * uy)

    return f * (dr * ux + dtx) + cpx, f * (dr * uy + dty) + cpy


def project_point(
    camera: pyopf.cameras.CalibratedCamera,
    internals: pyopf.cameras.sensor_internals.PerspectiveInternals,
    point: np.array,
):
    """
    computes the projection of a given 3d point in a camera given its internals
    """

    basis_change_from_camera_to_scene = make_basis_change_matrix_from_opk_in_degrees(
        camera.orientation_deg
    )

    camera_pose = np.identity(4)
    camera_pose[0:3, 0:3] = basis_change_from_camera_to_scene
    camera_pose[0:3, 3] = camera.position

    # as per the definition of the omega-phi-kappa angles, the camera frame axes are defined as
    # X: camera/image right (looking through the camera/image)
    # Y: camera/image top (looking through the camera/image)
    # Z: camera back (opposite to viewing direction through camera)
    # to go to the standard computer vision convention, the Y and Z axes need to be flipped

    flip_y_and_z = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    camera_pose_right_bottom_front = camera_pose @ flip_y_and_z

    camera_pose_inverse = invert_transformation(camera_pose_right_bottom_front)

    camera_intrinsic_matrix = make_camera_intrinsic_matrix(internals)

    point_homogeneous = np.append(point, 1.0)

    # project the point on the camera image

    point_in_camera_homogeneous = camera_pose_inverse @ point_homogeneous

    x, y, z = camera_intrinsic_matrix @ point_in_camera_homogeneous[:-1]

    ux = x / z
    uy = y / z

    # apply distortion model

    distorted_ux, distorted_uy = apply_distortion_model(ux, uy, internals)

    return np.array([distorted_ux, distorted_uy], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the reprojection error of GCPs in an OPF project."
    )

    parser.add_argument(
        "--opf_path", type=str, help="[REQUIRED] The path to your project.opf file."
    )

    parser.add_argument(
        "--point_type",
        type=str,
        choices=["mtps", "gcps"],
        help="[REQUIRED] Wheter to use MTPs or GCPs",
    )

    parser.add_argument(
        "--use_input_3d_coordinates",
        action="store_true",
        help="Use input 3d coordinates instead of calibrated ones. Only applicable if point_type is set to gcps",
    )

    args = parser.parse_args()

    if args.use_input_3d_coordinates and args.point_type == "mtps":
        raise ValueError("MTPs have no input 3d coordinates")

    return args


def main():
    args = parse_args()

    # == Load the OPF ==

    project = pyopf.resolve.resolve(pyopf.io.load(args.opf_path))

    if args.point_type == "mtps":
        input_points = project.input_control_points.mtps
    else:
        input_points = project.input_control_points.gcps

        if args.use_input_3d_coordinates:
            projected_input_points = project.projected_control_points.projected_gcps

    calibrated_control_points = project.calibration.calibrated_control_points.points

    calibrated_cameras = project.calibration.calibrated_cameras.cameras
    sensors = project.calibration.calibrated_cameras.sensors

    # == for all points, compute the reprojection error of all marks and the mean ==

    for point in input_points:

        if args.use_input_3d_coordinates:
            scene_point = find_object_with_given_id(projected_input_points, point.id)
        else:
            scene_point = find_object_with_given_id(calibrated_control_points, point.id)

            if scene_point is None:
                print(point.id, "not calibrated")
                continue

        scene_point_3d_coordinates = scene_point.coordinates

        all_reprojection_errors = []

        for mark in point.marks:

            # find the corresponding calibrated camera
            calibrated_camera = find_object_with_given_id(
                calibrated_cameras, mark.camera_id
            )

            # find the internal parameters for this camera
            calibrated_sensor = find_object_with_given_id(
                sensors, calibrated_camera.sensor_id
            )
            internal_parameters = calibrated_sensor.internals

            # project the 3d point on the image
            point_on_image = project_point(
                calibrated_camera, internal_parameters, scene_point_3d_coordinates
            )

            # compute reprojection error
            reprojection_error = point_on_image - mark.position_px

            all_reprojection_errors.append(reprojection_error)

        if len(all_reprojection_errors) > 0:
            # compute the mean of the norm of the reprojection errors
            all_reprojection_errors = np.array(all_reprojection_errors)
            mean_reprojection_error = np.mean(
                np.apply_along_axis(np.linalg.norm, 1, all_reprojection_errors)
            )

            print(point.id, mean_reprojection_error)
        else:
            print(point.id, "no marks")


if __name__ == "__main__":
    main()
