import numpy as np
from transforms3d.euler import euler2mat

from feature_tracker import (compute_algebraic_error, compute_two_point_translation, run_two_pt_ransac)
from math_utilities import skew_matrix
from tests.utilities import (compute_3x4_mat, compute_relative_rot12, compute_relative_trans12, project_point)


def create_essential(rot, trans):
    return skew_matrix(trans) @ rot


def generate_2_frame_data(num_points=10):
    """Generate some random data for 2 frames."""
    rotation_matrices = [euler2mat(0.1, .2, .4), euler2mat(0.3, -.1, .25)]

    translations = [np.array([0.1, 0.2, -.4]), np.array([-0.25, .3, 0.35])]

    x_coords = np.random.uniform(-.3, .4, num_points)
    y_coords = np.random.uniform(-.5, .5, num_points)
    z_coords = np.random.uniform(0.5, 2.5, num_points)
    points = np.column_stack((x_coords, y_coords, z_coords))

    poses = []
    projected_points = []
    for rot_mat, trans in zip(rotation_matrices, translations):
        T = compute_3x4_mat(rot_mat, trans)
        normalized_points = np.empty((num_points, 3))
        for idx in range(points.shape[0]):
            meas = project_point(T, points[idx])
            normalized_points[idx] = meas
        poses.append(T)
        projected_points.append(normalized_points)
    return poses, projected_points


def test_essential_error():
    """Test if the function to compute the essential matrix error for multiple points works."""
    poses, projected_points = generate_2_frame_data()

    T_A = poses[0]
    T_B = poses[1]

    A_R_B = compute_relative_rot12(T_A, T_B)

    A_t_B = compute_relative_trans12(T_A, T_B)

    measurements_A = projected_points[0]
    measurements_B = projected_points[1]

    A_Essential_B = create_essential(A_R_B, A_t_B)

    error = compute_algebraic_error(measurements_A, measurements_B, A_Essential_B)

    assert (np.all(np.isclose(error, 0.0)))


def test_estimated_translation_and_essential():
    poses, projected_points = generate_2_frame_data(2)

    T_A = poses[0]
    T_B = poses[1]

    A_R_B = compute_relative_rot12(T_A, T_B)

    B_R_A = A_R_B.T

    measurements_A = projected_points[0]
    measurements_B = projected_points[1]

    estimated_t = compute_two_point_translation(measurements_A[0], measurements_B[0], measurements_A[1],
                                                measurements_B[1], B_R_A)
    # Translation of Frame A in Frame B B_t_A
    real_t = compute_relative_trans12(T_B, T_A)

    scale_values = real_t / estimated_t

    # Check that all the scalar values are the same
    assert (np.all(np.isclose(scale_values, scale_values[0])))

    assert (np.allclose(estimated_t * scale_values[0], real_t))

    B_Essential_A = create_essential(B_R_A, estimated_t)

    error = compute_algebraic_error(measurements_B, measurements_A, B_Essential_A)

    assert (np.all(np.isclose(error, 0.0)))


def test_ransac():
    poses, projected_points = generate_2_frame_data()

    T_A = poses[0]
    T_B = poses[1]

    A_R_B = compute_relative_rot12(T_A, T_B)

    A_t_B = compute_relative_trans12(T_A, T_B)

    measurements_A = projected_points[0]
    measurements_B = projected_points[1]

    inliers = run_two_pt_ransac(measurements_A, measurements_B, 200, 0.5, A_R_B.T)
    print(inliers)
