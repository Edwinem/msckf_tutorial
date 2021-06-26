import numpy as np
from transforms3d.euler import euler2mat

from spatial_transformations import JPLPose, JPLQuaternion
from tests.utilities import (compute_relative_rot12, compute_relative_T12, compute_relative_trans12,
                             project_point_jpl_pose)
from triangulation import (linear_triangulate, linear_triangulate_2_view, optimize_point_location)


def test_linear_triangulate():

    point3d = np.array([0.15, 0.2, 3.1])

    rotation_matrices = [euler2mat(0.1, .2, .4), euler2mat(0.3, -.1, .25), euler2mat(0.05, 0.5, 0.3)]

    translations = [np.array([0.1, 0.2, -.4]), np.array([-0.25, .3, 0.35]), np.array([0.6, .1, -.1])]

    jpl_poses = []
    normalized_points = []

    for rot_mat, trans in zip(rotation_matrices, translations):
        pose = JPLPose.from_rotation_matrix_and_trans(rot_mat.T, trans)
        normalized_measurement = project_point_jpl_pose(pose, point3d)
        jpl_poses.append(pose)
        normalized_points.append(normalized_measurement)

    _, triangulated_point = linear_triangulate(jpl_poses, normalized_points)

    assert (np.allclose(triangulated_point, point3d))

    triangulated_point = optimize_point_location(point3d, jpl_poses, normalized_points)

    assert (np.allclose(triangulated_point, point3d))

    transforma = np.eye(4)
    transforma[0:3, 0:3] = rotation_matrices[0]
    transforma[0:3, 3] = translations[0]

    transformb = np.eye(4)
    transformb[0:3, 0:3] = rotation_matrices[2]
    transformb[0:3, 3] = translations[2]

    rel_T = compute_relative_T12(transforma, transformb)
    rel_trans = compute_relative_trans12(transforma, transformb)
    rel_rot = compute_relative_rot12(transforma, transformb)

    triangulated_point = linear_triangulate_2_view(jpl_poses[0], jpl_poses[2], normalized_points[0],
                                                   normalized_points[2])
    assert (np.allclose(triangulated_point, point3d))
