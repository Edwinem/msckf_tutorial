import numpy as np
from transforms3d.euler import euler2mat

from spatial_transformations import JPLPose, JPLQuaternion
from triangulation import linear_triangulate, optimize_point_location


def invert_pose(pose):
    new_rot = pose[0:3, 0:3].T
    trans = pose[0:3, 3]
    new_pose = np.eye(4, dtype=np.float64)
    new_pose[0:3, 0:3] = new_rot
    new_pose[0:3, 3] = -new_rot @ trans
    return new_pose


def compute_relative_rot12(world_SE3_apple, world_SE3_banana):
    rot_apple = world_SE3_apple[0:3, 0:3]
    rot_banana = world_SE3_banana[0:3, 0:3]

    return rot_apple.T @ rot_banana


def compute_relative_T12(world_SE3_apple, world_SE3_banana):
    return invert_pose(world_SE3_apple) @ world_SE3_banana


def compute_relative_trans12(world_SE3_apple, world_SE3_banana):
    t_apple = world_SE3_apple[0:3, 3]
    t_banana = world_SE3_banana[0:3, 3]
    world_R_apple = world_SE3_apple[0:3, 0:3]
    apple_R_world = world_R_apple.T
    return apple_R_world @ (-t_apple + t_banana)


def project_point_jpl_pose(pose_jpl, point_in_world):

    camera_R_world = pose_jpl.q.rotation_matrix()
    world_t_camera = pose_jpl.t

    pt_in_camera = camera_R_world @ (point_in_world - world_t_camera)

    return pt_in_camera / pt_in_camera[2]


def project_point(world_SE3_camera, pt_in_world):
    camera_SE3_world = invert_pose(world_SE3_camera)
    pt_in_camera = camera_SE3_world[0:3, 0:3] @ pt_in_world + camera_SE3_world[0:3, 3]
    pt_in_camera /= pt_in_camera[2]
    return pt_in_camera


def compute_3x4_mat(rotation, translation):
    t = np.empty((3, 4))
    t[0:3, 0:3] = rotation
    t[0:3, 3] = translation
    return t
