import numpy as np
import pytest
from transforms3d import quaternions
from transforms3d.euler import euler2mat

from msckf import *
from spatial_transformations import (JPLQuaternion, hamiltonian_quaternion_to_rot_matrix)


def test_quaternion_identity():
    identity_quat = JPLQuaternion.identity()


def test_jpl_quaternion_multiply():
    identity_quat = JPLQuaternion.identity()
    new_quat = identity_quat.multiply(identity_quat)


def test_jpl_quaternion_rot_matrix():

    rot_mat = euler2mat(2.7, 3.0, .5)

    # test conversion from and to rotation matrix
    q = JPLQuaternion.from_rot_mat(rot_mat)
    rot_mat2 = q.rotation_matrix()

    assert (np.allclose(rot_mat, rot_mat2))


def test_jpl_quaternion_multiply():

    rot_mat1 = euler2mat(2.7, 3.0, .5)
    rot_mat2 = euler2mat(1.0, 2.3, .45)

    q1 = JPLQuaternion.from_rot_mat(rot_mat1)
    q2 = JPLQuaternion.from_rot_mat(rot_mat2)

    expected_rot = rot_mat1 @ rot_mat2

    q_result = q1.multiply(q2)

    result_rot = q_result.rotation_matrix()
    assert (np.allclose(expected_rot, result_rot))


def test_jpl_quaternion_inverse():

    # Identity shouldn't change
    identity_quat = JPLQuaternion.identity()
    inverted_quat = identity_quat.inverse()
    assert (identity_quat == inverted_quat)

    rot_mat = euler2mat(2.7, 3.0, .5)
    q = JPLQuaternion.from_rot_mat(rot_mat)

    inverted_rot = rot_mat.T

    inv_q = q.inverse()

    assert (np.allclose(inverted_rot, inv_q.rotation_matrix()))
