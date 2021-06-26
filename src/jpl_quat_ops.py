"""Contains quaternion functions based on the JPL Style.

The reference for equations in this file can be found in the Tech Report
Trawny, Nikolas, and Stergios I. Roumeliotis. "Indirect Kalman filter for 3D attitude estimation."
http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf

Note this is different than the Quaternions you might find in other packages. They use something called the
Hamiltonian Quaternion. They both have a number of important differences:
* JPL_Quat = Inverse(Hamiltonian_Quat)
* JPL is left handed, Hamiltonian is right handed.
* JPL by convention is stored as (x,y,z,w). Hamiltonian is typically stored as (w,x,y,z) though this is not a
 hard requirement.(Eigen stores it as (x,y,z,w)
* JPL works in the direction of global to local(world to body), Hamiltonian works in the direction local to global
 (body to world). This part sounds confusing so lets take an example related to Visual Odometry(VO). We have
 generally two frames we are interested in Visual Odometry world(global/W) and camera(local/body/C). The camera is
 some frame within world composed of a Rotation(R) and a translation(t). For our purposes we are going to ignore
 the translation and solely focus on the Rotation which can be represented by a Quaternion(Q).

 Given a point viewed from the camera frame(p_C) we want to rotate it into the world frame(p_W). This is done with
 the rotation matrix R_W,C .

 p_W = R_W,C * p_C

 We can also rotate it back by inverting the rotation matrix: R_W,C^-1 = R_C,W
 p_C = R_C,W * p_W

 The first rotation matrix(R_W,C) works by transforming the point from the body frame(camera) to the world
 frame(global).
 The second rotation matrix(R_C,W) works by transforming the point from world frame(global) to camera frame(body).

 Therefore JPLQuat = R_C,W and Hamiltonian_Quat = R_W,C

For a better overview of the differences I recommend "Quaternion kinematics for the error-state Kalman filter"
by Joan Solà.
"""

from math import sqrt

import numpy as np

from math_utilities import normalize, skew_matrix

epilson = 1e-12


def rot_mat_to_jpl_quat(rot_mat):
    """ Converts a 3x3 rotation matrix to a JPL style quaternion.
    Args:
        rot_mat: A numpy array representing a 3x3 rotation matrix

    Returns:
        A JPL style Quaternion

    The equations for this function come from (http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf) Eq. 74.
    It is a numerically stable version by dividing always by the biggest diagonal element.
    """
    if rot_mat.shape != (3, 3):
        raise TypeError("Rotation matrix must have size 3x3")
    w, x, y, z = 0.0, 0.0, 0.0, 0.0
    trace = rot_mat.trace()
    if rot_mat[0, 0] >= trace and rot_mat[0, 0] >= rot_mat[1, 1] and rot_mat[0, 0] >= rot_mat[2, 2]:
        x = sqrt((1.0 + (2 * rot_mat[0, 0]) - trace) / 4.0)
        y = (1.0 / (4.0 * x)) * (rot_mat[0, 1] + rot_mat[1, 0])
        z = (1.0 / (4.0 * x)) * (rot_mat[0, 2] + rot_mat[2, 0])
        w = (1.0 / (4.0 * x)) * (rot_mat[1, 2] - rot_mat[2, 1])

    elif rot_mat[1, 1] >= trace and rot_mat[1, 1] >= rot_mat[0, 0] and rot_mat[1, 1] >= rot_mat[2, 2]:
        y = sqrt((1.0 + (2 * rot_mat[1, 1]) - trace) / 4.0)
        x = (1.0 / (4.0 * y)) * (rot_mat[0, 1] + rot_mat[1, 0])
        z = (1.0 / (4.0 * y)) * (rot_mat[1, 2] + rot_mat[2, 1])
        w = (1.0 / (4.0 * y)) * (rot_mat[2, 0] - rot_mat[0, 2])
    elif rot_mat[2, 2] >= trace and rot_mat[2, 2] >= rot_mat[0, 0] and rot_mat[2, 2] >= rot_mat[1, 1]:
        z = sqrt((1.0 + (2 * rot_mat[2, 2]) - trace) / 4.0)
        x = (1.0 / (4.0 * z)) * (rot_mat[0, 2] + rot_mat[2, 0])
        y = (1.0 / (4.0 * z)) * (rot_mat[1, 2] + rot_mat[2, 1])
        w = (1.0 / (4.0 * z)) * (rot_mat[0, 1] - rot_mat[1, 0])
    else:
        w = sqrt((1.0 + trace) / 4.0)
        x = (1.0 / (4.0 * w)) * (rot_mat[1, 2] - rot_mat[2, 1])
        y = (1.0 / (4.0 * w)) * (rot_mat[2, 0] - rot_mat[0, 2])
        z = (1.0 / (4.0 * w)) * (rot_mat[0, 1] - rot_mat[1, 0])

    if w < 0:
        x = -x
        y = -y
        z = -z
        w = -w

    return np.array([x, y, z, w])


def jpl_quat_to_rot_mat(q):
    """Convert jpl quaternion to rotation matrix.

    Args:
        q: 4x1 numpy array of jpl quaternion [x,y,z,w]

    Returns:
        3x3 rotation matrix

    http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf Eq 77
    """
    q_skew = skew_matrix(q[0:3])
    w = q[3]
    # q_real = self.q[0:3,np.newaxis]
    # return (2 * w**2 - 1) * np.eye(3, dtype=np.float64) - 2 * w * q_skew + 2 * (q_real@q_real.T)
    return np.eye(3, dtype=np.float64) - (2 * w * q_skew) + (2 * q_skew @ q_skew)


def jpl_error_quat(dx):
    """Generate a quaternion from the error state representation.

    Args:
        dx: numpy array representing the error state(size 3)

    Returns:
        A 4x1 jpl quaternion representing an update to an existing quaternion.

    The error state is a minimal representation of the rotation error(3 parameters) vs the 4 needed for a quaternion.
    It also requires special handling compared to other error state representations as it requires quaternion
    multiplication rather then just simple addition/subtraction.

    This formulation comes from (http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf) Eq. 155.

    It can then be used to update a quaternion via a left multiplication. See the example below.

    Example:
        dq = error_quaternion(dx)
        updated_quaternion = dq * old_quaternion
        # note that here * is quaternion multiplication
    """
    if dx.size != 3:
        raise TypeError("Numpy array must have size == 3")
    delta_q = 0.5 * dx
    return normalize(np.array([delta_q[0], delta_q[1], delta_q[2], 1.0]))


def jpl_omega(vec):
    """Compute the Omega matrix of a 3x1 vector.

    Args:
        vec: 3x1 numpy vector

    Returns:
        4x4 Omega matrix.

    http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf Eq 47

    """
    mat = np.empty((4, 4), dtype=np.float64)
    mat[0:3, 0:3] = -skew_matrix(vec)
    mat[3, 0:3] = -vec
    mat[0:3, 3] = vec
    mat[3, 3] = 0.0
    return mat


def jpl_quat_left_matrix(q):
    """ Creates the left multiplication matrix of the jpl quaternion.

    Args:
        q: 4, numpy array of jpl quaternion [x,y,z,w]

    Returns:
        4x4 numpy matrix.

    This implements the left multiplication matrix of the jpl quaternion. This can be used to implement quaternion
    multiplication. As from the name you need to multiply this from the left side in the equation.

    http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf Eq 61. Though it is expanded to its individual components.

    """
    # Ql = np.empty((4, 4), dtype=np.float64)
    # Ql[0:3, 0:3] = q[3] * np.eye(3) * -skew_matrix(q[0:3])
    # Ql[0:3, 3] = q[0:3]
    # Ql[3, 0:3] = q[0:3].transpose()
    # Ql[3, 3] = q[3]

    Ql = np.empty((4, 4), dtype=np.float64)
    Ql[0, 0] = q[3]
    Ql[0, 1] = q[2]
    Ql[0, 2] = -q[1]
    Ql[0, 3] = q[0]
    Ql[1, 0] = -q[2]
    Ql[1, 1] = q[3]
    Ql[1, 2] = q[0]
    Ql[1, 3] = q[1]
    Ql[2, 0] = q[1]
    Ql[2, 1] = -q[0]
    Ql[2, 2] = q[3]
    Ql[2, 3] = q[2]
    Ql[3, 0] = -q[0]
    Ql[3, 1] = -q[1]
    Ql[3, 2] = -q[2]
    Ql[3, 3] = q[3]

    return Ql


def invert_jpl_quat(q):
    """ Invert a jpl quaternion.

    Args:
        q: 4, numpy array of jpl quaternion [x,y,z,w]

    Returns:
        Inverted jpl quaternion

    """
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def multiply_jpl_quaternions(q1, q2):
    """ Multiply 2 jpl quaternions.

    Args:
        q1: 4, numpy array of jpl quaternion [x,y,z,w]
        q2: 4, numpy array of jpl quaternion [x,y,z,w]

    Returns:
        JPL quaternion representing the quaternion product of 2 jpl quaternions.

    """
    q_left_mat = jpl_quat_left_matrix(q1)
    new_q = q_left_mat @ q2
    if new_q[3] < 0.0:
        new_q = -new_q
    return new_q


class JPLQuaternion():
    """
    Class which represents a JPL Style Quaternion.

    Wraps the various quaternion functions into 1 class. It also will always normalize any modified quaternion, ensuring
    that the unit quaternion accurately represents a rotation.

    """
    def __init__(self, x, y, z, w):
        self.q = np.array([x, y, z, w], dtype=np.float64)
        self.normalize()

    def normalize(self):
        self.q /= np.linalg.norm(self.q)

    @classmethod
    def identity(cls):
        return cls(0, 0, 0, 1.0)

    @classmethod
    def from_array(cls, array):
        assert (array.size == 4)
        return cls(array[0], array[1], array[2], array[3])

    def rotation_matrix(self):
        return jpl_quat_to_rot_mat(self.q)

    def multiply(self, other_quat):
        new_quat = multiply_jpl_quaternions(self.q, other_quat.q)
        return JPLQuaternion.from_array(new_quat)

    def rotate_vector(self, vector):
        assert vector.size == 3

        vector_quat = np.zeros((4, ))
        vector_quat[1:4] = vector
        inverse_quat = self.inverse()
        tmp = QL(self.q) @ vector_quat
        return QL(tmp) @ inverse_quat.q

    def inverse(self):
        inverted_quat = invert_jpl_quat(self.q)
        return JPLQuaternion.from_array(inverted_quat)

    def conjugate(self):
        return self.inverse()

    @classmethod
    def from_hamiltonian_quat_arr(cls, array):
        assert (array.size == 4)
        return cls(-array[1], -array[2], -array[3], array[0])

    @classmethod
    def from_rot_mat(cls, rot_mat):
        q = rot_mat_to_jpl_quat(rot_mat)
        return JPLQuaternion.from_array(q)

    def __eq__(self, other_quat):
        return np.allclose(self.q, other_quat.q)

    def __str__(self):
        q = self.q
        return "x: {}, y: {}, z: {}, w: {}".format(q[0], q[1], q[2], q[3])
