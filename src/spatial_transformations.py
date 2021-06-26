from math import sqrt

import numpy as np

from jpl_quat_ops import JPLQuaternion

epilson = 1e-12


def hamiltonian_quaternion_to_rot_matrix(q, eps=np.finfo(np.float64).eps):
    w, x, y, z = q
    squared_norm = np.dot(q, q)
    if squared_norm < eps:
        return np.eye(3)
    s = 2.0 / squared_norm
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    return np.array([[1.0 - (yy + zz), xy - wz, xz + wy], [xy + wz, 1.0 - (xx + zz), yz - wx],
                     [xz - wy, yz + wx, 1.0 - (xx + yy)]])


def vee(mat_3x3):
    """The inverse of the skew_matrix function.

    Args:
        mat_3x3: A numpy array representing a 3x3 matrix.

    Returns:
        A numpy array representing a size 3 vector.

    This reverses the operation of the skew_matrix function.
        a = vee(skew_matrix(a))
    """
    if dx.size != 3:
        raise TypeError("Matrix must have 3x3 shape")
    return np.array([mat_3x3[2, 1], mat_3x3[0, 2], mat_3x3[1, 0]])


def exp_so3(vec3):
    skew = skew_matrix(vec3)
    theta = np.linalg.norm(vec3)
    A, B = 0
    if theta < epilson:
        A = 1.0
        B = 0.5
    else:
        A = np.sin(theta) / theta
        B = 1 - np.cos(theta) / (theta**2)
    if theta == 0:
        return np.eye(3, dtype=np.float64)
    else:
        return np.eye(3, dtype=np.float64) @ (A * skew) + (B * skew @ skew)


def log_so3(rot_mat):

    a = 0.5 * (rot_mat.trace() - 1.0)
    if a > 1.0:
        theta = np.arccos(1.0)
    else:
        if a < -1:
            theta = np.arccos(-1.0)
        else:
            theta = np.arccos(a)

    if theta < epilson:
        D = 0.5
    else:
        D = theta / (2 * np.sin(theta))

    mat = D * (rot_mat - np.linalg.transpose(rot_mat))
    identity = np.eye(3, np.float64)
    if np.all(np.isclose(mat, identity)):
        return np.zeros((3, 1))
    else:
        return vee(mat)


class JPLPose():
    """Represents a rigid body transform using the JPL Quaternion for rotation.

    A rigid body transform is composed of a rotation + translation. In this class we store the rotation as a
     JPL Quaternion. This makes the equations and usage a little bit different then what you might see in your standard
     textbook or codebase. This difference can be seen in the equation that transforms a point from Frame A to Frame B.

     Here Rot stands for the rotation matrix extracted from the JPLQuaternion.

     The conventional equations typically look like so:

     pt_B = B_Rot_A * pt_A + B_trans_A

     This corresponds to the 4x4 homogenous matrix multiplying a homogenous point.

     T = R t
         0 1

     pt_B = B_T_A * pt_A


     JPLPose equations:

     pt_B = B_Rot_A * (pt_A - A_trans_B)

     Homogenous matrix form:

     T = R -Rt
         0   1


     If you look at the frame order in the conventional equations you will notice that translation and rotation match.
     They both represent the rotation/translation of Frame A in Frame B.

     In the JPLPose equation the translation and rotation frame order do not match. Thus we need to utilize a slightly
     different equation for transforming a point.

    """
    def __init__(self, quat, trans):
        self.q = quat
        self.t = trans

    def quaternion(self):
        return self.q

    def translation(self):
        return self.t

    @classmethod
    def identity(cls):
        return cls(JPLQuaternion.identity, np.array([0.0, 0.0, 0.0]))

    @classmethod
    def from_array(cls, arr):
        if not isinstance(np.ndarray) and arr.size != 7:
            raise TypeError("arr is not a numpy object of the proper size(7)")
        return cls(JPLQuaternion(*arr[:4]), *arr[4:])

    @classmethod
    def from_rotation_matrix_and_trans(cls, rotation_matrix, translation):
        q = JPLQuaternion.from_rot_mat(rotation_matrix)
        return JPLPose(q, translation)

    def transform_vector(self, vec):
        """Transform a point by the rigid body transform.

        Args:
            vec: 3x1 or 4x1 numpy array. If 4x1 then the point needs to be in homogenous form([x,y,z,1.0])

        Returns:
            Numpy array representing the transformed point.

        This function corresponds to the following equation

        new_pt = R*(old_pt-trans)

        which is slightly different than the conventional rigid body equations. This is due to the use of the
         JPLQuaternion for the rotation.
        """
        if vec.size != 3 or vec.size != 4:
            raise TypeError("Point size must be 3 or 4 when transforming it")
        rot_mat = self.q.rotation_matrix()
        return rot_mat @ (vec[0:3] - self.t)

    def multiply_transform(self, other):
        new_q = self.q.multiply(other.q)
        new_t = self.q.rotation_matrix() @ (other.t - self.t)
        return JPLPose(new_q, new_t)

    def __matmul__(self, other):
        pass

    def __repr__(self):
        return "JPLPose: " +\
            "\n| quat:" + str(self.q) + \
            "\n| trans:" + str(self.t) + \
            "\n| Rot Matrix:" + str(self.q.rotation_matrix())
