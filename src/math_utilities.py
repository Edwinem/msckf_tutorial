import numpy as np


def normalize(vector):
    return vector / np.linalg.norm(vector)


def skew_matrix(vec3):
    """Convert a 3x1 vector into a symmetric skew matrix.

    Args:
        vec3: numpy array representing a 3x1 vector

    Returns:
        A 3x3 numpy array

    The skew symmetric matrix has several different meanings depending on the context. For instance it can be used
    as a substitute for the cross product
        a x b = skew_matrix(a)*b

    It is also useful in converting a vector into a rotation matrix. See the Rodrigues formula.
    """
    if vec3.size != 3:
        raise TypeError("Numpy array must have size == 3")
    return np.array([[0, -vec3[2], vec3[1]], [vec3[2], 0, -vec3[0]], [-vec3[1], vec3[0], 0]], dtype=np.float64)


def symmeterize_matrix(matrix):
    """Makes sure that the matrix is symmetric along the diagonal.

    Args:
        matrix: NxN matrix to make symmetric. Must be square.

    Returns:
        NxN matrix that is symmetric along the diagonal.


    A Kalman Filter expects that the covariance matrix is symmetric. However, due to numerical problems this may not
    always be the case as it gets modifed as the algorithm is run. Thus you can call this function after a modification
    (e.g. an Update or propogation) and make sure that it is symmetric.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise TypeError("Matrix must be square(equal rows and columns)")
    return (matrix + matrix.T) / 2
