import logging
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from math_utilities import skew_matrix
from params import AlgorithmConfig

logger = logging.getLogger(__name__)

Y_IDX = 0
X_IDX = 1

# Value we use mark if a numpy array is not filled.
NOT_FILLED_VALUE = 0.0


def draw_tracks(img, prev_points, cur_points):
    # Assume same size

    assert (prev_points.size == cur_points.size)
    num_points = int(prev_points.shape[0])
    color = np.random.randint(0, 255, (num_points, 3))
    color_image_shape = img.shape + (3, )
    mask = np.zeros(color_image_shape, dtype=img.dtype)
    for i, (old, new) in enumerate(zip(prev_points.astype(np.int32), cur_points.astype(np.int32))):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        mask = cv2.circle(mask, (a, b), 5, color[i].tolist(), -1)
    gray_image_in_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    output_img = cv2.add(gray_image_in_color, mask)

    cv2.imshow('frame', output_img)
    k = cv2.waitKey(0) & 0xff


# def estimate_via_linalg(prev_point_1,cur_point_1,prev_point_2,cur_point_2,rot):
#     row1 = (R @ prev_point_1).transpose() @ skew_matrix(cur_point_1)
#     row2 = (R @ prev_point_2).transpose() @ skew_matrix(cur_point_2)
#
#     A = np.vstack(row1,row2)
#
#     #At=0
#
#     t=scipy.linalg.null_space(A)
#
#     return t


def compute_two_point_translation(pt_1_A, pt_1_B, pt_2_A, pt_2_B, FrameB_R_FrameA):
    """Computes the translation given 2 landmark measurements and the rotation between frames.

    Args:
        pt_1_A: Numpy array representing a Measurement(u,v,1.0)/Point(x,y,z) corresponding to landmark 1 in Frame A
        pt_1_B: Numpy array representing a Measurement(u,v,1.0)/Point(x,y,z) corresponding to landmark 1 in Frame B
        pt_2_A: Numpy array representing a Measurement(u,v,1.0)/Point(x,y,z) corresponding to landmark 2 in Frame A
        pt_2_B: Numpy array representing a Measurement(u,v,1.0)/Point(x,y,z) corresponding to landmark 2 in Frame B
        FrameB_R_FrameA: 3x3 rotation matrix. Rotates a point from Frame A -> Frame B

    Returns:
        Numpy array for the translation between frames. If the given points are in the normalized coordinate system
        (u,v,1.0) then the translation is to scale. real_t = alpha*est_t . If the given points are the 3D coordinates
        in their respective frames then the scale is correct. real_t = est_t

    This function changes results depending on if the input points are measurements in the normalized camera space
    (u,v,1.0) or the actual 3D coordinates of the points in the respective frame.

    The actual equations can be found in the following paper "2-Point-based Outlier Rejection for Camera-IMU Systems
     with applications to Micro Aerial Vehicles" by Troiani, et al
    """

    # Rotate the points in Frame A to a new Frame C which has the same rotation as Frame B
    pt_1_C = FrameB_R_FrameA @ pt_1_A
    pt_2_C = FrameB_R_FrameA @ pt_2_A

    c1 = pt_1_B[0] * pt_1_C[1] - pt_1_C[0] * pt_1_B[1]
    c2 = pt_1_C[1] * pt_1_B[2] - pt_1_B[1] * pt_1_C[2]
    c3 = pt_1_B[0] * pt_1_C[2] - pt_1_C[0] * pt_1_B[2]
    c4 = pt_2_B[0] * pt_2_C[1] - pt_2_C[0] * pt_2_B[1]
    c5 = pt_2_C[1] * pt_2_B[2] - pt_2_B[1] * pt_2_C[2]
    c6 = pt_2_B[0] * pt_2_C[2] - pt_2_C[0] * pt_2_B[2]

    alpha = math.atan2(c3 * c5 - c2 * c6, c1 * c6 - c3 * c4)
    beta = math.atan2(-c3, c1 * math.sin(alpha) + c2 * math.cos(alpha))
    t = (math.sin(beta) * math.cos(alpha), math.cos(beta), -math.sin(beta) * math.sin(alpha))
    return np.array(t)


def compute_algebraic_error(points_A, points_B, A_EssentialMat_B):
    """Compute the sampson error given points matches in 2 frames and an Essential Matrix.

    Args:
        points_A: Normalized points in Frame/Image A
        points_B: Normalized points in Frame/Image B
        A_EssentialMat_B: The essential matrix which transforms points in Frame B to lines in Frame A.

    Returns: Numpy array of the error for each point pair.

    Given an Essential matrix the following should hold true:

        pt_A^T * A_E_B * pt_B = 0

    In reality though there will be some error

        pt_A^T * A_E_B * pt_B = 0

    So we simply use the absolute value of the error found from using the above equation.

    Please see "compute_sampson_error" sampson error for more detailed description.

    """

    # This computes p_A^T*E*p_B for all the points quickly. Else this operation takes too long.
    # Equivalent to the following
    # num_points = points_A.shape[0]
    # error1 = np.empty((num_points))
    # for i in range(num_points):
    #     error1[i] = points_A[i].T @ A_EssentialMat_B @ points_B[i]
    temp = np.dot(A_EssentialMat_B, points_B.T).T
    error = np.einsum("ij,ij->i", points_A, temp)
    return np.absolute(error)


def compute_sampson_error(points_A, points_B, A_EssentialMat_B):
    """Compute the sampson error given points matches in 2 frames and an Essential Matrix.

    Args:
        points_A: Normalized points in Frame/Image A
        points_B: Normalized points in Frame/Image B
        A_EssentialMat_B: The essential matrix which transforms points in Frame B to lines in Frame A.

    Returns: Numpy array of the error for each point pair.

    The essential matrix (E) allows one to relate points in two different frames with the following Equation.

    x'^T * E * x = 0

    or in our case

    pt_A^T * A_E_B * pt_B = 0

    or in reality

    pt_A^T * A_E_B * pt_B = error

    Since our estimation isn't perfect or, because pt_A and pt_B are not actually a match. We can directly use this
    error as the error between our pt_A and pt_B match. This is called the "algebraic error" as we simply take it from
    algebra.

    There is some debate regarding whether using an algebraic error is valid due to it "not minimizing the right
    thing". Instead people prefer to use a "geometric error" which "should" represent the problem better. However, they
    can be more expensive to compute. The sampson error thus uses a first order approximation of the geometric error.
    Please see https://cseweb.ucsd.edu/classes/sp04/cse252b/notes/lec11/lec11.pdf for the derivation and equation.

    """

    #The commented out and actual code compute the same values. The commented out verion is too slow to be practical,
    # but serves as a reference for what we are actually computing.

    # num_points = points_A.shape[0]
    # error1 = np.zeros((num_points))
    # for i in range(num_points):
    #     top = points_A[i].T @ A_EssentialMat_B @ points_B[i]
    #     top=top**2
    #     fx1 = A_EssentialMat_B @ points_B[i]
    #     fx2 =  points_A[i].T @A_EssentialMat_B
    #     error1[i]= top / (fx1[0]**2+fx1[1]**2+fx2[0]**2+fx2[1]**2)

    error = compute_algebraic_error(points_A, points_B, A_EssentialMat_B)
    squared_error = np.square(error)
    E_ptB = np.dot(A_EssentialMat_B, points_B.T).T
    E_ptA = np.dot(points_A, A_EssentialMat_B)

    sum = np.square(E_ptA[:, 0]) + np.square(E_ptA[:, 1]) + np.square(E_ptB[:, 0]) + np.square(E_ptB[:, 1])

    final_error = squared_error / sum
    return final_error


def rotation_vector_to_matrix(rot_vec):
    assert (rot_vec.size == 3)
    theta = np.linalg.norm(rot_vec)

    r = rot_vec / theta

    s = np.sin(theta)
    c = np.cos(theta)
    cs = 1.0 - np.cos(theta)

    w_x = skew_matrix(r)

    I = np.eye(3, dtype=np.float64)

    x = r[0] * 2
    y = r[1] * 2
    z = r[2] * 2

    rrt = np.array([[x * x, x * y, x * z], [x * y, y * y, y * z], [x * z, y * z, z * z]])
    return c * I + cs * rrt - s * w_x


def compute_rotation_from_imu_measurements(imu_measurements):

    accumulated_R = np.eye(3, dtype=np.float64)
    for imu_data in imu_measurements:
        dt = imu_data.time_interval
        if math.isclose(dt, 0.0):
            continue
        w = imu_data.angular_vel
        wdt = w * dt
        delta_R = rotation_vector_to_matrix(wdt)

        accumulated_R = delta_R @ accumulated_R
    return accumulated_R


def compute_rotation_from_imu_measurements2(imu_measurements):
    tempR = np.eye(3)
    I = np.eye(3)
    for imu in imu_measurements:
        wm = imu.angular_vel
        dt = imu.time_interval

        bIsSmallAngle = False
        if (np.linalg.norm(wm) < 0.001745329):
            bIsSmallAngle = True

        w1 = np.linalg.norm(wm)
        wdt = w1 * dt
        wx = skew_matrix(wm)
        wx2 = wx @ wx

        deltaR = np.eye(3)
        if (bIsSmallAngle):
            deltaR = I - dt * wx + (.5 * dt**2) * wx2
        else:
            deltaR = I - (np.sin(wdt) / w1) * wx + ((1 - np.cos(wdt)) / w1**2) * wx2

        tempR = deltaR * tempR
    return tempR
    #
    # mag = np.linalg.norm(w)
    # mag2 = mag**2
    # wdt = w*dt
    #
    # skew_w = skew_matrix(w)
    # skew_w2 = skew_w@skew_w
    #
    # dt2 = dt**2
    #
    # if mag< self.params.small_angle_threshold:
    #     delta_R =  np.eye(3)-(dt*wx)+(.5*dt2)*wx2
    # else:
    #     deltaR = np.eye(3)-(np.sin(wdt) / w1)*wx + ((1 - cos(wdt)) / mag2) * wx2;


def run_two_pt_ransac(prev_points, cur_points, num_iterations, inlier_threshold, cur_R_prev, use_sampson_error=True):
    """Run RANSAC algorithm on 2 point translation estimation.

    Args:
        prev_points: Normalized coordinate points in the previous image.
        cur_points: Normalized coordiante in the current image.
        num_iterations: Number of iterations to run the RANSAC algorithm.
        inlier_threshold: Threshold on whether a point is considered an inlier. The error for a point <= this value.
        cur_R_prev: Rotation matrix from the current frame to the previous
        use_sampson_error: Flag on whether to use the sampson error metric.

    Returns: Numpy mask which marks whether a point is an inlier.

    This function runs the RANSAC algorithm(https://en.wikipedia.org/wiki/Random_sample_consensus) for outlier
    estimation. For our model we utilize the the two point algorithm from the paper "2-Point-based Outlier Rejection
    for Camera-IMU Systems with applications to Micro Aerial Vehicles" by Troiani, et al. Given a known rotation and 2
    pairs of feature matches we can estimate the translation to scale. This allows us to compute the Essential matrix
    which can then be used to check the number of inliers.

    """
    num_points = int(prev_points.shape[0])
    prev_points_copy = prev_points
    cur_points_copy = cur_points
    # The points are in the normalized camera coordinates (u,v,1.0) but they might not have the 1.0 column. Here we
    # check that the 1.0 column exists and if not then we create a new array with it.
    if prev_points.shape[1] != 3:
        prev_points_copy = np.column_stack((prev_points, np.ones(num_points)))
        cur_points_copy = np.column_stack((cur_points, np.ones(num_points)))

    best_inliers = None
    best_inlier_count = 0
    for iteraton in range(num_iterations):

        index_1 = np.random.randint(0, num_points)
        index_2 = np.random.randint(0, num_points)

        while (index_1 == index_2):
            index_2 = np.random.randint(0, num_points)

        prev_point_1 = prev_points_copy[index_1]
        cur_point_1 = cur_points_copy[index_1]
        prev_point_2 = prev_points_copy[index_2]
        cur_point_2 = cur_points_copy[index_2]

        # compute the translation

        cur_t_prev = compute_two_point_translation(cur_point_1, prev_point_1, cur_point_2, prev_point_2, cur_R_prev)

        # Build the essential matrix.
        cur_E_prev = skew_matrix(cur_t_prev) * cur_R_prev

        # For every point pair compute the error given their locations and the computed essential matrix.
        if use_sampson_error:
            error = compute_sampson_error(cur_points_copy, prev_points_copy, cur_E_prev)
        else:
            error = compute_algebraic_error(cur_points_copy, prev_points_copy, cur_E_prev)

        # Any point with an error less than the threshold is considered an inlier.
        inlier_mask = error <= inlier_threshold
        num_inliers = np.sum(inlier_mask)
        # Store the data if there are mre inliers than previous iterations.
        if num_inliers > best_inlier_count:
            best_inliers = inlier_mask
            best_inlier_count = num_inliers
    return best_inliers


def contains_duplicates(X):
    return len(np.unique(X)) != len(X)


class FeatureTracker():
    """Implements a basic 2d feature tracker based on sparse optical flow(Lucas and Kanade algorithm)

    This is a optical flow based feature tracker which tries to track keypoints over subsequent images. It is composed
    of 3 parts:

        1. Keypoint detection
            * This detects new keypoints in an image using opencvs 'goodFeaturesToTrack' function. In order to ensure
            a good distribution of features we also implement bucketing. We divide up the image into a grid, and require
            that each cell within the grid can only contain a maximum amount of keypoints.
        2. Feature tracking:
            * Given keypoints in a previous frame we try to track where they move to using sparse optical flow from
            opencv's 'calcOpticalFlowPyrLK'.
        3. Outlier removal
            * The feature tracking will introduce some mismatched pairs. We remove these outliers by running RANSAC. We
            use a version called the two-point algorithm, which utilizes a known rotation to estimate the translation.
            We thereforce use the angular velocity readings from an IMU to estimate the rotation.

    Each track(collection of matched keypoints) is assigned a unique ID, and is how the user of this class can associate
    keypoints over time. We also provide the keypoints in the normalized image space (u,v,1.0) so that it is independent
    of the camera calibration.
    """
    def __init__(self, params: AlgorithmConfig.FeatureTrackerParams, camera_calibration):
        # Flag on whether we we are running on the first image of the system.
        self.first_image = True

        # Set the feature tracker parameters
        self.params = params
        # Params for the optical flow tracker. Requires a special format so we create that here.
        self.lk_dict = params.lk_params.to_opencv_dict()

        self.prev_img = None
        self.prev_keypoints = None
        self.prev_ids = None
        self.id_counter = 1

        # The grid structure we use for bucketing.
        self.grid = None
        # Number of rows in the grid.
        self.n_grid_rows = 0
        # Number of columns in the grid.
        self.n_grid_cols = 0

        # Store the individual parts of the camera calibration.
        self.imu_R_camera = camera_calibration.imu_R_camera
        self.camera_R_imu = camera_calibration.imu_R_camera.T
        self.intrinsics = camera_calibration.intrinsics.generate_K()
        self.dist_coeffs = camera_calibration.intrinsics.dist_coeffs

    def get_current_normalized_keypoints_and_ids(self):
        """Get the normalized keypoints and their associated IDs.

        Returns:
            - normalized_keypoints: nx3 numpy array of the keypoints as normalized camera coordinates(u,v,1.0)
            - ids: nx1 numpy array containing the ID of each keypoint.

        """

        # Note that at the end of 'track()' function we actually set the previous values equal to the current values. So
        # this step is valid.
        normalized_keypoints = np.squeeze(cv2.undistortPoints(self.prev_keypoints, self.intrinsics, self.dist_coeffs))
        return normalized_keypoints, self.prev_ids

    def _initialize_grid(self, img, params):
        """Initialize the grid structure we use for bucketing.

        Args:
            img: Image to provide the sizes
            params: Parameter struct which contains details about the grid size.

        For our purposes the grid datastructure is a numpy array of size n_grid_rows x n_grid_cols x max_keypoints*2
        The depth is set so that each cell/bucket can only store a maximum amount of keypoints(*2 because each keypoint
        has 2 values).

        """
        rows, cols = img.shape
        self.n_grid_rows = math.floor(rows / params.grid_block_size) + 1
        self.n_grid_cols = math.floor(cols / params.grid_block_size) + 1

        self.grid = np.zeros((self.n_grid_rows, self.n_grid_cols, params.max_keypoints_per_block * 2), dtype=np.float32)

    def fill_grid(self, old_points):
        """Fill our grid datastructure with some points.

        Args:
            old_points: nx2 numpy array contains keypoints

        By filling our grid with our current points we can check which ones are full, and which ones can accept new
        features.
        """
        for pt in old_points:
            # compute in which cell each point will end up.
            row = math.floor(pt[0] / self.params.block_size)
            col = math.floor(pt[1] / self.params.block_size)

            cell = self.grid[row][col]
            # For the cell check if there is space to place the keypoint.
            for i in range(self.params.max_keypoints_per_block):
                if cell[i * 2] == NOT_FILLED_VALUE:
                    cell[i * 2 + 0] = pt[X_IDX]
                    cell[i * 2 + 1] = pt[Y_IDX]
                    break

    def grid_cluster(self, new_points, max_number=-1):
        """ Filter out the new points, by only accepting points that end up in an empty bucket.

        Args:
            new_points: nx2 array of keypoints
            max_number: Maximum number of new points to accept. So even if the grid has empty buckets, we will not
                add anymore points.


        """

        # Create our numpy mask which we use to mark whether a new point is valid.
        good_points_mask = np.zeros((new_points.shape[0]), dtype=bool)
        num_good_point = 0
        for idx, new_pt in enumerate(new_points):
            #Stop if we have hit the maximum number of keypoints we want.
            if num_good_point == max_number:
                break

            row = math.floor(new_pt[1] / self.params.grid_block_size)
            col = math.floor(new_pt[0] / self.params.grid_block_size)

            cell = self.grid[row][col]

            # Essentially here we are computing 2 things.
            # 1. Is there space for a new point in the cell?
            # 2. If so does the new point have sufficient distance from other points already in the cell. This helps
            # ensure that even with the cell the points are well distributed.
            new_point_is_bad = False
            i = 0
            while i < self.params.max_keypoints_per_block:
                # Check if the current index is empty, and if so end.
                if cell[2 * i] == NOT_FILLED_VALUE:
                    break
                # If it is not empty then it means we are storing a point at this index.
                pt_in_cell = cell[2 * i:2 * i + 2]

                # Check if our new point and the point at our current index have enough distance between them. If
                # the distance is too small then we consider the new point bad.
                squared_dist = (pt_in_cell[X_IDX] - new_pt[X_IDX])**2 + (pt_in_cell[Y_IDX] - new_pt[Y_IDX])**2
                if squared_dist < self.params.min_dist_between_keypoints:
                    new_point_is_bad = True
                    break
                i += 1
            # If the new point is not bad, and there is space in our cell then add the point.
            if not new_point_is_bad and i != self.params.max_keypoints_per_block:
                cell[2 * i + X_IDX] = new_pt[X_IDX]
                cell[2 * i + Y_IDX] = new_pt[Y_IDX]
                good_points_mask[idx] = True
                num_good_point += 1
        return good_points_mask

    def detect_keypoints(self, img, params):
        """Detect keypoints within the image.

        Args:
            img: Image we are detecting new keypoints in.
            params: Contains parameters regarding our keypoint detection method.

        Returns: nx2 numpy array of detected keypoints.

        """
        detected_corners = cv2.goodFeaturesToTrack(img, params.max_corners, params.quality_level, params.min_distance)

        if detected_corners.size != 0:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            # Try to detect with subpixel accuracy.
            detected_corners = cv2.cornerSubPix(img, detected_corners,
                                                (params.sub_pix_window_size, params.sub_pix_window_size),
                                                (params.sub_pix_zero_zone, params.sub_pix_zero_zone), criteria)

        return np.squeeze(detected_corners)

    def track(self, img, imu_measurements=[]):
        """Track and detect new keypoints in the new image.

        Args:
            img:
            imu_measurements: Buffer of IMU measurements. Used to calculate the rotation.

        Returns: True on success

        """

        # If this is the first time than initialize certain params and just detect new keypoints. We can't do any
        # tracking as we don't have any previous keypoints.
        if self.first_image:
            self._initialize_grid(img, self.params)
            p = self.params.detector_params
            detected_corners = self.detect_keypoints(img, p)
            num_keypoints = detected_corners.shape[0]
            id_start = self.id_counter
            id_end = self.id_counter + num_keypoints
            self.prev_ids = np.arange(id_start, id_end)
            self.id_counter += num_keypoints + 1
            self.first_image = False
            self.prev_img = img
            self.prev_keypoints = detected_corners
            return True

        if self.prev_keypoints is None or self.prev_keypoints.size == 0:
            logger.warn("Don't have any previous detected corners. Must reinitialize")
            return False

        tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_img, img, self.prev_keypoints, None,
                                                             **self.lk_dict)
        good_points_mask = np.squeeze(status == 1)
        logger.debug(f"Num points before {self.prev_keypoints.shape[0]}")
        good_prev = self.prev_keypoints[good_points_mask, :]
        good_cur = tracked_points[good_points_mask, :]
        good_ids = self.prev_ids[good_points_mask]
        logger.debug(f"Num points after {good_cur.shape[0]}")

        # If we have imu measurements then remove outliers by running two-point RANSAC.
        if imu_measurements:
            # Rotation of current IMU to previous
            imu2_R_imu1 = compute_rotation_from_imu_measurements(imu_measurements)
            imu2_R_imu1 = compute_rotation_from_imu_measurements2(imu_measurements)
            # Transform the rotation of the IMU into the camera frame
            camera2_R_camera1 = self.camera_R_imu @ imu2_R_imu1 @ self.imu_R_camera
            # Our RANSAC algorithm works with the keypoints in the normalized camera coordinate so convert them.
            good_prev_norm = np.squeeze(cv2.undistortPoints(good_prev, self.intrinsics, self.dist_coeffs))
            good_cur_norm = np.squeeze(cv2.undistortPoints(good_cur, self.intrinsics, self.dist_coeffs))
            ransac_inlier_mask = run_two_pt_ransac(good_prev_norm, good_cur_norm, self.params.ransac_iterations,
                                                   self.params.ransac_threshold, camera2_R_camera1)
            logger.debug("Ransac removed %i points", ransac_inlier_mask.size - np.sum(ransac_inlier_mask))
            good_prev = good_prev[ransac_inlier_mask]
            good_cur = good_cur[ransac_inlier_mask]
            good_ids = good_ids[ransac_inlier_mask]
            assert good_ids.size != 0

        if good_cur.shape[0] < self.params.min_tracked_features:
            self.grid = np.zeros_like(self.grid)
            self.fill_grid(good_cur)
            newly_detected_keypoints = self.detect_keypoints(img, self.params.detector_params)
            num_old_keypoints = good_cur.shape[0]
            max_num_new = self.params.max_tracked_features - num_old_keypoints
            good_keypoints_mask = self.grid_cluster(newly_detected_keypoints, max_num_new)
            good_keypoints = newly_detected_keypoints[good_keypoints_mask, :]
            num_new_keypoints = good_keypoints.shape[0]
            id_start = self.id_counter
            id_end = self.id_counter + num_new_keypoints
            new_keypoint_ids = np.arange(id_start, id_end)
            good_cur = np.concatenate([good_cur, good_keypoints])
            good_ids = np.concatenate([good_ids, new_keypoint_ids])
            self.id_counter = id_end
        self.prev_keypoints = good_cur
        self.prev_ids = good_ids
        self.prev_img = img
        # draw_tracks(img, good_prev, good_cur)
        return True

        #draw_tracks(img,good_prev,good_cur)
