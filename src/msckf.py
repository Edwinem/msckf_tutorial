import logging
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from math import sqrt

import numpy as np
import scipy
from scipy.stats import chi2

from jpl_quat_ops import JPLQuaternion, jpl_error_quat, jpl_omega
from math_utilities import normalize, skew_matrix, symmeterize_matrix
from msckf_types import FeatureTrack
from params import AlgorithmConfig
from spatial_transformations import JPLPose
from triangulation import linear_triangulate, optimize_point_location

logger = logging.getLogger(__name__)


class StateInfo():
    ''' Stores size and indexes related to the MSCKF state vector'''

    # Represents the indexing of the individual components  of the state vector.
    # The slice(a,b) class just represents a python slice [a:b]

    # Attitude, since we are storing the error state the size is 3 rather than the 4 required for the quaternion.
    ATT_SLICE = slice(0, 3)
    POS_SLICE = slice(3, 6)  # Position
    VEL_SLICE = slice(6, 9)  # Velocity
    BG_SLICE = slice(9, 12)  # Bias gyro
    BA_SLICE = slice(12, 15)  # Bias accelerometer

    IMU_STATE_SIZE = 15  # Size of the imu state(the above 5 slices)

    CLONE_START_INDEX = 15  # Index at which the camera clones start
    CLONE_STATE_SIZE = 6  # Size of the camera clone state(3 for attitude, 3 for position)

    CLONE_ATT_SLICE = slice(0, 3)  # Where within the camera clone the attitude error state is
    CLONE_POS_SLICE = slice(3, 6)  # Where within the camera clone the position error is


class CameraClone():
    def __init__(self, camera_JPLPose_global, camera_id):
        self.camera_JPLPose_global = camera_JPLPose_global
        self.timestamp = 0
        self.camera_id = camera_id


class State():
    """
    Contains the state vector and the associated covariance for our Kalman Filter.

    Within the state vector we keep track of our IMU State(contains position,attitude,biases,...) and a limited
    amount of Camera State Stochastic Clones. These clones contain the position and attitude of the camera at some
    timestamp in the past. They are what allow us to define EKF Update functions linking the past poses to our current
    pose.



    """
    def __init__(self):

        # Attitude of the camera. Stores the rotation of the IMU to the global frame as a JPL quaternion.
        self.imu_JPLQ_global = JPLQuaternion.identity()

        # Position of the IMU in the global frame.
        self.global_t_imu = np.zeros((3, ), dtype=np.float64)

        # Velocity of the IMU
        self.velocity = np.zeros((3, ), dtype=np.float64)

        self.bias_gyro = np.zeros((3, ), dtype=np.float64)

        self.bias_acc = np.zeros((3, ), dtype=np.float64)

        self.clones = OrderedDict()

        # The covariance matrix of our state.
        self.covariance = np.eye(StateInfo.IMU_STATE_SIZE, dtype=np.float64)

    def set_velocity(self, vel):
        assert (vel.size == 3)
        self.velocity = vel

    def set_gyro_bias(self, bias_gyro):
        assert (bias_gyro.size == 3)
        self.bias_gyro = bias_gyro

    def set_acc_bias(self, bias_acc):
        assert (bias_acc.size == 3)
        self.bias_acc = bias_acc

    def add_clone(self, camera_clone):
        self.clones[camera_clone.camera_id] = camera_clone

    def num_clones(self):
        return len(self.clones)

    def calc_clone_index(self, index_within_clones):
        return StateInfo.CLONE_START_INDEX + index_within_clones * StateInfo.CLONE_STATE_SIZE

    def update_state(self, delta_x):
        assert (delta_x.shape[0] == self.get_state_size())

        # For everything except for the rotations we can use a simple vector update
        # x' = x+delta_x

        self.global_t_imu += delta_x[StateInfo.POS_SLICE]
        self.velocity += delta_x[StateInfo.VEL_SLICE]
        self.bias_gyro += delta_x[StateInfo.BG_SLICE]
        self.bias_acc += delta_x[StateInfo.BA_SLICE]

        # Attitude requires a special Quaternion update
        # Note because we are using the left jacobians the update needs to be applied from the left side.
        error_quat = JPLQuaternion.from_array(jpl_error_quat(delta_x[StateInfo.ATT_SLICE]))
        self.imu_JPLQ_global = error_quat.multiply(self.imu_JPLQ_global)

        # Now do same thing for the rest of the clones

        for idx, clone in enumerate(self.clones.values()):

            delta_x_index = StateInfo.IMU_STATE_SIZE + idx * StateInfo.CLONE_STATE_SIZE

            # Position update
            pos_start_index = delta_x_index + StateInfo.CLONE_POS_SLICE.start
            pos_end_index = delta_x_index + StateInfo.CLONE_POS_SLICE.stop
            clone.camera_JPLPose_global.t += delta_x[pos_start_index:pos_end_index]

            # Attitude update
            att_start_index = delta_x_index + StateInfo.CLONE_ATT_SLICE.start
            att_end_index = delta_x_index + StateInfo.CLONE_ATT_SLICE.stop
            delta_x_slice = delta_x[att_start_index:att_end_index]
            error_quat = JPLQuaternion.from_array(jpl_error_quat(delta_x_slice))
            clone.camera_JPLPose_global.q = error_quat.multiply(clone.camera_JPLPose_global.q)

    def get_state_size(self):
        return StateInfo.IMU_STATE_SIZE + self.num_clones() * StateInfo.CLONE_STATE_SIZE

    def print_state(self):

        print("Position {}", self.global_t_imu)


class MSCKF():
    """ Implements the Multi-Constraint Kalman filter(MSCKF) for visual inertial odometry.

    This implements the MSCKF first seen in A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint Kalman Filter
     for Vision-Aided Inertial Navigation". It is a tightly coupled EKF based approach to solving the visual inertial
     odometry problem.
    """
    def __init__(self, params, camera_calibration):
        self.state = State()
        self.params = params
        self.imu_buffer = []
        self.map_id_to_feature_tracks = {}
        self.camera_id = 0
        self.camera_calib = camera_calibration
        self.gravity = np.array([0, 0, -9.81])
        self.noise_matrix = np.eye(4 * 3)
        self.chi_square_val = {}
        for i in range(1, 100):
            self.chi_square_val[i] = chi2.ppf(0.05, i)

    def set_imu_noise(self, sigma_gyro, sigma_acc, sigma_gyro_bias, sigma_acc_bias):
        """Set the noise/random walk parameters of the IMU sensor.

        Args:
            sigma_gyro: Standard deviation of the gyroscope measurement.
            sigma_acc: Standard deviation of the accelorometer measurement
            sigma_gyro_bias: Standard deviation of the gyroscope bias evolution
            sigma_acc_bias: Standard deviation of the accelerometer bias

        These parameters can typically be found in the datasheet associated with your IMU.
        https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model this link gives a good overview of what the parameters
        mean and how they can be found.
        """
        # Since it is the standard deviation we need to square the variable for the variance.
        values = np.array([[sigma_gyro**2, sigma_gyro**2, sigma_gyro**2], [sigma_acc**2, sigma_acc**2, sigma_acc**2],
                           [sigma_gyro_bias**2, sigma_gyro_bias**2, sigma_gyro_bias**2],
                           [sigma_acc_bias**2, sigma_acc_bias**2, sigma_acc_bias**2]])
        diag = np.eye(4 * 3, dtype=np.float64)
        np.fill_diagonal(diag, values)
        self.noise_matrix = diag

    def initialize(self, global_R_imu, global_t_imu, vel, bias_acc, bias_gyro):
        self.state.imu_JPLQ_global = JPLQuaternion.from_rot_mat(global_R_imu.T)
        self.state.global_t_imu = global_t_imu
        new_vel = global_R_imu.T @ vel
        self.state.velocity = new_vel
        self.state.bias_acc = bias_acc
        self.state.bias_gyro = bias_gyro

    def initialize2(self, quat, global_t_imu, vel, bias_acc, bias_gyro):
        self.state.imu_JPLQ_global = JPLQuaternion.from_array(quat)
        self.state.global_t_imu = global_t_imu
        self.state.velocity = vel
        self.state.bias_acc = bias_acc
        self.state.bias_gyro = bias_gyro

    def set_imu_covariance(self, att_var, pos_var, vel_var, bias_gyro_var, bias_acc_var):
        '''Used to set the covariance of the IMU section in the covariance matrix.

        Args:
            att_var:
            pos_var:
            vel_var:
            bias_gyro_var:
            bias_acc_var:

        This should typically be one of the first functions you call to initialize the system. It sets how confident we
        are in the initial values of the IMU(smaller is more confident). Note that even if you know a value perfectly,
        you should never set its covariance to 0 as it causes numerical problems. Instead set it to an extremely small
        value such as 1e-12.

        '''
        cov = self.state.covariance
        st = StateInfo
        arr = np.empty((st.IMU_STATE_SIZE))
        arr[st.ATT_SLICE] = att_var
        arr[st.POS_SLICE] = pos_var
        arr[st.VEL_SLICE] = vel_var
        arr[st.BG_SLICE] = bias_gyro_var
        arr[st.BA_SLICE] = bias_acc_var
        np.fill_diagonal(cov, arr)

    def propogate_and_update_msckf(self, imu_buffer, feature_ids, normalized_keypoints):
        pass

    def remove_old_clones(self):
        """Remove old camera clones from the state vector.

        In order to keep our computation bounded we remove certain camera clones from our state vector. In this
        implementation we implement a basic sliding window which always removes the oldest camera clone. This way our
        state vector is kept to a constant size.

        """

        num_clones = self.state.num_clones()

        # If we have yet to reach the maximum number of camera clones then skip
        if num_clones < self.params.max_clone_num:
            return

        # Remove the oldest
        ids_to_remove = []
        oldest_camera_id = list(self.state.clones.keys())[0]

        # Run the MSCKF update on any features which have this clone
        for id, track in self.map_id_to_feature_tracks.items():
            track_cam_id = track.camera_ids[0]
            if track_cam_id == oldest_camera_id:
                ids_to_remove.append(id)

        self.msckf_update(ids_to_remove)

        # Remove the clone from the state vector

        # Since it is the oldest it is the first clone within our covariance matrix.
        clone_start_index = StateInfo.CLONE_START_INDEX
        clone_end_index = clone_start_index + StateInfo.CLONE_STATE_SIZE
        s = slice(clone_start_index, clone_end_index)

        new_cov = np.delete(self.state.covariance, s, axis=0)
        new_cov = np.delete(new_cov, s, axis=1)
        assert (new_cov.shape[0] == new_cov.shape[1])
        self.state.covariance = new_cov

        del self.state.clones[oldest_camera_id]

    def add_camera_features(self, feature_ids, normalized_keypoints):

        mature_feature_ids = []
        newest_clone_id = self.augment_camera_state(0)
        for id, keypoint in zip(feature_ids, normalized_keypoints):
            if id not in self.map_id_to_feature_tracks:
                track = FeatureTrack(id, keypoint, self.camera_id)
                self.map_id_to_feature_tracks[id] = track
                continue

            # Id is in feature track
            track = self.map_id_to_feature_tracks[id]
            track.tracked_keypoints.append(keypoint)
            track.camera_ids.append(newest_clone_id)
            if len(track.tracked_keypoints) >= self.params.max_track_length:
                mature_feature_ids.append(id)

        lost_feature_ids = []
        for id, track in self.map_id_to_feature_tracks.items():
            if track.camera_ids[-1] != newest_clone_id:
                lost_feature_ids.append(id)

        ids_to_update = mature_feature_ids + lost_feature_ids
        self.msckf_update(ids_to_update)

    def check_feature_motion(self, id, min_motion_dist=0.05, use_orthogonal_dist=False):
        track = self.map_id_to_feature_tracks[id]
        if len(track.tracked_keypoints) < 2:
            return False

        if not use_orthogonal_dist:
            first_camera_pose = self.state.clones[track.camera_ids[0]]
            second_camera_pose = self.state.clones[track.camera_ids[-1]]

            global_t_camera1 = first_camera_pose.camera_JPLPose_global.t
            global_t_camera2 = second_camera_pose.camera_JPLPose_global.t

            if np.linalg.norm(global_t_camera1 - global_t_camera2) > min_motion_dist:
                return True
            return False
        else:
            first_camera_pose = self.state.clones[track.camera_ids[0]]

            global_R_camera1 = first_camera_pose.camera_JPLPose_global.q.rotation_matrix().T
            feature_vec = np.append(track.tracked_keypoints[0], 1)
            bearing_vec_camera = feature_vec / np.linalg.norm(feature_vec)
            bearing_in_global = global_R_camera1 * bearing_vec_camera

            for idx in range(1, len(track.camera_ids)):
                clone = self.state.clones[track.camera_ids[idx]]
                global_R_camera = clone.camera_JPLPose_global.q.rotation_matrix().T
                trans = clone.camera_JPLPose_global.t - first_camera_pose.camera_JPLPose_global.t
                parallel_trans = trans.T @ bearing_in_global
                ortho_trans = trans - parallel_trans @ bearing_in_global
                if np.linalg.norm(ortho_trans) > 0.05:
                    return True
            return False

    def msckf_update(self, ids):
        if len(ids) == 0:
            return
        map_good_track_id_to_triangulated_pt = {}
        min_track_removed = 0
        bad_motion_removed = 0
        bad_triangulation = 0
        for id in ids:
            track = self.map_id_to_feature_tracks[id]

            # Check if the track has enough keypoints to justify the expense of a MSCKF update.
            if len(track.tracked_keypoints) < self.params.min_track_length_for_update:
                min_track_removed += 1
                continue

            if not self.check_feature_motion(id):
                bad_motion_removed += 1
                continue
            camera_JPLPose_world_list = []
            # Landmark is good so triangulate it.
            for cam_id in track.camera_ids:
                clone = self.state.clones[cam_id]
                #assert (clone.camera_id == track.camera_ids[idx])
                camera_JPLPose_world_list.append(clone.camera_JPLPose_global)

            is_valid, triangulated_pt = linear_triangulate(camera_JPLPose_world_list, track.tracked_keypoints)

            if not is_valid:
                bad_triangulation += 1
                continue

            is_valid, optimized_pt = optimize_point_location(triangulated_pt, camera_JPLPose_world_list,
                                                             track.tracked_keypoints)

            if not is_valid:
                bad_triangulation += 1
                continue

            map_good_track_id_to_triangulated_pt[id] = optimized_pt

        logger.info("Updating with %i tracks out of %i", len(map_good_track_id_to_triangulated_pt), len(ids))
        logger.info("Removed %i tracks due to track length, and %i due to not enough parallax, %i bad triangulation",
                    min_track_removed, bad_motion_removed, bad_triangulation)

        self.update_with_good_ids(map_good_track_id_to_triangulated_pt)

        for id in ids:
            del self.map_id_to_feature_tracks[id]

    def compute_residual_and_jacobian(self, track, pt_global):
        """
        Compute the jacobian and the residual of a 3D point.
        Args:
            track: Track which contains the measurements, and the associated camera poses
            pt_global: The 3D point in the global frame.

        Returns:


        """
        num_measurements = len(track.tracked_keypoints)

        # Preallocate the our matrices. Note that the number of measurements can end up being smaller
        # if one of the measurements corresponds to a invalid clone(was removed during pruning)
        H_f = np.zeros((2 * num_measurements, 3), dtype=np.float64)
        H_X = np.zeros((2 * num_measurements, self.state.get_state_size()), dtype=np.float64)
        residuals = np.empty((2 * num_measurements, ), dtype=np.float64)

        actual_measurement_count = 0
        last_cam_id = -1
        for idx in range(num_measurements):
            cam_id = track.camera_ids[idx]
            measurement = track.tracked_keypoints[idx]
            clone = None
            clone_index = None
            # We need to iterate through the OrderedDict rather than use the key as we need to find the
            # index within the state vector
            for index, (key, value) in enumerate(self.state.clones.items()):
                if key == cam_id:
                    clone = value
                    clone_index = index
            # Clone doesn't exist/ was removed. Skip this measurement
            if clone_index == None:
                continue
            clone = self.state.clones[cam_id]
            camera_R_global = clone.camera_JPLPose_global.q.rotation_matrix()
            camera_t_global = camera_R_global @ -clone.camera_JPLPose_global.t

            pt_camera = camera_R_global @ pt_global + camera_t_global
            # The actual measurement index. Needed if one of the camera clones is invalid.
            m_idx = actual_measurement_count
            # This slice corresponds to the rows that relate to this measurement.
            measurement_slice = slice(2 * m_idx, 2 * m_idx + 2)
            #Compute and set the residuals
            normalized_x = pt_camera[0] / pt_camera[2]
            normalized_y = pt_camera[1] / pt_camera[2]
            error = np.array([measurement[0] - normalized_x, measurement[1] - normalized_y])

            residuals[measurement_slice] = error

            # Compute the jacobian with respect to the feature position.
            # This can be found around Eq. 23 in the tech report
            X = pt_camera[0]
            Y = pt_camera[1]
            invZ = 1.0 / pt_camera[2]
            jac_i = invZ * np.array([[1.0, 0.0, -X * invZ], [0.0, 1.0, -Y * invZ]])

            H_f[measurement_slice] = jac_i @ camera_R_global

            # Compute jacobian with respect to the current camera clone
            # Eq 22 in the tech report
            jac_attitude = jac_i @ skew_matrix(pt_camera)
            jac_position = -jac_i @ camera_R_global

            # Get the index of the current clone within the state vector. As we need to set their computed jacobians
            clone_state_index = self.state.calc_clone_index(clone_index)

            att_start_index = clone_state_index + StateInfo.CLONE_ATT_SLICE.start
            att_end_index = clone_state_index + StateInfo.CLONE_ATT_SLICE.stop
            # Reads as the partial derivatives(jacobians) of the measurement with respect to the clone attitude.
            H_X[measurement_slice, att_start_index:att_end_index] = jac_attitude

            pos_start_index = clone_state_index + StateInfo.CLONE_POS_SLICE.start
            pos_end_index = clone_state_index + StateInfo.CLONE_POS_SLICE.stop
            H_X[measurement_slice, pos_start_index:pos_end_index] = jac_position

            actual_measurement_count += 1

        if actual_measurement_count != num_measurements:
            assert (False)

        return actual_measurement_count, residuals, H_X, H_f

    def project_left_nullspace(self, matrix):
        """Figure out the left nullspace of the matrix.

        Args:
            matrix: Matrix we want to compute the left nullspace of.

        Returns:
            Matrix representing the left nullspace of the input matrix.

        Linear Algebra recap:
            * The nullspace or kernel is the set of solutions that map to the zero vector.

             matrix * nullspace = 0.

            * The left nullspace or cokernel is the solution that will map to the zero vector if multiplied from the
            left side.

            left_nullspace * matrix = 0

            It can be found by finding the nullspace of the matrix transposed.

            left_nullspace = nullspace(matrix^T)
        """
        A = scipy.linalg.null_space(matrix.T)
        return A

    def chi_square_test(self, H_o, residual, dof):
        noise = np.eye(H_o.shape[0]) * self.params.keypoint_noise
        innovation = H_o @ self.state.covariance @ H_o.T + noise
        # gamma = r^T * (H*P*H^T + R)^-1 * r
        # (H*P*H^T + R)^-1 * r = np.linalg.solve((H*P*H^T + R), r)
        gamma = residual.T @ np.linalg.solve(innovation, residual)
        if gamma < self.chi_square_val[dof]:
            return True
        return False

    def update_with_good_ids(self, map_good_track_ids_to_point_3d):
        """ Run an EKF update with valid tracks.

        Args:
            map_good_track_ids_to_point_3d: Dict which maps track ids to its triangulated point. Should only contain
                tracks that have gone through some sort of preprocessing to remove outliers.

        This function computes the individual jacobians and residuals for each track and combines them into 2 large
        matrices for a big EKF Update at the end.

        It is in this function we use the so called MSCKF update, or nullspace projection which is the big innovation
        introduced in A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint Kalman Filter for Vision-Aided
        Inertial Navigation", and can be found on Equations 23,24.

        The nullspace projection allows us to remove

        """

        num_landmarks = len(map_good_track_ids_to_point_3d)

        if num_landmarks == 0:
            return

        # Here we preallocate our update matrices. This way we can do 1 big EKF update
        # rather then many small ones(is much more efficient).
        # This is the maximum size possible of our update matrix. Each feature provides 2
        # residuals per keypoint * the maximum number of keypoints(max_track_length). The -3
        # comes from the nullspace projection which is explained below.
        max_possible_size = num_landmarks * 2 * self.params.max_track_length - 3
        H = np.empty((max_possible_size, self.state.get_state_size()))
        r = np.zeros((max_possible_size, ))
        index = 0
        for id, triangulated_pt in map_good_track_ids_to_point_3d.items():
            track = self.map_id_to_feature_tracks[id]
            actual_num_measurements, residuals, H_X, H_f = self.compute_residual_and_jacobian(track, triangulated_pt)

            # Nullspace projection.
            A = self.project_left_nullspace(H_f)

            H_o = A.T @ H_X

            rows, cols = H_o.shape
            assert (rows == 2 * actual_num_measurements - 3)
            assert (cols == H_X.shape[1])

            r_o = A.T @ residuals

            dof = residuals.shape[0] / 2 - 1
            if not self.chi_square_test(H_o, r_o, dof):
                continue

            num_residuals = residuals.shape[0]
            start_row = index
            end = index + num_residuals - 3
            r[start_row:end] = r_o
            H[start_row:end] = H_o
            index += num_residuals - 3

        if index == 0:
            return
        final_r = r[0:index]
        final_H = H[0:index]

        R = np.zeros((final_r.shape[0], final_r.shape[0]))
        np.fill_diagonal(R, self.params.keypoint_noise**2)
        self.update_EKF(final_r, final_H, R)

    def residualize(self, track):
        camera_JPLPose_world_list = []

        num_measurements = len(track.camera_ids)

        for cam_id in track.camera_ids:
            clone = self.state.clones[cam_id]
            assert (clone.camera_id == track.camera_ids[idx])
            camera_JPLPose_world_list.append(clone.camera_JPLPose_global)

        is_valid, triangulated_pt = linear_triangulate_point(camera_JPLPose_world_list, track.track_keypoints)

        if not is_valid:
            return False

        optimized_pt = optimize_point_location(triangulated_pt, camera_JPLPose_world_list, track.track_keypoints)

    def integrate(self, imu_measurement):
        dt = imu_measurement.time_interval
        unbiased_gyro = imu_measurement.angular_vel - self.state.bias_gyro
        unbiased_acc = imu_measurement.linear_acc - self.state.bias_acc

        gyro_norm = np.linalg.norm(unbiased_gyro)

        q0 = self.state.imu_JPLQ_global
        p0 = self.state.global_t_imu
        v0 = self.state.velocity

        omega = jpl_omega(unbiased_gyro)

        if (gyro_norm > 1e-5):
            dq_dt_arr = (np.cos(gyro_norm * dt * 0.5) * np.eye(4) +
                         1 / gyro_norm * np.sin(gyro_norm * dt * 0.5) * omega) @ q0.q
            dq_dt2_arr = (np.cos(gyro_norm * dt * 0.25) * np.eye(4) +
                          1 / gyro_norm * np.sin(gyro_norm * dt * 0.25) * omega) @ q0.q
        else:
            dq_dt_arr = (np.eye(4) + 0.5 * dt * omega) * np.cos(gyro_norm * dt * 0.5) @ q0.q
            dq_dt2_arr = (np.eye(4) + 0.25 * dt * omega) * np.cos(gyro_norm * dt * 0.25) @ q0.q

        dR_dt_transpose = JPLQuaternion.from_array(dq_dt_arr).rotation_matrix().T
        dR_dt2_transpose = JPLQuaternion.from_array(dq_dt2_arr).rotation_matrix().T

        k1_v_dot = q0.rotation_matrix().T @ unbiased_acc + self.gravity

        k1_p_dot = v0

        # k2 = f(tn + dt / 2, yn + k1 * dt / 2)
        k1_v = v0 + k1_v_dot * dt / 2
        k2_v_dot = dR_dt2_transpose @ unbiased_acc + self.gravity

        k2_p_dot = k1_v

        # k3 = f(tn + dt / 2, yn + k2 * dt / 2)

        k2_v = v0 + k2_v_dot * dt / 2
        k3_v_dot = dR_dt2_transpose @ unbiased_acc + self.gravity
        k3_p_dot = k2_v

        # k4 = f(tn + dt, yn + k3 * dt)
        k3_v = v0 + k3_v_dot * dt
        k4_v_dot = dR_dt_transpose @ unbiased_acc + self.gravity
        k4_p_dot = k3_v

        # yn + 1 = yn + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        #        q = dq_dt
        v = v0 + dt / 6 * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot)
        p = p0 + dt / 6 * (k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot)

        self.state.imu_JPLQ_global = JPLQuaternion.from_array(dq_dt_arr)
        self.state.global_t_imu = p
        self.state.velocity = v

    def propogate(self, imu_buffer):

        for imu in imu_buffer:
            F = self.compute_F(imu)
            G = self.compute_G()
            self.integrate(imu)

            Phi = None
            transition_method = AlgorithmConfig.MSCKFParams.StateTransitionIntegrationMethod
            if self.params.state_transition_integration == transition_method.Euler:
                Phi = np.eye(15) + F * imu.time_interval
            elif self.params.state_transition_integration == transition_method.truncation_3rd_order:
                Fdt = F * imu.time_interval
                Fdt2 = Fdt @ Fdt
                Fdt3 = Fdt2 @ Fdt
                Phi = np.eye(15) + Fdt + 0.5 * Fdt2 + 1.0 / 6.0 * Fdt3
            elif self.params.state_transition_integration == transition_method.matrix_exponent:
                Fdt = F * imu.time_interval
                Phi = scipy.linalg.expm(Fdt)

            imu_covar = self.state.covariance[0:StateInfo.IMU_STATE_SIZE, 0:StateInfo.IMU_STATE_SIZE]

            transition_model = self.params.transition_matrix_method
            if transition_model == AlgorithmConfig.MSCKFParams.TransitionMatrixModel.continous_discrete:
                Q = G @ self.noise_matrix @ G.T * imu.time_interval
                new_covariance = Phi @ (imu_covar + Q) @ Phi.T
            else:
                Q = G @ self.noise_matrix @ G.T * imu.time_interval
                new_covariance = Phi @ imu_covar @ Phi.T + Q

            # Update the imu-camera covariance
            self.state.covariance[0:15, 15:] = (Phi @ self.state.covariance[0:15, 15:])
            self.state.covariance[15:, :15] = (self.state.covariance[15:, :15] @ Phi.T)

            new_cov_symmetric = symmeterize_matrix(new_covariance)
            self.state.covariance[0:StateInfo.IMU_STATE_SIZE, 0:StateInfo.IMU_STATE_SIZE] = new_cov_symmetric

    def update_EKF(self, res, H, R):
        assert (R.shape[0] == res.shape[0])
        assert (H.shape[0] == res.shape[0])
        logger.info("Residual norm is %f", np.linalg.norm(res))

        if H.shape[0] > H.shape[1] and self.params.use_QR_compression:
            # QR decomposition
            Q1, Q2 = np.linalg.qr(H, mode='reduced')  # if M > N, return (M, N), (N, N)
            H_thin = Q2  # shape (N, N)
            r_thin = Q1.T @ res  # shape (N,)
            R_thin = Q1.T @ R @ Q1
        else:
            H_thin = H  # shape (M, N)
            r_thin = res  # shape (M)
            R_thin = R

        H = H_thin
        res = r_thin
        R = R_thin
        H_T = H.transpose()

        cur_cov = self.state.covariance
        K = cur_cov @ H_T @ np.linalg.inv((H @ cur_cov @ H_T + R))
        state_size = self.state.get_state_size()

        # Update the covariance using the joseph form(is more numerically stable)
        new_cov = (np.eye(state_size) - K @ H) @ cur_cov @ (np.eye(state_size) - K @ H).T + K @ R @ K.T
        delta_x = K @ res

        # Apply the new covariance and the update
        self.state.covariance = new_cov
        self.state.update_state(delta_x)

        return 0

    def augment_camera_state(self, timestamp):
        imu_R_global = self.state.imu_JPLQ_global.rotation_matrix()
        camera_R_imu = self.camera_calib.imu_R_camera.T

        # Compute the pose of the camera in the global frame
        camera_R_global = camera_R_imu @ imu_R_global

        global_t_imu = self.state.global_t_imu
        imu_t_camera = self.camera_calib.imu_t_camera
        global_R_imu = imu_R_global.T
        global_t_camera = global_t_imu + global_R_imu @ imu_t_camera

        camera_JPLQ_global = JPLQuaternion.from_rot_mat(camera_R_global)

        cur_state_size = self.state.get_state_size()
        # This jacobian stores the partial derivatives of the camera position(6 states) with respect to the current
        # state vector
        jac = np.zeros((StateInfo.CLONE_STATE_SIZE, cur_state_size), dtype=np.float64)
        # Its almost all zeros except for with respect to the current pose(quaternion+position)

        # See A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint Kalman Filter for Vision-Aided Inertial
        # Navigation," Eq 16
        jac[StateInfo.CLONE_ATT_SLICE, StateInfo.ATT_SLICE] = camera_R_imu
        jac[StateInfo.CLONE_POS_SLICE, StateInfo.ATT_SLICE] = skew_matrix(global_R_imu @ imu_t_camera)
        jac[StateInfo.CLONE_POS_SLICE, StateInfo.POS_SLICE] = np.eye(3)

        new_state_size = cur_state_size + StateInfo.CLONE_STATE_SIZE
        # See Eq 15 from above reference
        augmentation_matrix = np.eye(new_state_size, cur_state_size)
        augmentation_matrix[cur_state_size:, :] = jac

        new_covariance = augmentation_matrix @ self.state.covariance @ augmentation_matrix.transpose()
        # Helps with numerical problems
        new_cov_sym = symmeterize_matrix(new_covariance)

        # Add the camera clone and set the new covariance matrix which includes it
        camera_JPLPose_global = JPLPose(camera_JPLQ_global, global_t_camera)
        self.camera_id += 1
        clone = CameraClone(camera_JPLPose_global, self.camera_id)
        self.state.add_clone(clone)
        self.state.covariance = new_cov_sym

        return self.camera_id

    def compute_F(self, imu):
        """Computes the transition matrix(F) for the extended kalman filter.

        Args:
            unbiased_gyro_meas:
            unbiased_acc_measurement:

        Returns:

            The transition matrix contains the jacobians of our process model with respect to our current state.

            The jacobians for this function can be found in A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint
            Kalman Filter for Vision-Aided Inertial Navigation", but with a slightly different variable ordering.

        """

        unbiased_gyro_meas = imu.angular_vel - self.state.bias_gyro
        unbiased_acc_meas = imu.linear_acc - self.state.bias_acc

        F = np.zeros((15, 15), dtype=np.float64)

        imu_SO3_global = self.state.imu_JPLQ_global.rotation_matrix()
        st = StateInfo

        # This matrix can be found in A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint Kalman Filter for
        # Vision-Aided Inertial Navigation," but with a slight different row/column ordering.
        # How to read the indexing. The first index shows the value we are taking the partial derivative of and the
        # second index is which partial derivative.
        # E.g F[st.ATTITUDE_SLICE,st.BIAS_GYRO_SLICE] is the partial derivative of the attitude with respect
        # to the gyro bias.
        F[st.ATT_SLICE, st.ATT_SLICE] = -skew_matrix(unbiased_gyro_meas)
        F[st.ATT_SLICE, st.BG_SLICE] = -np.eye(3)
        F[st.VEL_SLICE, st.ATT_SLICE] = -imu_SO3_global.transpose() @ skew_matrix(unbiased_acc_meas)
        F[st.VEL_SLICE, st.BA_SLICE] = -imu_SO3_global.transpose()
        F[st.POS_SLICE, st.VEL_SLICE] = np.eye(3)
        return F

    def compute_G(self):
        """ Computes the system matrix with respect to the control input.

        Returns:
            A numpy matrix(15x12) containing the jacobians with respect to the system input.

        This matrix contains the jacobians of our process model with respect to the system input.


        The jacobians for this function can be found in A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint
        Kalman Filter for Vision-Aided Inertial Navigation", but with a slightly different variable ordering.

        """

        G = np.zeros((15, 12), dtype=np.float64)
        imu_SO3_global = self.state.imu_JPLQ_global.rotation_matrix()
        st = StateInfo

        # This matrix can be found in A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint Kalman Filter for
        # Vision-Aided Inertial Navigation," but with a slightly different row/column ordering.

        # order of noise: gyro_m , acc_m , gyro_b, acc_b
        G[st.ATT_SLICE, 0:3] = -np.eye(3)
        G[st.BG_SLICE, 6:9] = np.eye(3)
        G[st.VEL_SLICE, 3:6] = -imu_SO3_global.T
        G[st.BA_SLICE, 9:12] = np.eye(3)

        return G
