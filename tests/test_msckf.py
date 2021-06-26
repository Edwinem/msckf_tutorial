import pytest
from transforms3d.euler import euler2mat

from msckf import *
from msckf_types import CameraCalibration, IMUData, PinholeIntrinsics
from params import Config, EurocDatasetCalibrationParams
from spatial_transformations import hamiltonian_quaternion_to_rot_matrix
from tests.utilities import project_point


def setup_euroc_calibration():
    euroc_calib = EurocDatasetCalibrationParams()
    camera_calib = CameraCalibration()
    camera_calib.intrinsics = PinholeIntrinsics.initialize(euroc_calib.cam0_intrinsics,
                                                           euroc_calib.cam0_distortion_model,
                                                           euroc_calib.cam0_distortion_coeffs)
    camera_calib.set_extrinsics(euroc_calib.T_imu_cam0)
    return camera_calib


def setup_msckf():
    camera_calib = setup_euroc_calibration()

    msckf = MSCKF(Config.msckf_params, camera_calib)
    msckf.set_imu_noise(0.0, 0.0, 0.0, 0.0)
    zero3 = np.zeros((3, ))
    msckf.initialize(np.eye(3), zero3, zero3, zero3, zero3)
    return msckf


# def test_msckf_imu_integration():
#
#     msckf = setup_msckf()
#
#     # Constant linear acceleration
#
#     linear_acceleration = 0.3
#     gravity = 9.81
#
#     acc_measurement = np.array([linear_acceleration,0,gravity])
#     gyro_meas =  np.zeros((3,))
#
#     dt = 1.0
#
#     data = IMUData(acc_measurement,gyro_meas,0,dt)
#
#
#     imu_buffer = []
#     imu_buffer.append(data)
#
#     msckf.propogate(imu_buffer)
#
#     expected_translation_x = linear_acceleration * dt**2/2
#     expected_velocity_x = linear_acceleration * dt
#
#     expected_translation = np.array([expected_translation_x,0,0])
#     expected_velocity = np.array([expected_velocity_x,0,0])
#
#     assert (np.allclose(expected_translation,msckf.state.global_t_imu))
#     assert(np.allclose(expected_velocity,msckf.state.velocity))
#
#     # Orientation should not change as no gyro measurement
#     assert(np.allclose(np.array([0,0,0,1.0]),msckf.state.imu_JPLQ_global.q))
#
#     msckf = setup_msckf()
#
#     imu_buffer.append(data)
#
#     msckf.propogate(imu_buffer)
#
#     new_x = expected_translation_x + expected_velocity_x*dt + linear_acceleration*dt**2/2
#
#     assert (np.allclose(np.array([new_x,0,0]), msckf.state.global_t_imu))


def test_constant_angular_velocity():

    msckf = setup_msckf()
    angular_velocity = np.pi / 180
    gravity = 9.81
    accel = np.array([0, 0, gravity])
    gyro = np.array([0, 0, angular_velocity])

    dt = 1.0

    data = IMUData(accel, gyro, 0, dt)

    msckf.integrate(data)

    final_w = np.cos(angular_velocity * dt / 2)
    final_z = np.sin(angular_velocity * dt / 2)

    final_quat = np.array([0, 0, final_z, final_w])
    assert (np.allclose(final_quat, msckf.state.imu_JPLQ_global.q))


def test_state_augmentation():

    msckf = setup_msckf()
    msckf.augment_camera_state(0)

    state = msckf.state

    assert (state.num_clones() == 1)

    assert (state.covariance.shape[0] == state.get_state_size())

    euroc_calib = setup_euroc_calibration()

    _, clone = state.clones.popitem()

    pose = clone.camera_JPLPose_global

    camera_R_global = pose.q.rotation_matrix()
    global_t_camera = pose.t

    global_R_camera = camera_R_global.T

    # Since imu is at the origin the camera extrinsics calibration and the clone pose should match

    assert (np.allclose(global_R_camera, euroc_calib.imu_R_camera))
    assert (np.allclose(global_t_camera, euroc_calib.imu_t_camera))


def test_state_augmentation_non_origin():

    msckf = setup_msckf()
    euroc_calib = setup_euroc_calibration()

    state = msckf.state
    new_imu_rot = euler2mat(2.7, 3.0, .5)
    new_camera_rotation = new_imu_rot @ euroc_calib.imu_R_camera

    new_imu_att = JPLQuaternion.from_rot_mat(new_imu_rot.T)

    state.imu_JPLQ_global = new_imu_att

    msckf.augment_camera_state(0)

    _, clone = state.clones.popitem()

    pose = clone.camera_JPLPose_global

    camera_R_global = pose.q.rotation_matrix()
    global_t_camera = pose.t

    global_R_camera = camera_R_global.T

    assert (np.allclose(global_R_camera, new_camera_rotation))
    assert (np.allclose(global_t_camera, new_imu_rot @ euroc_calib.imu_t_camera))


def zero_init_track():
    t = FeatureTrack(0, 0, 0)
    t.tracked_keypoints = []
    t.camera_ids = []
    return t


def test_residual_and_jacobian():
    msckf = setup_msckf()

    # Set identity for camera calibration so its pose and the IMUs are the same.
    msckf.camera_calib.imu_R_camera = np.eye(3)
    msckf.camera_calib.imu_t_camera = np.zeros((3, ))

    pt_in_world = np.array([-1, .5, 4.7])
    measurements = []
    translations = [np.array([0.1, 0.2, -.4]), np.array([-0.25, .3, 0.35]), np.array([0.6, .1, -.1])]
    start_camera_id = msckf.camera_id
    track = zero_init_track()
    for t in translations:
        pose = np.eye(4)
        pose[0:3, 3] = t
        measurement = (project_point(pose, pt_in_world))
        track.tracked_keypoints.append(measurement)
        track.camera_ids.append(start_camera_id + 1)
        msckf.state.global_t_imu = t
        msckf.augment_camera_state(0)
        start_camera_id = msckf.camera_id

    # actual_num_measurements, residuals, H_X, H_f=msckf.compute_residual_and_jacobian(track,pt_in_world)
    #
    # expected_size = len(translations)
    # assert (actual_num_measurements==expected_size)
    # assert (residuals.size ==expected_size*2)
    #
    # #Perfect measurements so residual should be 0
    # assert (np.linalg.norm(residuals)==0)
    #
    # A=msckf.project_left_nullspace(H_f)
    #
    # H_o = A.T @ H_X
    #
    # r_o = A.T @ residuals
    #
    # R = np.zeros((r_o.shape[0], r_o.shape[0]))
    # np.fill_diagonal(R, 0.001 ** 2)
    #
    # before_update = msckf.state.global_t_imu
    # msckf.update_EKF(r_o, H_o, R)
    #
    # after_update =msckf.state.global_t_imu
    #
    # assert (np.allclose(before_update,after_update))

    new_translation = np.array([0.3, -.1, .75])
    pose = np.eye(4)
    pose[0:3, 3] = new_translation
    measurement = (project_point(pose, pt_in_world))
    track.tracked_keypoints.append(measurement)
    track.camera_ids.append(start_camera_id + 1)
    perturbed_translation = new_translation + 0.4
    msckf.state.global_t_imu = perturbed_translation
    msckf.augment_camera_state(0)

    actual_num_measurements, residuals, H_X, H_f = msckf.compute_residual_and_jacobian(track, pt_in_world)
    # Perfect measurements so residual should be 0
    assert (np.linalg.norm(residuals) != 0)

    A = msckf.project_left_nullspace(H_f)

    H_o = A.T @ H_X

    r_o = A.T @ residuals

    R = np.zeros((r_o.shape[0], r_o.shape[0]))
    R2 = np.zeros((residuals.shape[0], residuals.shape[0]))
    np.fill_diagonal(R2, 0.00000000001)
    # R2[6,6] = 2.0
    # R2[7,7] = 2.0
    R = A.T @ R2 @ A
    #np.fill_diagonal(R, 1)

    before_update = np.copy(msckf.state.global_t_imu)
    msckf.state.covariance[:StateInfo.IMU_STATE_SIZE, :StateInfo.IMU_STATE_SIZE] = np.eye(StateInfo.IMU_STATE_SIZE) * 10
    msckf.state.covariance[-6:, -6:] = np.eye(6) * 10
    print("Clones before")
    for clone in msckf.state.clones.values():
        print(clone.camera_JPLPose_global)
    msckf.update_EKF(r_o, H_o, R)

    print("Clones after")
    for clone in msckf.state.clones.values():
        print(clone.camera_JPLPose_global.t)

    after_update = msckf.state.global_t_imu
    print(after_update)

    assert (np.allclose(before_update, after_update))
