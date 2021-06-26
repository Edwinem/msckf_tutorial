import csv
import logging
import math
import os
import sys
from dataclasses import dataclass

import click
import cv2
import numpy as np


import logging

import matplotlib.pyplot as plt
import PyVIOSIM

from feature_tracker import FeatureTracker
from msckf import MSCKF
from msckf_types import CameraCalibration, IMUData, PinholeIntrinsics
from params import Config, EurocDatasetCalibrationParams
from spatial_transformations import hamiltonian_quaternion_to_rot_matrix

logger = logging.getLogger(__name__)

LEFT_CAMERA_FOLDER = "cam0"
RIGHT_CAMERA_FOLDER = "cam1"
IMU_FOLDER = "imu0"
GT_FOLDER = "state_groundtruth_estimate0"
DATA_FILE = "data.csv"

TIMESTAMP_INDEX = 0

NANOSECOND_TO_SECOND = 1e-9


def run_on_euroc():
    logging.basicConfig(format='%(filename)s: %(message)s', level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True

    sim_params = PyVIOSIM.SimParams()
    sim_params.trajectory_file = "..."
    simulator = PyVIOSIM.Simulator(sim_params)

    euroc_calib = EurocDatasetCalibrationParams()
    camera_calib = CameraCalibration()
    camera_calib.set_extrinsics(np.eye(4))
    feature_tracker = FeatureTracker(Config().feature_tracker_params, camera_calib)

    msckf = MSCKF(Config.msckf_params, camera_calib)
    msckf.set_imu_noise(0.005, 0.05, 0.001, 0.01)
    msckf.set_imu_covariance(1e-5, 1e-12, 1e-2, 1e-2, 1e-2)

    poses_x = []
    poses_y = []
    imu_buffer = []
    last_imu_timestamp = -1
    first_time = True
    np.set_printoptions(precision=5)
    init_state = simulator.get_state(simulator.current_timestamp())
    assert (init_state[0])
    init_timestamp = init_state[1]
    init_vec = init_state[2]
    quat = init_vec[1:5]
    pos = init_vec[5:8]
    vel = init_vec[8:11]
    bias_gyro = init_vec[11:14]
    bias_acc = init_vec[14:17]

    msckf.initialize2(quat, pos, vel, bias_acc, bias_gyro)
    first_time = True
    while simulator.ok():

        imu_data = simulator.get_next_imu()
        if imu_data[0]:

            imu_timestamp = imu_data[1]
            print(imu_timestamp)
            gyro = imu_data[2]
            acc = imu_data[3]

            if last_imu_timestamp != -1:
                dt = imu_timestamp - last_imu_timestamp
            else:
                dt = 0.005
            imu_buffer.append(IMUData(acc, gyro, imu_timestamp, dt))

        cam_data = simulator.get_next_cam()
        if cam_data[0]:
            cam_timestamp = cam_data[1]
            cam_ids = cam_data[2]
            features_per_cam = cam_data[3]
            uv_points = []
            good_ids = []
            for feat in features_per_cam[0]:
                good_ids.append(feat[0])
                uv_points.append(feat[1][3])
                uv_points.append(feat[1][4])
            np_uv = np.array(uv_points).reshape(-1, 2)
            ids = np.array(good_ids)
            msckf.propogate(imu_buffer)
            state_vals = simulator.get_pose(imu_timestamp)
            assert state_vals[0]
            state_vec = state_vals[2]
            trans = state_vec[5:8]
            print("Error after IMU")
            print(np.linalg.norm(msckf.state.global_t_imu - trans))
            # print("IMU after")
            # print(msckf.state.global_t_imu)
            # print(gt_pos)
            msckf.add_camera_features(ids, np_uv)
            state_vals = simulator.get_pose(cam_timestamp)
            assert state_vals[0]
            state_vec = state_vals[2]
            trans = state_vec[5:8]
            print("Error after Camera")
            print(np.linalg.norm(msckf.state.global_t_imu - trans))
            msckf.remove_old_clones()
            x_val = msckf.state.global_t_imu[2]
            y_val = msckf.state.global_t_imu[1]
            # poses_x.append(msckf.state.global_t_imu[2])
            # poses_y.append(msckf.state.global_t_imu[1])
            # poses_x.append(trans[2])
            # poses_y.append(trans[1])
            imu_buffer.clear()
            # plt.scatter(x_val, y_val)
            # plt.pause(0.05)

    #plt.plot(poses_x, poses_y)
    plt.show()


if __name__ == '__main__':
    run_on_euroc()
