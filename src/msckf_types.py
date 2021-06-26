from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PinholeIntrinsics():

    fx: float = 1.0
    fy: float = 1.0
    cx: float = 0.0
    cy: float = 0.0

    distortion_model = ""

    dist_coeffs: np.ndarray = np.array([])

    def generate_K(self):
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K

    @classmethod
    def initialize(cls, intrinsics_arr, distortion_name="", dist_coeffs=None):
        if intrinsics_arr.size != 4:
            raise TypeError("Intrinsics numpy array must have 4 values(fx,fy,cx,cy)")
        intrinsics = PinholeIntrinsics(intrinsics_arr[0], intrinsics_arr[1], intrinsics_arr[2], intrinsics_arr[3])
        if distortion_name != "" and dist_coeffs is not None:
            intrinsics.distortion_model = distortion_name
            intrinsics.dist_coeffs = dist_coeffs
        return intrinsics

    def set_distortion(self, distortion_params, distortion_model="radtan"):
        self.dist_coeffs = distortion_params
        self.distortion_model = distortion_model


@dataclass
class CameraCalibration():

    # Position of camera with respect to imu/body
    imu_R_camera: np.ndarray = None
    imu_t_camera: np.ndarray = None

    # Pinhole camera calibration only
    intrinsics: PinholeIntrinsics = PinholeIntrinsics()

    def set_extrinsics(self, imu_T_camera):
        self.imu_R_camera = imu_T_camera[0:3, 0:3]
        self.imu_t_camera = imu_T_camera[0:3, 3]

    @classmethod
    def initialize(cls, imu_T_camera, intrinsics):
        c = CameraCalibration()
        c.set_extrinsics(imu_T_camera)
        c.intrinsics = intrinsics
        return c


@dataclass
class IMUData():

    linear_acc: np.ndarray
    angular_vel: np.ndarray
    timestamp: float
    time_interval: float = 0


class FeatureTrack():
    def __init__(self, id, keypoint, camera_id):
        self.id = id
        self.tracked_keypoints = []
        self.camera_ids = []
        self.tracked_keypoints.append(keypoint)
        self.camera_ids.append(camera_id)

        self.has_triangulated_pt = False
        self.triangulated_pt = None
