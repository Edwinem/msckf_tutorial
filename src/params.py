from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np


class ViewerConfig():
    """Stores the parameters for our Visualization/Viewer."""
    class WindowTypes():
        """Different ways to initialize the OpenGL window.

        We utilize the moderngl_window class to handle our OpenGL context and other stuff such as keyboard bindings,
        mouse control,...

        moderngl_window allows for multiple different ways to initialize the OpenGL context. These are the possible
        options.
        """
        default = None
        pyglet = "pyglet"
        pygame = "pygame2"
        glfw = "glfw"
        sdl2 = "sdl2"
        pyside2 = "pyside2"
        pyqt5 = "pyqt5"
        tk = "tk"

    def __init__(self):
        # Param which controls which library we use to initialize the OpenGL window. Must be from 'WindowTypes'
        self.window_type = ViewerConfig.WindowTypes.default
        self.vsync = True
        # Size of the Viewer window
        self.size = (1920, 1080)
        # Is the window resizeable
        self.resizable = True
        # Set the window to fullscreen.
        self.fullscreen = False
        # Show the cursor in the OpenGL window
        self.show_cursor = True


class AlgorithmConfig():

    @dataclass
    class FeatureTrackerParams():

        @dataclass
        class OpticalFlowTrackerParams():
            window_size = 15
            use_iteration_stopping_criteria = True
            use_minimum_error_stopping_criteria = True
            max_iterations = 30
            minimum_error = 1e-3
            max_pyramid_level = 3

            def to_opencv_dict(self):
                stopping_criteria = 0
                if self.use_iteration_stopping_criteria:
                    stopping_criteria = stopping_criteria | cv2.TERM_CRITERIA_COUNT
                if self.use_minimum_error_stopping_criteria:
                    stopping_criteria = stopping_criteria | cv2.TERM_CRITERIA_EPS

                return dict(winSize=(self.window_size, self.window_size),
                            maxLevel=self.max_pyramid_level,
                            criteria=(stopping_criteria, self.max_iterations, self.minimum_error))

        @dataclass()
        class KeyPointDetectorParams():

            max_corners = 200
            quality_level = 0.01
            min_distance = 10.0
            sub_pix_window_size = 10
            sub_pix_zero_zone = -1

        lk_params: OpticalFlowTrackerParams = OpticalFlowTrackerParams()
        detector_params: KeyPointDetectorParams = KeyPointDetectorParams()

        numpy_random_seed = None
        sub_pix_window_size = 5
        sub_pix_zero_zone = -1

        small_angle_threshold = 0.001745329

        block_size = 200

        ransac_iterations = 300
        ransac_threshold = 1e-4

        min_tracked_features = 200
        max_tracked_features = 200

        grid_block_size = 100
        max_keypoints_per_block = 10
        min_dist_between_keypoints = 10

    @dataclass
    class MSCKFParams():
        class StateTransitionIntegrationMethod(Enum):
            """ Enum which contains different ways to compute the state transition matrix.

            The state transition matrix(generally represented as Phi) requires solving an integral.

            Depending on the accuracy, and other factors we can use different integration schemes to compute/approximate
            the state transition matrix.

            Euler:
                Simplest and quickest. Simply multiply the transition matrix by the time interval.

            truncation_3rd_order:
                truncation is an integration scheme with varying levels of accuracy depending on the order. An overview of
                this method can be found in "Quaternion kinematics for the error-state Kalman filter" by Joan Sola in
                Appendix C.

            matrix_exponent:

            """
            Euler = auto()
            truncation_3rd_order = auto()  # Euler integration
            matrix_exponent = auto()  # Matrix exponential
            # RK4 = auto()  # Runge Kutta 4

        class TransitionMatrixModel(Enum):
            discrete = auto()
            continous_discrete = auto()

        max_track_length = 20
        min_track_length_for_update = 3

        use_observability_constraint = False

        keypoint_noise = 0.35

        max_clone_num = 20

        use_QR_compression = True

        transition_matrix_method = TransitionMatrixModel.discrete
        state_transition_integration = StateTransitionIntegrationMethod.truncation_3rd_order

    feature_tracker_params: FeatureTrackerParams = FeatureTrackerParams()
    msckf_params: MSCKFParams = MSCKFParams()


@dataclass()
class EurocDatasetCalibrationParams():
    T_imu_cam0 = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                           [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                           [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                           [0, 0, 0, 1.000000000000000]])
    cam0_distortion_model = 'radtan'
    cam0_distortion_coeffs = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
    cam0_intrinsics = np.array([458.654, 457.296, 367.215, 248.375])
    cam0_resolution = np.array([752, 480])
