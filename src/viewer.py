from multiprocessing import Process, Queue

import moderngl
import moderngl_window
import numpy as np
from pyrr import Matrix44

from params import ViewerConfig
from render_camera import CameraWindow
from render_primitives import (CameraShape, TrajectoryRenderable, default_fragment_shader, default_vertex_shader)


def convert_mat44_to_pyrr_format(mat4x4):
    """ Converts a 4x4 homogenous transfrom to a format the pyrr library will accept.

    Args:
        mat4x4: 4x4 numpy matrix representing a homogenous transform.

    Returns:
        A modified matrix which the pyrr library can directly accept.

    We store a homogenous matrix as

      R  | t
      0_3  1

    pyrr stores it as

    R^T | 0_3
     t     1
    """
    new_mat = np.empty((4, 4), dtype=mat4x4.dtype)
    new_mat[0:3, 0:3] = mat4x4[0:3, 0:3].T
    new_mat[3, 0:3] = mat4x4[0:3, 3]
    new_mat[0:3, 3] = 0
    new_mat[3, 3] = 1.0
    return new_mat


class SLAMViewer(CameraWindow):
    title = "SLAM Viewer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = False
        self.prog = self.ctx.program(vertex_shader=default_vertex_shader, fragment_shader=default_fragment_shader)
        self.estimated_trajectory = TrajectoryRenderable()
        self.ground_truth_trajectory = TrajectoryRenderable()
        self.ground_truth_trajectory.set_color(0, 1.0, 0)
        self.current_cam = CameraShape()
        self.current_cam.set_color(50 / 255.0, 205.0 / 255.0, 50.0 / 255.0)
        self.current_cam.can_render = False

        self.pose_queue = None
        self.gt_pose_queue = None

        self.renderables = []
        self.renderables.extend([self.estimated_trajectory, self.ground_truth_trajectory, self.current_cam])

    def add_gt_pose(self, mat):
        self.gt_pose_queue.put(mat)

    def add_est_pose(self, mat):
        self.pose_queue.put(mat)

    def render(self, time: float, frametime: float):
        self.ctx.enable_only(moderngl.DEPTH_TEST)

        if self.gt_pose_queue is not None:
            if not self.gt_pose_queue.empty():
                self.ground_truth_trajectory.can_render = True
            while not self.gt_pose_queue.empty():
                gt_pose = self.gt_pose_queue.get()
                self.ground_truth_trajectory.add_pose(gt_pose[0:3, 3])

        if not self.pose_queue.empty():
            while not self.pose_queue.empty():
                pose = self.pose_queue.get()
                self.estimated_trajectory.add_pose(pose[0:3, 3].astype(np.float32))
            pyrr_pose = Matrix44(convert_mat44_to_pyrr_format(pose), dtype=np.float32)
            self.current_cam.set_model_matrix(pyrr_pose)
            self.current_cam.can_render = True
            self.estimated_trajectory.can_render = True
            self.camera.target = pose[0:3, 3]

        for r in self.renderables:

            r.set_projection(self.camera.projection.matrix)
            r.set_camera_matrix(self.camera.matrix)
            r.render(self.prog)


def create_window(config_cls: moderngl_window.WindowConfig, config=ViewerConfig()):
    """Instantiates and returns a moderngl window config.

    Args:
        config_cls: The WindowConfig class to render. Note this is the class name not an object. The object gets
            created in this function.
        config: Configurations for the viewer

    Returns:
        An instantiated object of the class 'config_cls'
    """


    moderngl_window.setup_basic_logging(config_cls.log_level)
    window_cls = moderngl_window.get_local_window_cls(config.window_type)

    # Calculate window size
    size = config.size

    window = window_cls(title=config_cls.title,
                        size=size,
                        fullscreen=config.fullscreen,
                        resizable=config.resizable,
                        gl_version=config_cls.gl_version,
                        aspect_ratio=config_cls.aspect_ratio,
                        vsync=config.vsync,
                        samples=config_cls.samples,
                        cursor=config.show_cursor)
    window.print_context_info()
    moderngl_window.activate_context(window=window)
    timer = moderngl_window.Timer()
    window.config = config_cls(ctx=window.ctx, wnd=window, timer=timer)

    timer.start()
    return window


def run_viewer_loop(window):
    """Start a blocking run of our viewer.

    Args:
        window:

    """
    while not window.is_closing:
        current_time, delta = window.config.timer.next_frame()

        if window.config.clear_color is not None:
            window.clear(*window.config.clear_color)
        else:
            window.use()
        window.render(current_time, delta)
        if not window.is_closing:
            window.swap_buffers()

    window.destroy()


def create_and_run(est_pose_queue, gt_pose_queue=None,viewer_config = ViewerConfig()):
    """Creates the Viewer and starts the run loop.

    Args:
        est_pose_queue: Queue where the estimated poses are placed.
        gt_pose_queue: Optional queue where the ground truth poses are placed.

    This is a convenience function to run the Viewer. If you want to run the Viewer in a seperate process then you have
    to use this function.

    Do something like follows

    Example:

        '''
        from multiprocessing import Queue, Process
        pose_queue = Queue()
        viewer_process = Process(target=create_and_run,args=(pose_queue,))
        viewer_process.start()

        '''

    The reason we need this function is that OpenGL/modernGL do not play nice with creating and modifying objects in
    different processes. By using this function we ensure that the creation and modification all happens in the same
    process.


    """
    window = create_window(SLAMViewer,viewer_config)
    window.config.pose_queue = est_pose_queue
    window.config.gt_pose_queue = gt_pose_queue
    run_viewer_loop(window)
