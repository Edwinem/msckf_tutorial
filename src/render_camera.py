import time
from math import cos, radians, sin

import numpy as np
import pyrr
from moderngl_window import WindowConfig
from moderngl_window.context.base import BaseKeys
from moderngl_window.scene.camera import *
from pyrr import Matrix44, Quaternion, quaternion


class CameraHandler(Camera):
    """Implements a camera handler to view the 3D scene.

    This class implements the necessary functionality to have a moveable camera view the 3d scene. It can be moved
    using the WASD arrow keys, or set to follow/orbit a particular position.

    """
    def __init__(self, keys: BaseKeys, fov=60.0, aspect_ratio=1.0, near=1.0, far=100.0):
        """Initialize the camera

        Args:
            keys (BaseKeys): The key constants for the current window type
        Keyword Args:
            fov (float): Field of view
            aspect_ratio (float): Aspect ratio
            near (float): near plane
            far (float): far plane
        """
        # Position movement states
        self.keys = keys
        self._xdir = STILL
        self._zdir = STILL
        self._ydir = STILL
        self._last_time = 0
        self._last_rot_time = 0
        self.follow = False

        # Orbit follow params
        self.radius = 5.0  # radius in base units
        self.angle_x, self.angle_y = 45.0, -45.0  # angles in degrees
        self.target = (0, 0, 0)  # camera target in base units
        self.up = (0.0, 1.0, 0.0)  # camera up vector

        # Velocity in axis units per second
        self._velocity = 10.0
        self._mouse_sensitivity = 0.5

        super().__init__(fov=fov, aspect_ratio=aspect_ratio, near=near, far=far)

    @property
    def mouse_sensitivity(self) -> float:
        """float: Mouse sensitivity (rotation speed).

        This property can also be set::

            camera.mouse_sensitivity = 2.5
        """
        return self._mouse_sensitivity

    @mouse_sensitivity.setter
    def mouse_sensitivity(self, value: float):
        self._mouse_sensitivity = value

    @property
    def velocity(self):
        """float: The speed this camera move based on key inputs

        The property can also be modified::

            camera.velocity = 5.0
        """
        return self._velocity

    @velocity.setter
    def velocity(self, value: float):
        self._velocity = value

    def key_input(self, key, action, modifiers) -> None:
        """Process key inputs and move camera

        Args:
            key: The key
            action: key action release/press
            modifiers: key modifier states such as ctrl or shit
        """
        # Right
        if key == self.keys.D:
            if action == self.keys.ACTION_PRESS:
                self.move_right(True)
            elif action == self.keys.ACTION_RELEASE:
                self.move_right(False)
        # Left
        elif key == self.keys.A:
            if action == self.keys.ACTION_PRESS:
                self.move_left(True)
            elif action == self.keys.ACTION_RELEASE:
                self.move_left(False)
        # Forward
        elif key == self.keys.W:
            if action == self.keys.ACTION_PRESS:
                self.move_forward(True)
            if action == self.keys.ACTION_RELEASE:
                self.move_forward(False)
        # Backwards
        elif key == self.keys.S:
            if action == self.keys.ACTION_PRESS:
                self.move_backward(True)
            if action == self.keys.ACTION_RELEASE:
                self.move_backward(False)

        # UP
        elif key == self.keys.Q:
            if action == self.keys.ACTION_PRESS:
                self.move_down(True)
            if action == self.keys.ACTION_RELEASE:
                self.move_down(False)

        # Down
        elif key == self.keys.E:
            if action == self.keys.ACTION_PRESS:
                self.move_up(True)
            if action == self.keys.ACTION_RELEASE:
                self.move_up(False)

        elif key == self.keys.F:
            if action == self.keys.ACTION_PRESS:
                self.follow = not self.follow

    def move_left(self, activate) -> None:
        """The camera should be continiously moving to the left.

        Args:
            activate (bool): Activate or deactivate this state
        """
        self.move_state(LEFT, activate)

    def move_right(self, activate) -> None:
        """The camera should be continiously moving to the right.

        Args:
            activate (bool): Activate or deactivate this state
        """
        self.move_state(RIGHT, activate)

    def move_forward(self, activate) -> None:
        """The camera should be continiously moving forward.

        Args:
            activate (bool): Activate or deactivate this state
        """
        self.move_state(FORWARD, activate)

    def move_backward(self, activate) -> None:
        """The camera should be continiously moving backwards.

        Args:
            activate (bool): Activate or deactivate this state
        """
        self.move_state(BACKWARD, activate)

    def move_up(self, activate) -> None:
        """The camera should be continiously moving up.

        Args:
            activate (bool): Activate or deactivate this state
        """
        self.move_state(UP, activate)

    def move_down(self, activate):
        """The camera should be continiously moving down.

        Args:
            activate (bool): Activate or deactivate this state
        """
        self.move_state(DOWN, activate)

    def move_state(self, direction, activate) -> None:
        """Set the camera position move state.

        Args:
            direction: What direction to update
            activate: Start or stop moving in the direction
        """
        if direction == RIGHT:
            self._xdir = POSITIVE if activate else STILL
        elif direction == LEFT:
            self._xdir = NEGATIVE if activate else STILL
        elif direction == FORWARD:
            self._zdir = NEGATIVE if activate else STILL
        elif direction == BACKWARD:
            self._zdir = POSITIVE if activate else STILL
        elif direction == UP:
            self._ydir = POSITIVE if activate else STILL
        elif direction == DOWN:
            self._ydir = NEGATIVE if activate else STILL

    def rot_state(self, dx: int, dy: int) -> None:
        """Update the rotation of the camera.

        This is done by passing in the relative
        mouse movement change on x and y (delta x, delta y).

        In the past this method took the viewport position
        of the mouse. This does not work well when
        mouse exclusivity mode is enabled.

        Args:
            dx: Relative mouse position change on x
            dy: Relative mouse position change on y
        """
        now = time.time()
        delta = now - self._last_rot_time
        self._last_rot_time = now

        # Greatly decrease the chance of camera popping.
        # This can happen when the mouse enters and leaves the window
        # or when getting focus again.
        if delta > 0.1 and max(abs(dx), abs(dy)) > 2:
            return

        dx *= self._mouse_sensitivity
        dy *= self._mouse_sensitivity

        self._yaw -= dx
        self._pitch += dy

        if self.pitch > 85.0:
            self.pitch = 85.0
        if self.pitch < -85.0:
            self.pitch = -85.0

        self._update_yaw_and_pitch()

    @property
    def matrix(self) -> np.ndarray:
        """numpy.ndarray: The current view matrix for the camera"""
        if self.follow:
            return Matrix44.look_at(
                (
                    cos(radians(self.angle_x)) * sin(radians(self.angle_y)) * self.radius + self.target[0],
                    cos(radians(self.angle_y)) * self.radius + self.target[1],
                    sin(radians(self.angle_x)) * sin(radians(self.angle_y)) * self.radius + self.target[2],
                ),  # camera (eye) position, calculated from angles and radius
                self.target,  # what to look at
                self.up,  # camera up direction (change for rolling the camera)
                dtype="f4",
            )
        else:
            now = time.time()
            # If the camera has been inactive for a while, a large time delta
            # can suddenly move the camera far away from the scene
            t = max(now - self._last_time, 0)
            self._last_time = now

            # X Movement
            if self._xdir == POSITIVE:
                self.position += self.right * self._velocity * t
            elif self._xdir == NEGATIVE:
                self.position -= self.right * self._velocity * t

            # Z Movement
            if self._zdir == NEGATIVE:
                self.position += self.dir * self._velocity * t
            elif self._zdir == POSITIVE:
                self.position -= self.dir * self._velocity * t

            # Y Movement
            if self._ydir == POSITIVE:
                self.position += self.up * self._velocity * t
            elif self._ydir == NEGATIVE:
                self.position -= self.up * self._velocity * t
            return self._gl_look_at(self.position, self.position + self.dir, self._up)

    @property
    def angle_x(self) -> float:
        """float: camera angle x in degrees.

        This property can also be set::
            camera.angle_x = 45.
        """
        return self._angle_x

    @angle_x.setter
    def angle_x(self, value: float):
        """Set camera rotation_x in degrees."""
        self._angle_x = value

    @property
    def angle_y(self) -> float:
        """float: camera angle y in degrees.

        This property can also be set::
            camera.angle_y = 45.
        """
        return self._angle_y

    @angle_y.setter
    def angle_y(self, value: float):
        """Set camera rotation_y in degrees."""
        self._angle_y = value

    @property
    def zoom_sensitivity(self) -> float:
        """float: Mousewheel zooming sensitivity (zoom speed).

        This property can also be set::
            camera.zoom_sensitivity = 2.5
        """
        return self._mouse_sensitivity

    @zoom_sensitivity.setter
    def zoom_sensitivity(self, value: float):
        self._zoom_sensitivity = value

    def rot_state(self, dx: float, dy: float) -> None:
        """Update the rotation of the camera around the target point.

        This is done by passing relative mouse change in the x and y axis (delta x, delta y)

        Args:
            dx: Relative mouse position change on x axis
            dy: Relative mouse position change on y axis
        """
        self.angle_x += dx * self.mouse_sensitivity / 10.0
        self.angle_y += dy * self.mouse_sensitivity / 10.0

        # clamp the y angle to avoid weird rotations
        self.angle_y = max(min(self.angle_y, -5.0), -175.0)

    def zoom_state(self, y_offset: float) -> None:
        # allow zooming in/out
        self.radius -= y_offset * self._zoom_sensitivity
        self.radius = max(1.0, self.radius)


class CameraWindow(WindowConfig):
    """A base class to use our custom Camera Handler.

    A moderngl.window config class that utilizes our 'CameraHandler' class.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = CameraHandler(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio)
        # Good starting position for Euroc dataset
        self.camera.set_position(0,0,12)
        self.camera_enabled = True

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()

    # def mouse_position_event(self, x: int, y: int, dx, dy):
    #     if self.camera_enabled:
    #         self.camera.rot_state(-dx, -dy)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)
