"""
    Contains basic objects that one can render using a basic OpenGL shader.
"""

import moderngl
import moderngl_window
import numpy as np
import pyrr
from moderngl_window.opengl.vao import VAO
from pyrr import Matrix44, Quaternion, quaternion

RED = (1.0, 0.0, 0.0)

# The default OpenGL Shader meant to be used with these simple render primitives.
# It simply computes the vertex position given the standard 3 OpenGL matrices(projection,model and camera)
# Only 1 color is set.
default_vertex_shader = """
#version 330

in vec3 in_position;

uniform mat4 m_proj;
uniform mat4 m_model;
uniform mat4 m_camera;

void main() {
	gl_Position = m_proj * m_camera * m_model * vec4(in_position, 1.0);
}
"""
default_fragment_shader = """
#version 330

out vec4 fragColor;
uniform vec3 color;

void main()
{
    fragColor = vec4(color, 1.0);
}

"""


class BasicRenderable():
    """
    A base class representing a basic object to render with the above default shader.
    """
    POSITION = "in_position"

    def __init__(self, shape_name='', mode=moderngl.POINTS, render_on_init=True):
        self.draw_mode = mode

        # moderngl version of a vertex array object. Stores our vertices so that they can be sent to OpenGL.
        self.vao = VAO(shape_name)
        # The 3 standard OpenGL matrices. See
        # http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/#the-model-view-and-projection-matrices
        # for a description of them.
        self.projection_matrix = None
        self.camera_matrix = None
        self.model_matrix = None

        # The color we will render the primitive to.
        self.color = RED

        # Flag controlling whether to render the object.
        self.can_render = render_on_init

    def set_color(self, r, g, b):
        self.color = (r, g, b)

    def set_projection(self, projection_matrix):
        self.projection_matrix = projection_matrix

    def set_model_matrix(self, model_matrix):
        self.model_matrix = model_matrix

    def set_camera_matrix(self, camera_matrix):
        self.camera_matrix = camera_matrix

    def set_matrices(self, projection, model, camera):
        self.set_projection(projection)
        self.set_model_matrix(model)
        self.set_camera_matrix(camera)

    def set_vertices(self, vertices):
        if vertices.size % 3 != 0:
            raise TypeError("Each vertex has 3 points so the numpy array must be divisable by 3")
        if vertices.dtype != np.float32:
            raise TypeError("Must be float32")
        self.vao.buffer(vertices.astype(np.float32), "3f", [BasicRenderable.POSITION])

    def render(self, shader_program: moderngl.Program, first_index=0, num_vertices=-1):
        """Render the object with the given shader.

        Args:
            shader_program: The shader program. It should be the default_shader program.
            first_index: The index of the first vertice to render. Typically you should ignore this value.
            num_vertices: How many vertices from our index should we render. Typically you should ignore this value

        This function renders the object using the passed in shader. Note it should almost always be the default_shader,
        or else the the variable names won't make sense.

        """
        if not self.can_render:
            return
        shader_program['m_model'].write(self.model_matrix)
        shader_program['m_camera'].write(self.camera_matrix)
        shader_program['m_proj'].write(self.projection_matrix)
        shader_program['color'].value = self.color
        self.vao.render(shader_program, self.draw_mode, num_vertices, first_index)


class DynamicNumpyArray():
    def __init__(self):

        self.internal_array = np.array([])
        self.internal_size = 0
        self.growth_factor = 2.5

    def set_points(self, points):
        self.internal_array = points
        self.internal_size = points.size

    def add_points(self, points):
        capacity = self.internal_array.size
        size_needed = points.size + self.internal_size
        if size_needed >= capacity:
            self.internal_size = size_needed * self.growth_factor


class PointCloudRenderable(BasicRenderable):
    def __init__(self):
        super().__init__("pointcloud", moderngl.POINTS)

        self.internal_pointcloud = np.array([])

    def set_points(self, point_cloud):
        self.internal_pointcloud = point_cloud
        self.set_vertices(self.internal_pointcloud)

    def add_points(self, points):
        self.internal_pointcloud = np.concatenate([self.internal_pointcloud, points])
        self.set_vertices(self.internal_pointcloud)


class TrajectoryRenderable(BasicRenderable):
    """Renders a continous trajectory.

    """
    def __init__(self):
        super().__init__("trajectory", moderngl.LINE_STRIP, False)

        self.internal_pointcloud = np.array([], dtype=np.float32)
        super().set_model_matrix(Matrix44.identity(dtype=np.float32))

    def set_points(self, point_cloud):
        self.internal_pointcloud = point_cloud
        self.set_vertices(self.internal_pointcloud)

    def add_poses(self, points):
        self.internal_pointcloud = np.concatenate([self.internal_pointcloud, points])
        self.vao.release()
        self.set_vertices(self.internal_pointcloud)

    def add_pose(self, pose, min_dist=0.05):
        if self.internal_pointcloud.size:
            end_pose = self.internal_pointcloud[-3:]
            if np.linalg.norm(end_pose - pose) < min_dist:
                return
        self.add_poses(pose)

    def set_model_matrix(self, model_matrix):
        """
            Since the trajectory points already are in the global frame, we don't need a model matrix to view them.

        """
        pass


class CameraShape(BasicRenderable):
    """
    Renders a trapezoidal/truncated pyramid to represent the camera.
    """
    def __init__(self, width=1.0, height_ratio=0.75, depth_ratio=0.6):
        super().__init__("camera_frustrum", moderngl.LINES)
        w = width
        h = w * height_ratio
        z = w * depth_ratio
        vertices = np.array([
            0, 0, 0, w, h, z, 0, 0, 0, w, -h, z, 0, 0, 0, -w, -h, z, 0, 0, 0, -w, h, z, w, h, z, w, -h, z, -w, h, z, -w,
            -h, z, -w, h, z, w, h, z, -w, -h, z, w, -h, z
        ],
                            dtype=np.float32)
        self.set_vertices(vertices)


class CubeRenderable(BasicRenderable):
    """
        Renders a 3D cube
    """
    def __init__(self, width, height, depth):
        super().__init__("cube", moderngl.TRIANGLES)

        width, height, depth = width / 2.0, height / 2.0, depth / 2.0
        center = (0, 0, 0)
        pos = np.array([
            center[0] + width,
            center[1] - height,
            center[2] + depth,
            center[0] + width,
            center[1] + height,
            center[2] + depth,
            center[0] - width,
            center[1] - height,
            center[2] + depth,
            center[0] + width,
            center[1] + height,
            center[2] + depth,
            center[0] - width,
            center[1] + height,
            center[2] + depth,
            center[0] - width,
            center[1] - height,
            center[2] + depth,
            center[0] + width,
            center[1] - height,
            center[2] - depth,
            center[0] + width,
            center[1] + height,
            center[2] - depth,
            center[0] + width,
            center[1] - height,
            center[2] + depth,
            center[0] + width,
            center[1] + height,
            center[2] - depth,
            center[0] + width,
            center[1] + height,
            center[2] + depth,
            center[0] + width,
            center[1] - height,
            center[2] + depth,
            center[0] + width,
            center[1] - height,
            center[2] - depth,
            center[0] + width,
            center[1] - height,
            center[2] + depth,
            center[0] - width,
            center[1] - height,
            center[2] + depth,
            center[0] + width,
            center[1] - height,
            center[2] - depth,
            center[0] - width,
            center[1] - height,
            center[2] + depth,
            center[0] - width,
            center[1] - height,
            center[2] - depth,
            center[0] - width,
            center[1] - height,
            center[2] + depth,
            center[0] - width,
            center[1] + height,
            center[2] + depth,
            center[0] - width,
            center[1] + height,
            center[2] - depth,
            center[0] - width,
            center[1] - height,
            center[2] + depth,
            center[0] - width,
            center[1] + height,
            center[2] - depth,
            center[0] - width,
            center[1] - height,
            center[2] - depth,
            center[0] + width,
            center[1] + height,
            center[2] - depth,
            center[0] + width,
            center[1] - height,
            center[2] - depth,
            center[0] - width,
            center[1] - height,
            center[2] - depth,
            center[0] + width,
            center[1] + height,
            center[2] - depth,
            center[0] - width,
            center[1] - height,
            center[2] - depth,
            center[0] - width,
            center[1] + height,
            center[2] - depth,
            center[0] + width,
            center[1] + height,
            center[2] - depth,
            center[0] - width,
            center[1] + height,
            center[2] - depth,
            center[0] + width,
            center[1] + height,
            center[2] + depth,
            center[0] - width,
            center[1] + height,
            center[2] - depth,
            center[0] - width,
            center[1] + height,
            center[2] + depth,
            center[0] + width,
            center[1] + height,
            center[2] + depth,
        ],
                       dtype=np.float32)
        self.set_vertices(pos)
