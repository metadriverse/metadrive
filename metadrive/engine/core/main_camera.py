import math
import queue
from collections import deque
from typing import Tuple, Union

import numpy as np
from direct.controls.InputState import InputState
from panda3d.core import Vec3, Point3, PNMImage, NodePath
from panda3d.core import WindowProperties

from metadrive.constants import CollisionGroup, CameraTagStateKey, Semantics
from metadrive.engine.engine_utils import get_engine
from metadrive.utils.coordinates_shift import panda_heading, panda_vector
from metadrive.utils.cuda import check_cudart_err

_cuda_enable = True
try:
    import cupy as cp
    from OpenGL.GL import GL_TEXTURE_2D  # noqa F403
    from cuda import cudart
    from cuda.cudart import cudaGraphicsRegisterFlags
    from panda3d.core import GraphicsOutput, Texture, GraphicsStateGuardianBase, DisplayRegionDrawCallbackData
except ImportError:
    _cuda_enable = False
from metadrive.component.sensors.base_sensor import BaseSensor
from metadrive import constants


class MainCamera(BaseSensor):
    """
    It is a third-person perspective camera for chasing the vehicle. The view in this camera will be rendered into the
    main image buffer (main window). It is also a sensor, so perceive() can be called to access the rendered image.
    """

    queue_length = 3
    CHASE_TASK_NAME = "update main chase camera"
    TOP_DOWN_TASK_NAME = "update main bird camera"
    FOLLOW_LANE = False
    TOP_DOWN_VIEW_HEIGHT = 120
    WHEEL_SCROLL_SPEED = 10
    MOUSE_RECOVER_TIME = 8
    STATIC_MOUSE_HOLD_TIME = 100  # in steps
    MOUSE_MOVE_INTO_LATENCY = 2
    MOUSE_SPEED_MULTIPLIER = 1

    num_channels = 3

    def __init__(self, engine, camera_height: float, camera_dist: float):
        self._origin_height = camera_height
        self.run_task = True
        # self.engine = engine

        # vehicle chase camera
        self.camera = engine.camera
        # lens property
        lens = engine.cam.node().getLens()
        lens.setFov(engine.global_config["camera_fov"])

        from metadrive.engine.core.terrain import Terrain
        engine.cam.node().setTagStateKey(CameraTagStateKey.RGB)
        engine.cam.node().setTagState(
            Semantics.TERRAIN.label, Terrain.make_render_state(engine, "terrain.vert.glsl", "terrain.frag.glsl")
        )

        self.camera_queue = None
        self.camera_dist = camera_dist
        self.camera_pitch = -engine.global_config["camera_pitch"] if engine.global_config["camera_pitch"
                                                                                          ] is not None else None
        self.camera_smooth = engine.global_config["camera_smooth"]
        self.camera_smooth_buffer_size = engine.global_config["camera_smooth_buffer_size"]
        self.direction_running_mean = deque(maxlen=self.camera_smooth_buffer_size if self.camera_smooth else 1)
        self.world_light = engine.world_light  # light chases the chase camera, when not using global light
        self.inputs = InputState()
        self.current_track_agent = None

        # height control
        self.chase_camera_height = camera_height
        self.inputs.watchWithModifiers('high', '+')
        self.inputs.watchWithModifiers('high', '=')
        self.inputs.watchWithModifiers('low', '-')
        self.inputs.watchWithModifiers('low', '_')

        # free bird view camera
        self.top_down_camera_height = self.TOP_DOWN_VIEW_HEIGHT
        self.camera_x = 0
        self.camera_y = 0
        self.inputs.watchWithModifiers('up', 'w')
        self.inputs.watchWithModifiers('down', 's')
        self.inputs.watchWithModifiers('left', 'a')
        self.inputs.watchWithModifiers('right', 'd')
        self.inputs.watchWithModifiers('left_rotate', '[')
        self.inputs.watchWithModifiers('right_rotate', ']')
        self.engine = engine

        engine.accept("wheel_up", self._wheel_up_height)
        engine.accept("wheel_down", self._wheel_down_height)
        engine.accept("mouse1", self._move_to_pointer)

        # default top-down
        self.top_down_camera_height = engine.global_config["top_down_camera_initial_z"]
        self.camera_x = engine.global_config["top_down_camera_initial_x"]
        self.camera_y = engine.global_config["top_down_camera_initial_y"]
        self.camera_hpr = [0, 0, 0]
        engine.interface.undisplay()
        engine.task_manager.add(self._top_down_task, self.TOP_DOWN_TASK_NAME, extraArgs=[], appendTask=True)

        # TPP rotate
        if not engine.global_config["show_mouse"]:
            props = WindowProperties()
            props.setCursorHidden(True)
            props.setMouseMode(WindowProperties.MConfined)
            engine.win.requestProperties(props)
        self.mouse_rotate = 0
        self.last_mouse_pos = engine.mouseWatcherNode.getMouseX() if self.has_mouse else 0
        self.static_timer = 0
        self.move_into_window_timer = 0
        self._in_recover = False
        self._last_frame_has_mouse = False

        need_cuda = "main_camera" in engine.global_config["sensors"]
        self.enable_cuda = engine.global_config["image_on_cuda"] and need_cuda

        self.cuda_graphics_resource = None
        if "main_camera" in engine.global_config["sensors"]:
            self.engine.sensors["main_camera"] = self
        if self.enable_cuda:
            assert _cuda_enable, "Can not enable cuda rendering pipeline, if you are on Windows, try 'pip install pypiwin32'"

            # returned tensor property
            self.cuda_dtype = np.uint8
            self.cuda_shape = engine.global_config["window_size"]
            self.cuda_strides = None
            self.cuda_order = "C"

            self._cuda_buffer = None

            # make texture
            self.cuda_texture = Texture()
            engine.win.addRenderTexture(self.cuda_texture, GraphicsOutput.RTMCopyTexture)

            def _callback_func(cbdata: DisplayRegionDrawCallbackData):
                # print("DRAW CALLBACK!!!!!!!!!!!!!!!11")
                cbdata.upcall()
                if not self.registered and self.texture_context_future.done():
                    self.register()
                if self.registered:
                    with self as array:
                        self.cuda_rendered_result = array

            # Fill the buffer due to multi-thread (3 stage)
            for _ in range(3):
                engine.graphicsEngine.renderFrame()
            engine.cam.node().getDisplayRegion(0).setDrawCallback(_callback_func)

            self.gsg = GraphicsStateGuardianBase.getDefaultGsg()
            self.texture_context_future = self.cuda_texture.prepare(self.gsg.prepared_objects)
            self.cuda_texture_identifier = None
            self.new_cuda_mem_ptr = None
            self.cuda_rendered_result = None

    def set_bird_view_pos(self, position):
        """
        Set the x,y position for the main camera
        Args:
            position:

        Returns:

        """
        self.set_bird_view_pos_hpr(position)

    def set_bird_view_pos_hpr(self, position, hpr=None):
        """
        Set the x,y position and heading, pitch, roll  for the main camera
        Args:
            position:
            hpr:

        Returns:

        """
        if self.engine.task_manager.hasTaskNamed(self.TOP_DOWN_TASK_NAME):
            # adjust hpr
            p_pos = panda_vector(position, self.engine.global_config["top_down_camera_initial_z"])
            self.camera_x, self.camera_y, self.top_down_camera_height = p_pos[0], p_pos[1], p_pos[2]
            self.camera_hpr = hpr or [0, 0, 0]
            self.engine.task_manager.add(self._top_down_task, self.TOP_DOWN_TASK_NAME, extraArgs=[], appendTask=True)

    def reset(self):
        self.direction_running_mean.clear()

    def update_mouse_info(self):
        self.move_into_window_timer -= 1 if self.move_into_window_timer > 0 else 0
        if self.mouse_into_window:
            self._in_recover = True
            self.set_mouse_to_center()
            self.move_into_window_timer = self.MOUSE_MOVE_INTO_LATENCY

        if not self._in_recover and self.has_mouse and not self.mouse_into_window and self.move_into_window_timer == 0:
            new_mouse_pos = self.engine.mouseWatcherNode.getMouseX()
            last_rotate = self.mouse_rotate
            self.mouse_rotate = -new_mouse_pos * self.MOUSE_SPEED_MULTIPLIER
            diff = abs(last_rotate - self.mouse_rotate)
            if diff == 0.:
                self.static_timer += 1
            else:
                self.static_timer = 0
            if self.static_timer > self.STATIC_MOUSE_HOLD_TIME:
                self._in_recover = True
                self.set_mouse_to_center()
        else:
            self.mouse_rotate += -self.mouse_rotate / self.MOUSE_RECOVER_TIME

        if self._in_recover and abs(self.mouse_rotate) < 0.01:
            self._in_recover = False
            self.static_timer = 0
            self.last_mouse_pos = 0

        self._last_frame_has_mouse = self.has_mouse

    def _chase_task(self, vehicle, task):
        if not self.run_task:
            return task.cont
        self.update_mouse_info()
        self.chase_camera_height = self._update_height(self.chase_camera_height)
        chassis_pos = vehicle.chassis.get_pos()
        self.camera_queue.put(chassis_pos)
        if not self.FOLLOW_LANE:
            forward_dir = vehicle.system.get_forward_vector()
            # camera is facing to y
            current_forward_dir = [forward_dir[0], forward_dir[1]]
        else:
            current_forward_dir = self._dir_of_lane(vehicle.navigation.current_ref_lanes[0], vehicle.position)
        self.direction_running_mean.append(current_forward_dir)
        forward_dir = np.mean(self.direction_running_mean, axis=0) if self.camera_smooth else current_forward_dir
        forward_dir[0] = np.cos(self.mouse_rotate) * current_forward_dir[0] - np.sin(self.mouse_rotate) * \
                         current_forward_dir[1]
        forward_dir[1] = np.sin(self.mouse_rotate) * current_forward_dir[0] + np.cos(self.mouse_rotate) * \
                         current_forward_dir[1]

        # don't put this line to if-else, strange bug happened
        camera_pos = list(self.camera_queue.get())
        if not self.camera_smooth:
            camera_pos = chassis_pos
        camera_pos[2] += self.chase_camera_height + vehicle.HEIGHT / 2
        camera_pos[0] += -forward_dir[0] * self.camera_dist
        camera_pos[1] += -forward_dir[1] * self.camera_dist

        self.camera.setPos(*camera_pos)
        if self.engine.global_config["show_coordinates"]:
            self.engine.set_coordinates_indicator_pos([chassis_pos[0], chassis_pos[1]])
        current_pos = vehicle.chassis.getPos()
        current_pos[2] += 2

        if self.camera_pitch is None:
            self.camera.lookAt(current_pos)
            # camera is facing to y
            self.camera.setH(vehicle.origin.getH() + np.rad2deg(self.mouse_rotate))
        else:
            # camera is facing to y
            self.camera.setH(vehicle.origin.getH())
            self.camera.setP(self.camera_pitch)
        if self.FOLLOW_LANE:
            self.camera.setH(
                self._heading_of_lane(vehicle.navigation.current_ref_lanes[0], vehicle.position) / np.pi * 180 - 90
            )

        # Don't use reparentTo()
        # pos = vehicle.convert_to_world_coordinates([0.8, 0.], vehicle.position)
        # look_at = vehicle.convert_to_world_coordinates([10.4, 0.], vehicle.position)
        # self.camera.setPos(pos[0], pos[1], 1.5 + vehicle.origin.getZ())
        # self.camera.lookAt(look_at[0], look_at[1], 1.6 + vehicle.origin.getZ())
        return task.cont

    @staticmethod
    def _heading_of_lane(lane, pos: Tuple) -> float:
        """
        Calculate the heading of a position on lane
        :param lane: Abstract lane
        :param pos: Tuple, MetaDrive coordinates
        :return: heading theta
        """
        heading_theta = panda_heading(lane.heading_theta_at(lane.local_coordinates(pos)[0]))
        return heading_theta

    @staticmethod
    def _dir_of_lane(lane, pos: Tuple) -> Tuple:
        """
        Get direction of lane
        :param lane: AbstractLane
        :param pos: metadrive position, tuple
        :return: dir, tuple
        """
        heading = MainCamera._heading_of_lane(lane, pos)
        return math.cos(heading), math.sin(heading)

    def track(self, vehicle):
        """
        Use this function to chase a new vehicle !
        :param vehicle: Vehicle to chase
        :return: None
        """
        self.current_track_agent = vehicle
        self.engine.interface.display()
        pos = None
        if self.FOLLOW_LANE:
            pos = self._pos_on_lane(vehicle)  # Return None if routing system is not ready
        if pos is None:
            pos = vehicle.position

        if self.engine.task_manager.hasTaskNamed(self.CHASE_TASK_NAME):
            self.engine.task_manager.remove(self.CHASE_TASK_NAME)
        if self.engine.task_manager.hasTaskNamed(self.TOP_DOWN_TASK_NAME):
            self.engine.task_manager.remove(self.TOP_DOWN_TASK_NAME)
        self.mouse_rotate = 0
        self.last_mouse_pos = self.engine.mouseWatcherNode.getMouseX() if self.has_mouse else 0
        self.static_timer = 0
        self.set_mouse_to_center()
        self.engine.task_manager.add(self._chase_task, self.CHASE_TASK_NAME, extraArgs=[vehicle], appendTask=True)
        self.camera_queue = queue.Queue(self.queue_length)
        for i in range(self.queue_length - 1):
            self.camera_queue.put(Vec3(pos[0], -pos[1], 0))

    @staticmethod
    def _pos_on_lane(vehicle) -> Tuple:
        """
        Recalculate cam place
        :param vehicle: BaseVehicle
        :return: position on the center lane
        """
        if vehicle.navigation.current_ref_lanes is None:
            raise ValueError("No routing module, I don't know which lane to follow")

        lane = vehicle.navigation.current_ref_lanes[0]
        lane_num = len(vehicle.navigation.current_ref_lanes)

        longitude, _ = lane.local_coordinates(vehicle.position)
        lateral = lane_num * lane.width / 2 - lane.width / 2
        return longitude, lateral

    def set_follow_lane(self, flag: bool):
        """
        Camera will go follow reference lane instead of vehicle
        :return: None
        """
        self.FOLLOW_LANE = flag

    def destroy(self):
        engine = get_engine()
        if engine.task_manager.hasTaskNamed(self.CHASE_TASK_NAME):
            engine.task_manager.remove(self.CHASE_TASK_NAME)
        if engine.task_manager.hasTaskNamed(self.TOP_DOWN_TASK_NAME):
            engine.task_manager.remove(self.TOP_DOWN_TASK_NAME)
        self.current_track_agent = None
        if self.registered:
            self.unregister()
            # self.camera.node().getDisplayRegion(0).clearDrawCallback()

    def stop_track(self, bird_view_on_current_position=True):
        self.engine.interface.undisplay()
        if self.engine.task_manager.hasTaskNamed(self.CHASE_TASK_NAME):
            self.engine.task_manager.remove(self.CHASE_TASK_NAME)
        if not self.engine.task_manager.hasTaskNamed(self.TOP_DOWN_TASK_NAME):
            # adjust hpr
            if bird_view_on_current_position:
                current_pos = self.camera.getPos()
                self.camera_x, self.camera_y = current_pos[0], current_pos[1]
                self.camera_hpr = [0, 0, 0]
            self.engine.task_manager.add(self._top_down_task, self.TOP_DOWN_TASK_NAME, extraArgs=[], appendTask=True)

    def _top_down_task(self, task):
        if not self.run_task:
            return task.cont
        self.top_down_camera_height = self._update_height(self.top_down_camera_height)

        if self.inputs.isSet("up"):
            self.camera_y += 1.0
        if self.inputs.isSet("down"):
            self.camera_y -= 1.0
        if self.inputs.isSet("left"):
            self.camera_x -= 1.0
        if self.inputs.isSet("right"):
            self.camera_x += 1.0

        self.camera.setPos(self.camera_x, self.camera_y, self.top_down_camera_height)
        if self.engine.global_config["show_coordinates"]:
            self.engine.set_coordinates_indicator_pos([self.camera_x, self.camera_y])
        if abs(sum(self.camera_hpr)) < 0.0001:
            self.camera.lookAt(self.camera_x, self.camera_y, 0)
        else:
            self.camera.setHpr(Vec3(*self.camera_hpr))
        return task.cont

    def _update_height(self, height):
        if self.inputs.isSet("high"):
            height += 1.0
        if self.inputs.isSet("low"):
            height -= 1.0
        return height

    def _wheel_down_height(self):
        if self.engine.task_manager.hasTaskNamed(self.TOP_DOWN_TASK_NAME):
            self.top_down_camera_height += self.WHEEL_SCROLL_SPEED
        else:
            self.chase_camera_height += self.WHEEL_SCROLL_SPEED

    def _wheel_up_height(self):
        if self.engine.task_manager.hasTaskNamed(self.TOP_DOWN_TASK_NAME):
            self.top_down_camera_height -= self.WHEEL_SCROLL_SPEED
        else:
            self.chase_camera_height -= self.WHEEL_SCROLL_SPEED

    def _move_to_pointer(self):
        if self.engine.task_manager.hasTaskNamed(self.TOP_DOWN_TASK_NAME):
            # Get to and from pos in camera coordinates
            try:
                pMouse = self.engine.mouseWatcherNode.getMouse()
            except:
                return
            pFrom = Point3()
            pTo = Point3()
            self.engine.cam.node().getLens().extrude(pMouse, pFrom, pTo)

            # Transform to global coordinates
            pFrom = self.engine.render.getRelativePoint(self.engine.cam, pFrom)
            pTo = self.engine.render.getRelativePoint(self.engine.cam, pTo)
            ret = self.engine.physics_world.dynamic_world.rayTestClosest(pFrom, pTo, CollisionGroup.Terrain)
            self.camera_x = ret.getHitPos()[0]
            self.camera_y = ret.getHitPos()[1]

    def is_bird_view_camera(self):
        return True if self.engine.task_manager.hasTaskNamed(self.TOP_DOWN_TASK_NAME) else False

    @property
    def has_mouse(self):
        if self.engine.mouseWatcherNode is None:
            return False
        return True if self.engine.mouseWatcherNode.hasMouse() else False

    def set_mouse_to_center(self):
        mouse_id = 0
        if self.has_mouse:
            win_middle_x = self.engine.win.getXSize() / 2
            win_middle_y = self.engine.win.getYSize() / 2
            self.engine.win.movePointer(mouse_id, int(win_middle_x), int(win_middle_y))

    @property
    def mouse_into_window(self):
        return True if not self._last_frame_has_mouse and self.has_mouse else False

    def perceive(
        self, to_float=True, new_parent_node: Union[NodePath, None] = None, position=None, hpr=None
    ) -> np.ndarray:
        """
        When to_float is set to False, the image will be represented by unit8 with component value ranging from [0-255].
        Otherwise, it will be float type with component value ranging from [0.-1.]. By default, the reset parameters are
        all None. In this case, the camera will render the result with poses and position set by track() function.

        When the reset parameters are not None, this camera will be mounted to a new place and render corresponding new
        results. After this, the camera will be returned to the original states. This process is like borrowing the
        camera to capture a new image and return the camera to the owner. This usually happens when using one camera to
        render multiple times from different positions and poses.

        new_parent_node should be a NodePath like object.origin or vehicle.origin or self.engine.origin, which
        means the world origin. When new_parent_node is set, both position and hpr have to be set as well. The position
        and hpr are all 3-dim vector representing:
            1) the relative position to the reparent node
            2) the heading/pitch/roll of the sensor

        Args:
            to_float: When to_float is set to False, the image will be represented by unit8 with component value ranging
                from [0-255]. Otherwise, it will be float type with component value ranging from [0.-1.].
            new_parent_node: new_parent_node should be a NodePath like object.origin or vehicle.origin or
                self.engine.origin, which means the world origin. When new_parent_node is set, both position and hpr
                have to be set as well. The position and hpr are all 3-dim vector representing:
            position: the relative position to the reparent node
            hpr: the heading/pitch/roll of the sensor

        Return:
            Array representing the image.
        """

        if new_parent_node:
            if position is None:
                position = constants.DEFAULT_SENSOR_OFFSET
            if hpr is None:
                position = constants.DEFAULT_SENSOR_HPR

            # return camera to original state
            original_object = self.camera.getParent()
            original_hpr = self.camera.getHpr()
            original_position = self.camera.getPos()

            # reparent to new parent node
            self.camera.reparentTo(new_parent_node)
            # relative position
            assert len(position) == 3, "The first parameter of camera.perceive() should be a BaseObject instance " \
                                       "or a 3-dim vector representing the (x,y,z) position."
            self.camera.setPos(Vec3(*position))
            assert len(hpr) == 3, "The hpr parameter of camera.perceive() should be  a 3-dim vector representing " \
                                  "the heading/pitch/roll."
            self.camera.setHpr(Vec3(*hpr))
            self.run_task = False
            self.engine.taskMgr.step()
            self.run_task = True

        engine = get_engine()
        if self.enable_cuda:
            assert self.cuda_rendered_result is not None
            img = self.cuda_rendered_result[..., :-1][..., ::-1][::-1]
        else:
            origin_img = engine.win.getDisplayRegion(1).getScreenshot()
            img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)
            img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), 4))
            img = img[::-1]
            img = img[..., :self.num_channels]

        # restore
        if new_parent_node:
            self.camera.reparentTo(original_object)
            self.camera.setHpr(original_hpr)
            self.camera.setPos(original_position)

        if not to_float:
            return img.astype(np.uint8, copy=False, order="C")
        else:
            return img / 255

    def __del__(self):
        if self.enable_cuda:
            self.unregister()

    """
    Following functions are cuda support
    """

    @property
    def cuda_array(self):
        assert self.mapped
        return cp.ndarray(
            shape=(self.cuda_shape[1], self.cuda_shape[0], 4),
            dtype=self.cuda_dtype,
            strides=self.cuda_strides,
            order=self.cuda_order,
            memptr=self._cuda_buffer
        )

    @property
    def cuda_buffer(self):
        assert self.mapped
        return self._cuda_buffer

    @property
    def graphics_resource(self):
        assert self.registered
        return self.cuda_graphics_resource

    @property
    def registered(self):
        return self.cuda_graphics_resource is not None

    @property
    def mapped(self):
        return self._cuda_buffer is not None

    def __enter__(self):
        return self.map()

    def __exit__(self, exc_type, exc_value, trace):
        self.unmap()
        return False

    def register(self):
        self.cuda_texture_identifier = self.texture_context_future.result().getNativeId()
        assert self.cuda_texture_identifier is not None
        if self.registered:
            return self.cuda_graphics_resource
        self.cuda_graphics_resource = check_cudart_err(
            cudart.cudaGraphicsGLRegisterImage(
                self.cuda_texture_identifier, GL_TEXTURE_2D, cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
            )
        )
        return self.cuda_graphics_resource

    def unregister(self):
        if self.registered:
            self.unmap()
            self.cuda_graphics_resource = check_cudart_err(
                cudart.cudaGraphicsUnregisterResource(self.cuda_graphics_resource)
            )

    def map(self, stream=0):
        if not self.registered:
            raise RuntimeError("Cannot map an unregistered buffer.")
        if self.mapped:
            return self._cuda_buffer
        check_cudart_err(cudart.cudaGraphicsMapResources(1, self.cuda_graphics_resource, stream))
        array = check_cudart_err(cudart.cudaGraphicsSubResourceGetMappedArray(self.graphics_resource, 0, 0))
        channelformat, cudaextent, flag = check_cudart_err(cudart.cudaArrayGetInfo(array))

        depth = 1
        byte = 4  # four channel
        if self.new_cuda_mem_ptr is None:
            success, self.new_cuda_mem_ptr = cudart.cudaMalloc(cudaextent.height * cudaextent.width * byte * depth)
        check_cudart_err(
            cudart.cudaMemcpy2DFromArray(
                self.new_cuda_mem_ptr, cudaextent.width * byte * depth, array, 0, 0, cudaextent.width * byte * depth,
                cudaextent.height, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
            )
        )
        if self._cuda_buffer is None:
            self._cuda_buffer = cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(self.new_cuda_mem_ptr, cudaextent.width * depth * byte * cudaextent.height, self),
                0
            )
        return self.cuda_array

    def unmap(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot unmap an unregistered buffer.")
        if not self.mapped:
            return self
        self._cuda_buffer = check_cudart_err(cudart.cudaGraphicsUnmapResources(1, self.cuda_graphics_resource, stream))
        return self

    # def get_image(self):
    #     # The Tracked obj arg is only for compatibility
    #     img = PNMImage()
    #     self.engine.win.getScreenshot(img)
    #     return img

    def save_image(self, tracked_obj, file_name="main_camera.png", **kwargs):
        # The Tracked obj arg is only for compatibility
        # img = self.get_image()
        img = PNMImage()
        self.engine.win.getScreenshot(img)
        img.write(file_name)
