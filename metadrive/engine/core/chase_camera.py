import math
import queue
from collections import deque
from typing import Tuple

import numpy as np
from direct.controls.InputState import InputState
from panda3d.core import Vec3, Point3
from panda3d.core import WindowProperties

from metadrive.constants import CollisionGroup
from metadrive.engine.engine_utils import get_engine
from metadrive.utils.coordinates_shift import panda_heading, panda_position


class MainCamera:
    """
    Only chase vehicle now
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

    def __init__(self, engine, camera_height: float, camera_dist: float):
        self._origin_height = camera_height
        self.engine = engine

        # vehicle chase camera
        self.camera = engine.cam
        self.camera_queue = None
        self.camera_dist = camera_dist
        self.direction_running_mean = deque(maxlen=20)
        self.world_light = self.engine.world_light  # light chases the chase camera, when not using global light
        self.inputs = InputState()
        self.current_track_vehicle = None

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

        self.engine.accept("wheel_up", self._wheel_up_height)
        self.engine.accept("wheel_down", self._wheel_down_height)
        self.engine.accept("mouse1", self._move_to_pointer)

        # default top-down
        self.top_down_camera_height = self.engine.global_config["top_down_camera_initial_z"]
        self.camera_x = self.engine.global_config["top_down_camera_initial_x"]
        self.camera_y = self.engine.global_config["top_down_camera_initial_y"]
        self.camera_rotate = 0
        self.engine.interface.stop_track()
        self.engine.task_manager.add(self._top_down_task, self.TOP_DOWN_TASK_NAME, extraArgs=[], appendTask=True)

        # TPP rotate
        if not self.engine.global_config["show_mouse"]:
            props = WindowProperties()
            props.setCursorHidden(True)
            props.setMouseMode(WindowProperties.MConfined)
            self.engine.win.requestProperties(props)
        self.mouse_rotate = 0
        self.last_mouse_pos = self.engine.mouseWatcherNode.getMouseX() if self.has_mouse else 0
        self.static_timer = 0
        self.move_into_window_timer = 0
        self._in_recover = False
        self._last_frame_has_mouse = False

    def set_bird_view_pos(self, position):
        if self.engine.task_manager.hasTaskNamed(self.TOP_DOWN_TASK_NAME):
            # adjust hpr
            p_pos = panda_position(position)
            self.camera_x, self.camera_y = p_pos[0], p_pos[1]
            self.camera_rotate = 0
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
        self.update_mouse_info()
        self.chase_camera_height = self._update_height(self.chase_camera_height)
        self.camera_queue.put(vehicle.chassis.get_pos())
        if not self.FOLLOW_LANE:
            forward_dir = vehicle.system.get_forward_vector()
            current_forward_dir = forward_dir[0], forward_dir[1]
        else:
            current_forward_dir = self._dir_of_lane(vehicle.navigation.current_ref_lanes[0], vehicle.position)
        self.direction_running_mean.append(current_forward_dir)
        forward_dir = np.mean(self.direction_running_mean, axis=0)
        forward_dir[0] = np.cos(self.mouse_rotate) * current_forward_dir[0] - np.sin(self.mouse_rotate) * \
                         current_forward_dir[1]
        forward_dir[1] = np.sin(self.mouse_rotate) * current_forward_dir[0] + np.cos(self.mouse_rotate) * \
                         current_forward_dir[1]

        camera_pos = list(self.camera_queue.get())
        camera_pos[2] += self.chase_camera_height + vehicle.HEIGHT / 2
        camera_pos[0] += -forward_dir[0] * self.camera_dist
        camera_pos[1] += -forward_dir[1] * self.camera_dist

        self.camera.setPos(*camera_pos)
        current_pos = vehicle.chassis.getPos()
        current_pos[2] += 2
        self.camera.lookAt(current_pos)
        if self.FOLLOW_LANE:
            self.camera.setH(
                self._heading_of_lane(vehicle.navigation.current_ref_lanes[0], vehicle.position) / np.pi * 180 - 90
            )

        if self.world_light is not None:
            self.world_light.step(current_pos)
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
        :param lane: Abstractlane
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
        self.current_track_vehicle = vehicle
        self.engine.interface.track(vehicle)
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
        self.current_track_vehicle = None

    def stop_track(self, bird_view_on_current_position=True):
        self.engine.interface.stop_track()
        if self.engine.task_manager.hasTaskNamed(self.CHASE_TASK_NAME):
            self.engine.task_manager.remove(self.CHASE_TASK_NAME)
        if not self.engine.task_manager.hasTaskNamed(self.TOP_DOWN_TASK_NAME):
            # adjust hpr
            if bird_view_on_current_position:
                current_pos = self.camera.getPos()
                self.camera_x, self.camera_y = current_pos[0], current_pos[1]
                self.camera_rotate = 0
            self.engine.task_manager.add(self._top_down_task, self.TOP_DOWN_TASK_NAME, extraArgs=[], appendTask=True)

    def _top_down_task(self, task):
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
        self.camera.lookAt(self.camera_x, self.camera_y, 0)

        if self.inputs.isSet("right_rotate"):
            self.camera_rotate += 3
        if self.inputs.isSet("left_rotate"):
            self.camera_rotate -= 3
        self.camera.setH(self.camera_rotate)
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
            pMouse = self.engine.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            self.camera.node().getLens().extrude(pMouse, pFrom, pTo)

            # Transform to global coordinates
            pFrom = self.engine.render.getRelativePoint(self.camera, pFrom)
            pTo = self.engine.render.getRelativePoint(self.camera, pTo)
            ret = self.engine.physics_world.dynamic_world.rayTestClosest(pFrom, pTo, CollisionGroup.Terrain)
            self.camera_x = ret.getHitPos()[0]
            self.camera_y = ret.getHitPos()[1]

    def is_bird_view_camera(self):
        return True if self.engine.task_manager.hasTaskNamed(self.TOP_DOWN_TASK_NAME) else False

    @property
    def has_mouse(self):
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
