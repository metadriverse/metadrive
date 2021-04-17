import queue
from collections import deque
from typing import Tuple

import numpy as np
from direct.controls.InputState import InputState
from panda3d.core import Vec3, Camera, Point3, BitMask32
from pgdrive.constants import CollisionGroup
from pgdrive.utils.coordinates_shift import panda_heading
from pgdrive.world.pg_world import PGWorld


class ChaseCamera:
    """
    Only chase vehicle now
    """

    queue_length = 3
    CHASE_TASK_NAME = "update main chase camera"
    BIRD_TASK_NAME = "update main bird camera"
    FOLLOW_LANE = True
    BIRD_VIEW_HEIGHT = 120
    WHEEL_SCROLL_SPEED = 10

    def __init__(self, camera: Camera, camera_height: float, camera_dist: float, pg_world: PGWorld):
        self._origin_height = camera_height

        # vehicle chase camera
        self.camera = camera
        self.camera_queue = None
        self.camera_dist = camera_dist
        self.direction_running_mean = deque(maxlen=20)
        self.light = pg_world.light  # light position is updated with the chase camera when control vehicle
        self.inputs = InputState()

        # height control
        self.chase_camera_height = camera_height
        self.inputs.watchWithModifiers('high', '+')
        self.inputs.watchWithModifiers('high', '=')
        self.inputs.watchWithModifiers('low', '-')
        self.inputs.watchWithModifiers('low', '_')

        # free bird view camera
        pg_world.accept("b", self.stop_chase, extraArgs=[pg_world])
        self.bird_camera_height = self.BIRD_VIEW_HEIGHT
        self.camera_x = 0
        self.camera_y = 0
        self.inputs.watchWithModifiers('up', 'w')
        self.inputs.watchWithModifiers('down', 's')
        self.inputs.watchWithModifiers('left', 'a')
        self.inputs.watchWithModifiers('right', 'd')

        pg_world.accept("wheel_up", self._wheel_up_height, extraArgs=[pg_world])
        pg_world.accept("wheel_down", self._wheel_down_height, extraArgs=[pg_world])
        pg_world.accept("mouse1", self._move_to_pointer, extraArgs=[pg_world])

    def reset(self):
        self.direction_running_mean.clear()

    def renew_camera_place(self, vehicle, task):
        self.chase_camera_height = self._update_height(self.chase_camera_height)
        self.camera_queue.put(vehicle.chassis_np.get_pos())
        if not self.FOLLOW_LANE:
            forward_dir = vehicle.system.get_forward_vector()
        else:
            forward_dir = self._dir_of_lane(vehicle.routing_localization.current_ref_lanes[0], vehicle.position)

        self.direction_running_mean.append(forward_dir)
        forward_dir = np.mean(self.direction_running_mean, axis=0)

        camera_pos = list(self.camera_queue.get())
        camera_pos[2] += self.chase_camera_height
        camera_pos[0] += -forward_dir[0] * self.camera_dist
        camera_pos[1] += -forward_dir[1] * self.camera_dist

        self.camera.setPos(*camera_pos)
        current_pos = vehicle.chassis_np.getPos()
        current_pos[2] += 2
        self.camera.lookAt(current_pos)
        if self.FOLLOW_LANE:
            self.camera.setH(
                self._heading_of_lane(vehicle.routing_localization.current_ref_lanes[0], vehicle.position) / np.pi *
                180 - 90
            )

        if self.light is not None:
            self.light.step(current_pos)
        return task.cont

    @staticmethod
    def _heading_of_lane(lane, pos: Tuple) -> float:
        """
        Calculate the heading of a position on lane
        :param lane: Abstract lane
        :param pos: Tuple, PGDrive coordinates
        :return: heading theta
        """
        heading_theta = panda_heading(lane.heading_at(lane.local_coordinates(pos)[0]))
        return heading_theta

    @staticmethod
    def _dir_of_lane(lane, pos: Tuple) -> Tuple:
        """
        Get direction of lane
        :param lane: Abstractlane
        :param pos: pgdrive position, tuple
        :return: dir, tuple
        """
        heading = ChaseCamera._heading_of_lane(lane, pos)
        return math.cos(heading), math.sin(heading)

    def chase(self, vehicle, pg_world):
        """
        Use this function to chase a new vehicle !
        :param vehicle: Vehicle to chase
        :param pg_world: pg_world class
        :return: None
        """
        pos = None
        if self.FOLLOW_LANE:
            pos = self._pos_on_lane(vehicle)  # Return None if routing system is not ready
        if pos is None:
            pos = vehicle.position

        if pg_world.taskMgr.hasTaskNamed(self.CHASE_TASK_NAME):
            pg_world.taskMgr.remove(self.CHASE_TASK_NAME)
        if pg_world.taskMgr.hasTaskNamed(self.BIRD_TASK_NAME):
            pg_world.taskMgr.remove(self.BIRD_TASK_NAME)
        pg_world.taskMgr.add(self.renew_camera_place, self.CHASE_TASK_NAME, extraArgs=[vehicle], appendTask=True)
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
        if vehicle.routing_localization.current_ref_lanes is None:
            raise ValueError("No routing module, I don't know which lane to follow")

        lane = vehicle.routing_localization.current_ref_lanes[0]
        lane_num = len(vehicle.routing_localization.current_ref_lanes)

        longitude, _ = lane.local_coordinates(vehicle.position)
        lateral = lane_num * lane.width / 2 - lane.width / 2
        return longitude, lateral

    def set_follow_lane(self, flag: bool):
        """
        Camera will go follow reference lane instead of vehicle
        :return: None
        """
        self.FOLLOW_LANE = flag

    def destroy(self, pg_world):
        if pg_world.taskMgr.hasTaskNamed(self.CHASE_TASK_NAME):
            pg_world.taskMgr.remove(self.CHASE_TASK_NAME)
        if pg_world.taskMgr.hasTaskNamed(self.BIRD_TASK_NAME):
            pg_world.taskMgr.remove(self.BIRD_TASK_NAME)

    def stop_chase(self, pg_world: PGWorld):
        if pg_world.taskMgr.hasTaskNamed(self.CHASE_TASK_NAME):
            pg_world.taskMgr.remove(self.CHASE_TASK_NAME)
        if not pg_world.taskMgr.hasTaskNamed(self.BIRD_TASK_NAME):
            # adjust hpr
            current_pos = self.camera.getPos()
            self.camera.lookAt(current_pos[0], current_pos[1], 0)
            pg_world.taskMgr.add(self.manual_control_camera, self.BIRD_TASK_NAME, extraArgs=[], appendTask=True)

    def manual_control_camera(self, task):
        self.bird_camera_height = self._update_height(self.bird_camera_height)

        if self.inputs.isSet("up"):
            self.camera_y += 1.0
        if self.inputs.isSet("down"):
            self.camera_y -= 1.0
        if self.inputs.isSet("left"):
            self.camera_x -= 1.0
        if self.inputs.isSet("right"):
            self.camera_x += 1.0
        self.camera.setPos(self.camera_x, self.camera_y, self.bird_camera_height)
        return task.cont

    def _update_height(self, height):
        if self.inputs.isSet("high"):
            height += 1.0
        if self.inputs.isSet("low"):
            height -= 1.0
        return height

    def _wheel_down_height(self, pg_world):
        if pg_world.taskMgr.hasTaskNamed(self.BIRD_TASK_NAME):
            self.bird_camera_height += self.WHEEL_SCROLL_SPEED
        else:
            self.chase_camera_height += self.WHEEL_SCROLL_SPEED

    def _wheel_up_height(self, pg_world):
        if pg_world.taskMgr.hasTaskNamed(self.BIRD_TASK_NAME):
            self.bird_camera_height -= self.WHEEL_SCROLL_SPEED
        else:
            self.chase_camera_height -= self.WHEEL_SCROLL_SPEED

    def _move_to_pointer(self, pg_world):
        if pg_world.taskMgr.hasTaskNamed(self.BIRD_TASK_NAME):
            # Get to and from pos in camera coordinates
            pMouse = pg_world.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            self.camera.node().getLens().extrude(pMouse, pFrom, pTo)

            # Transform to global coordinates
            pFrom = pg_world.render.getRelativePoint(self.camera, pFrom)
            pTo = pg_world.render.getRelativePoint(self.camera, pTo)
            ret = pg_world.physics_world.dynamic_world.rayTestClosest(pFrom, pTo, BitMask32.bit(CollisionGroup.Terrain))
            self.camera_x = ret.getHitPos()[0]
            self.camera_y = ret.getHitPos()[1]
