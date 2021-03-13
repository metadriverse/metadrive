import queue
from typing import Tuple

import numpy as np
from direct.controls.InputState import InputState
from panda3d.core import Vec3, Camera
from pgdrive.scene_creator.ego_vehicle.base_vehicle import BaseVehicle
from pgdrive.utils.coordinates_shift import panda_heading
from pgdrive.world.pg_world import PGWorld


class ChaseCamera:
    """
    Only chase vehicle now
    """

    queue_length = 3
    TASK_NAME = "update main camera"
    FOLLOW_LANE = False

    def __init__(
        self, camera: Camera, vehicle: BaseVehicle, camera_height: float, camera_dist: float, pg_world: PGWorld
    ):
        self.camera = camera
        self.camera_queue = None
        self.camera_height = camera_height
        self.camera_dist = camera_dist
        self.light = pg_world.light  # light position is updated with the chase camera when control vehicle
        self.inputs = InputState()
        self.inputs.watchWithModifiers('up', 'k')
        self.inputs.watchWithModifiers('down', 'j')
        self.reset(vehicle, pg_world)

    def renew_camera_place(self, vehicle, task):
        if self.inputs.isSet("up"):
            self.camera_height += 0.5
        if self.inputs.isSet("down"):
            self.camera_height -= 0.5
        self.camera_queue.put(vehicle.chassis_np.get_pos())
        if not self.FOLLOW_LANE:
            forward_dir = vehicle.system.get_forward_vector()
        else:
            forward_dir = self._dir_of_lane(vehicle.routing_localization.current_ref_lanes[0], vehicle.position)

        camera_pos = list(self.camera_queue.get())
        camera_pos[2] += self.camera_height
        camera_pos[0] += -forward_dir[0] * self.camera_dist
        camera_pos[1] += -forward_dir[1] * self.camera_dist

        self.camera.setPos(*camera_pos)
        current_pos = vehicle.chassis_np.getPos()
        current_pos[2] += 2
        if not self.FOLLOW_LANE:
            self.camera.lookAt(current_pos)
        else:
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
        return np.cos(heading), np.sin(heading)

    def reset(self, vehicle, pg_world):
        """
        Use this function to chase a new vehicle !
        :param vehicle: Vehicle to chase
        :param pg_world: pg_world class
        :return: None
        """

        pos = self._pos_on_lane(vehicle) if self.FOLLOW_LANE else vehicle.position

        if pg_world.taskMgr.hasTaskNamed(self.TASK_NAME):
            pg_world.taskMgr.remove(self.TASK_NAME)
        pg_world.taskMgr.add(self.renew_camera_place, self.TASK_NAME, extraArgs=[vehicle], appendTask=True)
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
        lane = vehicle.routing_localization.current_ref_lanes[0]
        lane_num = len(vehicle.routing_localization.current_ref_lanes)

        longitude, _ = lane.local_coordinates(vehicle.position)
        lateral = lane_num * lane.width / 2 - lane.width / 2
        return longitude, lateral

    def set_follow_lane(self, follow: bool = True):
        """
        Camera will go follow reference lane instead of vehicle
        :return: None
        """
        self.FOLLOW_LANE = follow

    def destroy(self, pg_world):
        if pg_world.taskMgr.hasTaskNamed(self.TASK_NAME):
            pg_world.taskMgr.remove(self.TASK_NAME)
