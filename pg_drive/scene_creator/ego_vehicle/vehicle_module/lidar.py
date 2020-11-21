import logging
import math
import os

import numpy as np
from panda3d.bullet import BulletGhostNode, BulletSphereShape
from pg_drive.scene_creator.highway_vehicle.behavior import IDMVehicle
from pg_drive.scene_creator.pg_traffic_vehicle.traffic_vehicle import PgTrafficVehicle
from panda3d.core import Point3, BitMask32, DrawMask, Vec3, NodePath, RigidBodyCombiner
from pg_drive.utils.visualization_loader import VisLoader
from pg_drive.pg_config.body_name import BodyName
from pg_drive.pg_config.cam_mask import CamMask
from typing import Set


class Lidar:
    Lidar_point_cloud_obs_dim = 240

    def __init__(self, parent_node_np: NodePath, laser_num: int = 240, distance: float = 50, show=False):
        self.Lidar_point_cloud_obs_dim = laser_num
        self.laser_num = laser_num
        self.perceive_distance = distance
        self.radian_unit = 2 * np.pi / laser_num
        self.detection_results = None
        self.node_path = parent_node_np.attachNewNode("cloudPoints")
        self.node_path.hide(CamMask.FrontCam | CamMask.Shadow | CamMask.Shadow | CamMask.MainCam)
        self.cloud_points = [] if show else None
        if show:
            for laser_debug in range(self.laser_num):
                ball = VisLoader.loader.loadModel(os.path.join(VisLoader.path, "models/box.egg"))
                ball.setScale(0.001)
                ball.setColor(0., 0.5, 0.5, 1)
                shape = BulletSphereShape(0.1)
                ghost = BulletGhostNode('Lidar Point')
                ghost.setIntoCollideMask(BitMask32.allOff())
                ghost.addShape(shape)
                laser_np = self.node_path.attachNewNode(ghost)
                self.cloud_points.append(laser_np)
                ball.getChildren().reparentTo(laser_np)
            # self.node_path.flattenStrong()

    def perceive(self, vehicle_position, heading_theta, bt_physics_world):
        """
        Call me to update the perception info
        """
        start_position = Vec3(vehicle_position[0], -vehicle_position[1], 1.0)
        self.detection_results = []
        heading_theta -= math.pi / 2
        mask = BitMask32.bit(PgTrafficVehicle.COLLISION_MASK)
        laser_heading = np.arange(0, self.laser_num) * self.radian_unit + heading_theta
        point_x = self.perceive_distance * np.cos(laser_heading) + start_position[0]
        point_y = self.perceive_distance * np.sin(laser_heading) + start_position[1]

        for laser_index in range(self.laser_num):
            laser_end = Point3(point_x[laser_index], point_y[laser_index], 1.0)
            result = bt_physics_world.rayTestClosest(start_position, laser_end, mask)
            self.detection_results.append(result)
            if self.cloud_points is not None:
                if result.hasHit():
                    curpos = result.getHitPos()
                else:
                    curpos = laser_end
                self.cloud_points[laser_index].setPos(curpos)

    def _get_surrounding_vehicles(self) -> Set[IDMVehicle]:
        vehicles = set()
        for ret in self.detection_results:
            if ret.hasHit():
                vehicles.add(ret.getNode().getPythonTag(BodyName.Traffic_vehicle).kinematic_model)
        return vehicles

    def get_surrounding_vehicles_info(self, ego_vehicle, max_v_num: int = 4):
        from pg_drive.utils.math_utils import norm, clip
        surrounding_vehicles = list(self._get_surrounding_vehicles())
        surrounding_vehicles.sort(
            key=lambda v: norm(ego_vehicle.position[0] - v.position[0], ego_vehicle.position[1] - v.position[1])
        )
        surrounding_vehicles += [None] * max_v_num
        res = []
        for vehicle in surrounding_vehicles[:max_v_num]:
            if vehicle is not None:
                assert isinstance(vehicle, IDMVehicle), "Now PgDrive Doesn't support other vehicle type"
                relative_position = ego_vehicle.projection(vehicle.position - ego_vehicle.position)
                # It is possible that the centroid of other vehicle is too far away from ego but lidar shed on it.
                # So the distance may greater than perceive distance.
                res.append(clip((relative_position[0] / self.perceive_distance + 1) / 2, 0.0, 1.0))
                res.append(clip((relative_position[1] / self.perceive_distance + 1) / 2, 0.0, 1.0))

                relative_velocity = ego_vehicle.projection(vehicle.velocity - ego_vehicle.velocity)
                res.append(clip((relative_velocity[0] / ego_vehicle.max_speed + 1) / 2, 0.0, 1.0))
                res.append(clip((relative_velocity[1] / ego_vehicle.max_speed + 1) / 2, 0.0, 1.0))
            else:
                res += [0.0] * 4
        return res

    def get_cloud_points(self):
        return [point.getHitFraction() for point in self.detection_results]

    def __del__(self):
        logging.debug("Lidar is destroyed.")
