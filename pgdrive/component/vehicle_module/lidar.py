from typing import Set

from panda3d.core import BitMask32, NodePath
from pgdrive.constants import BodyName, CamMask, CollisionGroup
from pgdrive.component.vehicle.traffic_vehicle import TrafficVehicle
from pgdrive.component.vehicle_module.distance_detector import DistanceDetector


class Lidar(DistanceDetector):
    ANGLE_FACTOR = True
    Lidar_point_cloud_obs_dim = 240
    DEFAULT_HEIGHT = 0.5

    def __init__(self, parent_node_np: NodePath, num_lasers: int = 240, distance: float = 50, enable_show=False):
        super(Lidar, self).__init__(parent_node_np, num_lasers, distance, enable_show)
        self.node_path.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam)
        self.mask = BitMask32.bit(TrafficVehicle.COLLISION_MASK) | BitMask32.bit(
            CollisionGroup.EgoVehicle
        ) | BitMask32.bit(CollisionGroup.InvisibleWall)

    def get_surrounding_vehicles(self) -> Set:
        vehicles = set()
        objs = self.get_detected_objects()
        for ret in objs:
            if ret.getNode().hasPythonTag(BodyName.Traffic_vehicle):
                vehicles.add(ret.getNode().getPythonTag(BodyName.Traffic_vehicle).kinematic_model)
            elif ret.getNode().hasPythonTag(BodyName.Base_vehicle):
                vehicles.add(ret.getNode().getPythonTag(BodyName.Base_vehicle))
        return vehicles

    # def _get_surrounding_objects(self) -> Set[Object]:
    #     """
    #     TODO may be static objects info should be added in obs, now this func is useless
    #     :return: a set of objects
    #     """
    #     objects = set()
    #     for ret in self.detection_results:
    #         if ret.hasHit() and ret.getNode().getName() in [BodyName.Traffic_cone, BodyName.Traffic_triangle]:
    #             objects.add(ret.getNode().getPythonTag(BodyName.Traffic_vehicle).kinematic_model)
    #     return objects

    def get_surrounding_vehicles_info(self, ego_vehicle, num_others: int = 4):
        from pgdrive.utils.math_utils import norm, clip
        surrounding_vehicles = list(self.get_surrounding_vehicles())
        surrounding_vehicles.sort(
            key=lambda v: norm(ego_vehicle.position[0] - v.position[0], ego_vehicle.position[1] - v.position[1])
        )
        surrounding_vehicles += [None] * num_others
        res = []
        for vehicle in surrounding_vehicles[:num_others]:
            if vehicle is not None:
                # assert isinstance(vehicle, IDMVehicle or Base), "Now PGDrive Doesn't support other vehicle type"
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
