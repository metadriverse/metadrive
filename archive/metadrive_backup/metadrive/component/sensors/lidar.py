import math
from typing import Set

import numpy as np
from panda3d.bullet import BulletGhostNode, BulletCylinderShape
from panda3d.core import NodePath

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.sensors.distance_detector import DistanceDetector
from metadrive.constants import CamMask, CollisionGroup
from metadrive.utils.coordinates_shift import panda_vector
from metadrive.utils.math import norm, clip
from metadrive.utils.utils import get_object_from_node


class Lidar(DistanceDetector):
    ANGLE_FACTOR = True
    DEFAULT_HEIGHT = 1.2

    BROAD_PHASE_EXTRA_DIST = 0

    _disable_detector_mask = False

    def __init__(self, engine):
        super(Lidar, self).__init__(engine)
        self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam | CamMask.SemanticCam)
        self.mask = CollisionGroup.can_be_lidar_detected()
        self.enable_mask = True if not Lidar._disable_detector_mask else False

        # lidar can calculate the detector mask by itself
        self.broad_detectors = {}

    def get_broad_phase_detector(self, radius):
        radius = int(radius)
        if radius in self.broad_detectors:
            broad_detector = self.broad_detectors[radius]
        else:
            broad_phase_distance = int(radius)
            broad_detector = NodePath(BulletGhostNode("detector_mask"))
            broad_detector.node().addShape(BulletCylinderShape(self.BROAD_PHASE_EXTRA_DIST + broad_phase_distance, 5))
            broad_detector.node().setIntoCollideMask(CollisionGroup.LidarBroadDetector)
            broad_detector.node().setStatic(True)
            self._node_path_list.append(broad_detector)
            self.engine.physics_world.static_world.attach(broad_detector.node())
            self.broad_detectors[broad_phase_distance] = broad_detector
        return broad_detector

    def perceive(
        self,
        base_vehicle,
        physics_world,
        num_lasers,
        distance,
        height=None,
        detector_mask: np.ndarray = None,
        show=False
    ):
        res = self._get_lidar_mask(base_vehicle, num_lasers, distance)
        if self.enable_mask:
            lidar_mask = detector_mask or res[0]
        else:
            lidar_mask = None
        detected_objects = res[1]
        return super(Lidar, self).perceive(
            base_vehicle,
            physics_world,
            distance=distance,
            height=height,
            num_lasers=num_lasers,
            detector_mask=lidar_mask,
            show=show
        )[0], detected_objects

    @staticmethod
    def get_surrounding_vehicles(detected_objects) -> Set:
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        vehicles = set()
        objs = detected_objects
        for ret in objs:
            if isinstance(ret, BaseVehicle):
                vehicles.add(ret)
        return vehicles

    def _project_to_vehicle_system(self, target, vehicle, perceive_distance):
        diff = target - vehicle.position
        norm_distance = norm(diff[0], diff[1])
        if norm_distance > perceive_distance:
            diff = diff / norm_distance * perceive_distance
        relative = vehicle.convert_to_local_coordinates(diff, 0.0)
        return relative

    def get_surrounding_vehicles_info(
        self, ego_vehicle, detected_objects, perceive_distance, num_others, add_others_navi
    ):
        surrounding_vehicles = list(self.get_surrounding_vehicles(detected_objects))
        surrounding_vehicles.sort(
            key=lambda v: norm(ego_vehicle.position[0] - v.position[0], ego_vehicle.position[1] - v.position[1])
        )
        surrounding_vehicles += [None] * num_others
        res = []

        for vehicle in surrounding_vehicles[:num_others]:
            if vehicle is not None:
                ego_position = ego_vehicle.position

                # assert isinstance(vehicle, IDMVehicle or Base), "Now MetaDrive Doesn't support other vehicle type"
                relative_position = ego_vehicle.convert_to_local_coordinates(vehicle.position, ego_position)
                # It is possible that the centroid of other vehicle is too far away from ego but lidar shed on it.
                # So the distance may greater than perceive distance.
                res.append(clip((relative_position[0] / perceive_distance + 1) / 2, 0.0, 1.0))
                res.append(clip((relative_position[1] / perceive_distance + 1) / 2, 0.0, 1.0))

                relative_velocity = ego_vehicle.convert_to_local_coordinates(
                    vehicle.velocity_km_h, ego_vehicle.velocity_km_h
                )
                res.append(clip((relative_velocity[0] / ego_vehicle.max_speed_km_h + 1) / 2, 0.0, 1.0))
                res.append(clip((relative_velocity[1] / ego_vehicle.max_speed_km_h + 1) / 2, 0.0, 1.0))

                if add_others_navi:
                    ckpt1, ckpt2 = vehicle.navigation.get_checkpoints()

                    relative_ckpt1 = self._project_to_vehicle_system(ckpt1, ego_vehicle, perceive_distance)
                    res.append(clip((relative_ckpt1[0] / perceive_distance + 1) / 2, 0.0, 1.0))
                    res.append(clip((relative_ckpt1[1] / perceive_distance + 1) / 2, 0.0, 1.0))

                    relative_ckpt2 = self._project_to_vehicle_system(ckpt2, ego_vehicle, perceive_distance)
                    res.append(clip((relative_ckpt2[0] / perceive_distance + 1) / 2, 0.0, 1.0))
                    res.append(clip((relative_ckpt2[1] / perceive_distance + 1) / 2, 0.0, 1.0))

            else:

                if add_others_navi:
                    res += [0.0] * 8
                else:
                    res += [0.0] * 4

        return res

    def _get_lidar_mask(self, vehicle, num_lasers, radius):
        pos1 = vehicle.position
        head1 = vehicle.heading_theta

        mask = np.zeros((num_lasers, ), dtype=bool)
        mask.fill(False)
        objs = self.get_surrounding_objects(vehicle, int(radius))
        for obj in objs:
            pos2 = obj.position
            length = obj.LENGTH if hasattr(obj, "LENGTH") else vehicle.LENGTH
            width = obj.WIDTH if hasattr(obj, "WIDTH") else vehicle.WIDTH
            half_max_span_square = ((length + width) / 2)**2
            diff = (pos2[0] - pos1[0], pos2[1] - pos1[1])
            dist_square = diff[0]**2 + diff[1]**2
            if dist_square < half_max_span_square:
                mask.fill(True)
                continue

            span = math.asin(math.sqrt(half_max_span_square / dist_square))
            # relative heading of v2's center when compared to v1's center
            relative_head = math.atan2(diff[1], diff[0])
            head_in_1 = relative_head - head1
            head_in_1_max = head_in_1 + span
            head_in_1_min = head_in_1 - span
            head_1_max = np.rad2deg(head_in_1_max)
            head_1_min = np.rad2deg(head_in_1_min)
            mask = self._mark_this_range(head_1_min, head_1_max, mask, num_lasers)

        return mask, objs

    def get_surrounding_objects(self, vehicle, radius=50):
        broad_detector = self.get_broad_phase_detector(int(radius))
        broad_detector.setPos(panda_vector(vehicle.position))
        physics_world = vehicle.engine.physics_world.dynamic_world
        contact_results = physics_world.contactTest(broad_detector.node(), True).getContacts()
        objs = set()
        for contact in contact_results:
            node0 = contact.getNode0()
            node1 = contact.getNode1()
            nodes = [node0, node1]
            nodes.remove(broad_detector.node())
            obj = get_object_from_node(nodes[0])
            if not isinstance(obj, AbstractLane) and obj is not None:
                objs.add(obj)
        if vehicle in objs:
            objs.remove(vehicle)
        return objs

    def _mark_this_range(self, small_angle, large_angle, mask, num_lasers):
        # We use clockwise to determine small and large angle.
        # For example, if you wish to fill 355 deg to 5 deg, then small_angle is 355, large_angle is 5.
        small_angle = small_angle % 360
        large_angle = large_angle % 360

        assert 0 <= small_angle <= 360
        assert 0 <= large_angle <= 360

        angle_delta = 360 / num_lasers if num_lasers > 0 else None

        small_index = math.floor(small_angle / angle_delta)
        large_index = math.ceil(large_angle / angle_delta)
        if large_angle < small_angle:  # We are in the case like small=355, large=5
            mask[small_index:] = True
            mask[:large_index + 1] = True
        else:
            mask[small_index:large_index + 1] = True
        return mask

    def destroy(self):
        for detector in self.broad_detectors.values():
            self.engine.physics_world.static_world.remove(detector.node())
            detector.removeNode()
        super(Lidar, self).destroy()
