import logging
import math
from collections import defaultdict

import numpy as np
from panda3d.bullet import BulletGhostNode, BulletSphereShape, BulletAllHitsRayResult
from panda3d.core import BitMask32, NodePath

from pgdrive.constants import CamMask, CollisionGroup
from pgdrive.utils.asset_loader import AssetLoader
from pgdrive.utils.coordinates_shift import panda_position


class DetectorMask:
    def __init__(self, num_lasers: int, max_span: float, max_distance: float = 1e6):
        self.num_lasers = num_lasers
        self.angle_delta = 360 / self.num_lasers
        # self.max_span = max_span
        self.half_max_span_square = (max_span / 2)**2
        self.masks = defaultdict(lambda: np.zeros((self.num_lasers, ), dtype=np.bool))
        # self.max_distance = max_distance + max_span
        self.max_distance_square = (max_distance + max_span)**2

    def update_mask(self, position_dict: dict, heading_dict: dict, is_target_vehicle_dict: dict):
        assert set(position_dict.keys()) == set(heading_dict.keys()) == set(is_target_vehicle_dict.keys())
        if not position_dict:
            return

        for k in self.masks.keys():
            self.masks[k].fill(False)

        for k, is_target in is_target_vehicle_dict.items():
            if is_target:
                self.masks[k]  # Touch to create entry

        keys = list(position_dict.keys())
        for c1, k1 in enumerate(keys[:-1]):
            for c2, k2 in enumerate(keys[c1 + 1:]):

                if (not is_target_vehicle_dict[k1]) and (not is_target_vehicle_dict[k2]):
                    continue

                pos1 = position_dict[k1]
                pos2 = position_dict[k2]
                head1 = heading_dict[k1]
                head2 = heading_dict[k2]

                diff = (pos2[0] - pos1[0], pos2[1] - pos1[1])
                dist_square = diff[0]**2 + diff[1]**2
                if dist_square < self.half_max_span_square:
                    if is_target_vehicle_dict[k1]:
                        self._mark_all(k1)
                    if is_target_vehicle_dict[k2]:
                        self._mark_all(k2)
                    continue

                if dist_square > self.max_distance_square:
                    continue

                span = None
                if is_target_vehicle_dict[k1]:
                    span = math.asin(math.sqrt(self.half_max_span_square / dist_square))
                    # relative heading of v2's center when compared to v1's center
                    relative_head = math.atan2(diff[1], diff[0])
                    head_in_1 = relative_head - head1
                    head_in_1_max = head_in_1 + span
                    head_in_1_min = head_in_1 - span
                    head_1_max = np.rad2deg(head_in_1_max)
                    head_1_min = np.rad2deg(head_in_1_min)
                    self._mark_this_range(head_1_min, head_1_max, name=k1)

                if is_target_vehicle_dict[k2]:
                    if span is None:
                        span = math.asin(math.sqrt(self.half_max_span_square / dist_square))
                    diff2 = (-diff[0], -diff[1])
                    # relative heading of v2's center when compared to v1's center
                    relative_head2 = math.atan2(diff2[1], diff2[0])
                    head_in_2 = relative_head2 - head2
                    head_in_2_max = head_in_2 + span
                    head_in_2_min = head_in_2 - span
                    head_2_max = np.rad2deg(head_in_2_max)
                    head_2_min = np.rad2deg(head_in_2_min)
                    self._mark_this_range(head_2_min, head_2_max, name=k2)

    def _mark_this_range(self, small_angle, large_angle, name):
        # We use clockwise to determine small and large angle.
        # For example, if you wish to fill 355 deg to 5 deg, then small_angle is 355, large_angle is 5.
        small_angle = small_angle % 360
        large_angle = large_angle % 360

        assert 0 <= small_angle <= 360
        assert 0 <= large_angle <= 360

        small_index = math.floor(small_angle / self.angle_delta)
        large_index = math.ceil(large_angle / self.angle_delta)
        if large_angle < small_angle:  # We are in the case like small=355, large=5
            self.masks[name][small_index:] = True
            self.masks[name][:large_index + 1] = True
        else:
            self.masks[name][small_index:large_index + 1] = True

    def _mark_all(self, name):
        self.masks[name].fill(True)

    def get_mask(self, name):
        assert name in self.masks, "It seems that you have not initialized the mask for vehicle {} yet!".format(name)
        return self.masks[name]

    def clear(self):
        self.masks.clear()

    def get_mask_ratio(self):
        total = 0
        masked = 0
        for k, v in self.masks.items():
            total += v.size
            masked += v.sum()
        return masked / total


class DistanceDetector:
    """
    It is a module like lidar, used to detect sidewalk/center line or other static things
    """
    Lidar_point_cloud_obs_dim = 240
    DEFAULT_HEIGHT = 0.2

    # for vis debug
    MARK_COLOR = (51 / 255, 221 / 255, 1)
    ANGLE_FACTOR = False

    def __init__(self, parent_node_np: NodePath, num_lasers: int = 16, distance: float = 50, enable_show=False):
        # properties
        assert num_lasers > 0
        show = enable_show and (AssetLoader.loader is not None)
        self.dim = num_lasers
        self.num_lasers = num_lasers
        self.perceive_distance = distance
        self.height = self.DEFAULT_HEIGHT
        self.radian_unit = 2 * np.pi / num_lasers
        self.start_phase_offset = 0
        self.node_path = parent_node_np.attachNewNode("Could_points")
        self._lidar_range = np.arange(0, self.num_lasers) * self.radian_unit + self.start_phase_offset

        # detection result
        self.cloud_points = np.ones((self.num_lasers, ), dtype=float)
        self.detected_objects = []

        # override these properties to decide which elements to detect and show
        self.node_path.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam)
        self.mask = BitMask32.bit(CollisionGroup.BrokenLaneLine)
        self.cloud_points_vis = [] if show else None
        logging.debug("Load Vehicle Module: {}".format(self.__class__.__name__))
        if show:
            for laser_debug in range(self.num_lasers):
                ball = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                ball.setScale(0.001)
                ball.setColor(0., 0.5, 0.5, 1)
                shape = BulletSphereShape(0.1)
                ghost = BulletGhostNode('Lidar Point')
                ghost.setIntoCollideMask(BitMask32.allOff())
                ghost.addShape(shape)
                laser_np = self.node_path.attachNewNode(ghost)
                self.cloud_points_vis.append(laser_np)
                ball.getChildren().reparentTo(laser_np)
            # self.node_path.flattenStrong()

    def perceive(
        self,
        vehicle_position,
        heading_theta,
        pg_physics_world,
        extra_filter_node: set = None,
        detector_mask: np.ndarray = None
    ):
        """
        Call me to update the perception info
        """
        assert detector_mask is not "WRONG"
        # coordinates problem here! take care
        extra_filter_node = extra_filter_node or set()
        pg_start_position = panda_position(vehicle_position, self.height)

        # init
        self.cloud_points.fill(1.0)
        self.detected_objects = []

        # lidar calculation use pg coordinates
        mask = self.mask
        # laser_heading = self._lidar_range + heading_theta
        # point_x = self.perceive_distance * np.cos(laser_heading) + vehicle_position[0]
        # point_y = self.perceive_distance * np.sin(laser_heading) + vehicle_position[1]

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        for laser_index in range(self.num_lasers):
            # # coordinates problem here! take care

            if (detector_mask is not None) and (not detector_mask[laser_index]):
                # update vis
                if self.cloud_points_vis is not None:
                    laser_end = self._get_laser_end(laser_index, heading_theta, vehicle_position)
                    self._add_cloud_point_vis(laser_index, laser_end)
                continue

            laser_end = self._get_laser_end(laser_index, heading_theta, vehicle_position)
            result = pg_physics_world.rayTestClosest(pg_start_position, laser_end, mask)
            node = result.getNode()
            if node in extra_filter_node:
                # Fall back to all tests.
                results: BulletAllHitsRayResult = pg_physics_world.rayTestAll(pg_start_position, laser_end, mask)
                hits = results.getHits()
                hits = sorted(hits, key=lambda ret: ret.getHitFraction())
                for result in hits:
                    if result.getNode() in extra_filter_node:
                        continue
                    self.detected_objects.append(result)
                    self.cloud_points[laser_index] = result.getHitFraction()
                    break
            else:
                hits = result.hasHit()
                self.cloud_points[laser_index] = result.getHitFraction()
                if node:
                    self.detected_objects.append(result)

            # update vis
            if self.cloud_points_vis is not None:
                self._add_cloud_point_vis(laser_index, result.getHitPos() if hits else laser_end)

    def _add_cloud_point_vis(self, laser_index, pos):
        self.cloud_points_vis[laser_index].setPos(pos)
        f = laser_index / self.num_lasers if self.ANGLE_FACTOR else 1
        self.cloud_points_vis[laser_index].setColor(
            f * self.MARK_COLOR[0], f * self.MARK_COLOR[1], f * self.MARK_COLOR[2]
        )

    def _get_laser_end(self, laser_index, heading_theta, vehicle_position):
        point_x = self.perceive_distance * math.cos(self._lidar_range[laser_index] + heading_theta) + \
                  vehicle_position[0]
        point_y = self.perceive_distance * math.sin(self._lidar_range[laser_index] + heading_theta) + \
                  vehicle_position[1]
        laser_end = panda_position((point_x, point_y), self.height)
        return laser_end

    def get_cloud_points(self):
        return self.cloud_points.tolist()

    def get_detected_objects(self):
        return self.detected_objects

    def destroy(self):
        if self.cloud_points_vis:
            for vis_laser in self.cloud_points_vis:
                vis_laser.removeNode()
        self.node_path.removeNode()
        self.cloud_points = None
        self.detected_objects = None

    def set_start_phase_offset(self, angle: float):
        """
        Change the start phase of lidar lasers
        :param angle: phasse offset in [degree]
        :return: None
        """
        self.start_phase_offset = np.deg2rad(angle)
        self._lidar_range = np.arange(0, self.num_lasers) * self.radian_unit + self.start_phase_offset

    def __del__(self):
        logging.debug("Lidar is destroyed.")


class SideDetector(DistanceDetector):
    def __init__(self, parent_node_np: NodePath, num_lasers: int = 2, distance: float = 50, enable_show=False):
        super(SideDetector, self).__init__(parent_node_np, num_lasers, distance, enable_show)
        self.set_start_phase_offset(90)
        self.node_path.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam)
        self.mask = BitMask32.bit(CollisionGroup.ContinuousLaneLine)


class LaneLineDetector(SideDetector):
    MARK_COLOR = (1, 77 / 255, 77 / 255)

    def __init__(self, parent_node_np: NodePath, num_lasers: int = 2, distance: float = 50, enable_show=False):
        super(SideDetector, self).__init__(parent_node_np, num_lasers, distance, enable_show)
        self.set_start_phase_offset(90)
        self.node_path.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam)
        self.mask = BitMask32.bit(CollisionGroup.ContinuousLaneLine) | BitMask32.bit(CollisionGroup.BrokenLaneLine)
