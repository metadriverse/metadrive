import logging
import math
from collections import namedtuple

import numpy as np
from panda3d.bullet import BulletGhostNode, BulletSphereShape
from panda3d.core import NodePath

from metadrive.constants import CamMask, CollisionGroup
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import get_engine
from metadrive.utils.coordinates_shift import panda_position
from metadrive.utils.math_utils import panda_position, get_laser_end

detect_result = namedtuple("detect_result", "cloud_points detected_objects")


def add_cloud_point_vis(
    point_x, point_y, height, num_lasers, laser_index, ANGLE_FACTOR, MARK_COLOR0, MARK_COLOR1, MARK_COLOR2
):
    f = laser_index / num_lasers if ANGLE_FACTOR else 1
    return laser_index, (point_x, point_y, height), (f * MARK_COLOR0, f * MARK_COLOR1, f * MARK_COLOR2)


def perceive(
    cloud_points, detector_mask, mask, lidar_range, perceive_distance, heading_theta, vehicle_position_x,
    vehicle_position_y, num_lasers, height, physics_world, extra_filter_node, require_colors, ANGLE_FACTOR, MARK_COLOR0,
    MARK_COLOR1, MARK_COLOR2
):
    cloud_points.fill(1.0)
    detected_objects = []
    colors = []
    pg_start_position = panda_position(vehicle_position_x, vehicle_position_y, height)

    for laser_index in range(num_lasers):
        if (detector_mask is not None) and (not detector_mask[laser_index]):
            # update vis
            if require_colors:
                point_x, point_y = get_laser_end(
                    lidar_range, perceive_distance, laser_index, heading_theta, vehicle_position_x, vehicle_position_y
                )
                point_x, point_y, point_z = panda_position(point_x, point_y, height)
                colors.append(
                    add_cloud_point_vis(
                        point_x, point_y, height, num_lasers, laser_index, ANGLE_FACTOR, MARK_COLOR0, MARK_COLOR1,
                        MARK_COLOR2
                    )
                )
            continue

        # # coordinates problem here! take care
        point_x, point_y = get_laser_end(
            lidar_range, perceive_distance, laser_index, heading_theta, vehicle_position_x, vehicle_position_y
        )
        laser_end = panda_position(point_x, point_y, height)
        result = physics_world.rayTestClosest(pg_start_position, laser_end, mask)
        node = result.getNode()
        if node in extra_filter_node:
            # Fall back to all tests.
            results = physics_world.rayTestAll(pg_start_position, laser_end, mask)
            hits = results.getHits()
            hits = sorted(hits, key=lambda ret: ret.getHitFraction())
            for result in hits:
                if result.getNode() in extra_filter_node:
                    continue
                detected_objects.append(result)
                cloud_points[laser_index] = result.getHitFraction()
                laser_end = result.getHitPos()
                break
        else:
            cloud_points[laser_index] = result.getHitFraction()
            if result.hasHit():
                laser_end = result.getHitPos()
            if node:
                detected_objects.append(result)
        if require_colors:
            colors.append(
                add_cloud_point_vis(
                    laser_end[0], laser_end[1], height, num_lasers, laser_index, ANGLE_FACTOR, MARK_COLOR0, MARK_COLOR1,
                    MARK_COLOR2
                )
            )
    return cloud_points, detected_objects, colors


class DistanceDetector:
    """
    It is a module like lidar, used to detect sidewalk/center line or other static things
    """
    Lidar_point_cloud_obs_dim = 240
    DEFAULT_HEIGHT = 0.2

    # for vis debug
    MARK_COLOR = (51 / 255, 221 / 255, 1)
    ANGLE_FACTOR = False

    def __init__(self, num_lasers: int = 16, distance: float = 50, enable_show=True):
        # properties
        self.available = True if num_lasers > 0 and distance > 0 else False
        parent_node_np: NodePath = get_engine().render
        self.origin = parent_node_np.attachNewNode("Could_points")
        show = enable_show and (AssetLoader.loader is not None)
        self.dim = num_lasers
        self.num_lasers = num_lasers
        self.perceive_distance = distance
        self.height = self.DEFAULT_HEIGHT
        self.radian_unit = 2 * np.pi / num_lasers if self.num_lasers > 0 else None
        self.start_phase_offset = 0
        self._lidar_range = np.arange(0, self.num_lasers) * self.radian_unit + self.start_phase_offset

        # override these properties to decide which elements to detect and show
        self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam)
        self.mask = CollisionGroup.BrokenLaneLine
        self.cloud_points_vis = [] if show else None
        logging.debug("Load Vehicle Module: {}".format(self.__class__.__name__))
        if show:
            for laser_debug in range(self.num_lasers):
                ball = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                ball.setScale(0.001)
                ball.setColor(0., 0.5, 0.5, 1)
                shape = BulletSphereShape(0.1)
                ghost = BulletGhostNode('Lidar Point')
                ghost.setIntoCollideMask(CollisionGroup.AllOff)
                ghost.addShape(shape)
                laser_np = self.origin.attachNewNode(ghost)
                self.cloud_points_vis.append(laser_np)
                ball.getChildren().reparentTo(laser_np)
            # self.origin.flattenStrong()

    def perceive(self, base_vehicle, physics_world, detector_mask: np.ndarray = None):
        assert self.available
        extra_filter_node = set(base_vehicle.dynamic_nodes)
        vehicle_position = base_vehicle.position
        heading_theta = base_vehicle.heading_theta
        assert not isinstance(detector_mask, str), "Please specify detector_mask either with None or a numpy array."
        cloud_points, detected_objects, colors = perceive(
            cloud_points=np.ones((self.num_lasers, ), dtype=float),
            detector_mask=detector_mask.astype(dtype=np.uint8) if detector_mask is not None else None,
            mask=self.mask,
            lidar_range=self._lidar_range,
            perceive_distance=self.perceive_distance,
            heading_theta=heading_theta,
            vehicle_position_x=vehicle_position[0],
            vehicle_position_y=vehicle_position[1],
            num_lasers=self.num_lasers,
            height=self.height,
            physics_world=physics_world,
            extra_filter_node=extra_filter_node if extra_filter_node else set(),
            require_colors=self.cloud_points_vis is not None,
            ANGLE_FACTOR=self.ANGLE_FACTOR,
            MARK_COLOR0=self.MARK_COLOR[0],
            MARK_COLOR1=self.MARK_COLOR[1],
            MARK_COLOR2=self.MARK_COLOR[2]
        )
        if self.cloud_points_vis is not None:
            for laser_index, pos, color in colors:
                self.cloud_points_vis[laser_index].setPos(pos)
                self.cloud_points_vis[laser_index].setColor(*color)
        return detect_result(cloud_points=cloud_points.tolist(), detected_objects=detected_objects)

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

    def destroy(self):
        if self.cloud_points_vis:
            for vis_laser in self.cloud_points_vis:
                vis_laser.removeNode()
        self.origin.removeNode()

    def set_start_phase_offset(self, angle: float):
        """
        Change the start phase of lidar lasers
        :param angle: phasse offset in [degree]
        """
        self.start_phase_offset = np.deg2rad(angle)
        self._lidar_range = np.arange(0, self.num_lasers) * self.radian_unit + self.start_phase_offset

    def __del__(self):
        logging.debug("Lidar is destroyed.")

    def detach_from_world(self):
        if isinstance(self.origin, NodePath):
            self.origin.detachNode()

    def attach_to_world(self, engine):
        if isinstance(self.origin, NodePath):
            self.origin.reparentTo(engine.render)


class SideDetector(DistanceDetector):
    def __init__(self, num_lasers: int = 2, distance: float = 50, enable_show=True):
        super(SideDetector, self).__init__(num_lasers, distance, enable_show)
        self.set_start_phase_offset(90)
        self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam)
        self.mask = CollisionGroup.ContinuousLaneLine


class LaneLineDetector(SideDetector):
    MARK_COLOR = (1, 77 / 255, 77 / 255)

    def __init__(self, num_lasers: int = 2, distance: float = 50, enable_show=True):
        super(SideDetector, self).__init__(num_lasers, distance, enable_show)
        self.set_start_phase_offset(90)
        self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam)
        self.mask = CollisionGroup.ContinuousLaneLine | CollisionGroup.BrokenLaneLine
