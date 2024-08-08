from collections import namedtuple
from metadrive.engine.core.draw import ColorLineNodePath

import numpy as np
from panda3d.core import NodePath, LVecBase4
from metadrive.component.sensors.base_sensor import BaseSensor
from metadrive.constants import CamMask, CollisionGroup
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.logger import get_logger
from metadrive.utils.coordinates_shift import panda_vector
from metadrive.utils.math import panda_vector, get_laser_end

detect_result = namedtuple("detect_result", "cloud_points detected_objects")

logger = get_logger()


def add_cloud_point_vis(
    point_x, point_y, height, num_lasers, laser_index, ANGLE_FACTOR, MARK_COLOR0, MARK_COLOR1, MARK_COLOR2
):
    f = laser_index / num_lasers if ANGLE_FACTOR else 1
    f *= 0.9
    f += 0.1
    return laser_index, (point_x, point_y, height), (f * MARK_COLOR0, f * MARK_COLOR1, f * MARK_COLOR2)


def perceive(
    cloud_points, detector_mask, mask, lidar_range, perceive_distance, heading_theta, vehicle_position_x,
    vehicle_position_y, num_lasers, height, physics_world, extra_filter_node, require_colors, ANGLE_FACTOR, MARK_COLOR0,
    MARK_COLOR1, MARK_COLOR2
):
    cloud_points.fill(1.0)
    detected_objects = []
    colors = []
    pg_start_position = panda_vector(vehicle_position_x, vehicle_position_y, height)

    for laser_index in range(num_lasers):
        if (detector_mask is not None) and (not detector_mask[laser_index]):
            # update vis
            if require_colors:
                point_x, point_y = get_laser_end(
                    lidar_range, perceive_distance, laser_index, heading_theta, vehicle_position_x, vehicle_position_y
                )
                point_x, point_y, point_z = panda_vector(point_x, point_y, height)
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
        laser_end = panda_vector(point_x, point_y, height)
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


class DistanceDetector(BaseSensor):
    """
    It is a module like lidar, used to detect sidewalk/center line or other static things
    """
    DEFAULT_HEIGHT = 0.2

    # for vis debug
    MARK_COLOR = (51 / 255, 221 / 255, 1)
    ANGLE_FACTOR = False

    def __init__(self, engine):
        self.logger = get_logger()
        self.engine = engine
        # properties
        self._node_path_list = []
        parent_node_np: NodePath = engine.render
        self.origin = parent_node_np.attachNewNode("Could_points")
        self.start_phase_offset = 0

        # override these properties to decide which elements to detect and show
        self.mask = CollisionGroup.BrokenLaneLine
        # visualization
        self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam | CamMask.SemanticCam)
        self.cloud_points_vis = ColorLineNodePath(
            self.origin, thickness=3.0
        ) if AssetLoader.loader is not None else None
        self.logger.debug("Load Vehicle Module: {}".format(self.__class__.__name__))
        self._current_frame = None

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
        height = height or self.DEFAULT_HEIGHT
        extra_filter_node = set(base_vehicle.dynamic_nodes)
        vehicle_position = base_vehicle.position
        heading_theta = base_vehicle.heading_theta
        assert not isinstance(detector_mask, str), "Please specify detector_mask either with None or a numpy array."
        cloud_points, detected_objects, colors = perceive(
            cloud_points=np.ones((num_lasers, ), dtype=float),
            detector_mask=detector_mask.astype(dtype=np.uint8) if detector_mask is not None else None,
            mask=self.mask,
            lidar_range=self._get_lidar_range(num_lasers, self.start_phase_offset),
            perceive_distance=distance,
            heading_theta=heading_theta,
            vehicle_position_x=vehicle_position[0],
            vehicle_position_y=vehicle_position[1],
            num_lasers=num_lasers,
            height=height,
            physics_world=physics_world,
            extra_filter_node=extra_filter_node if extra_filter_node else set(),
            require_colors=self.cloud_points_vis is not None,
            ANGLE_FACTOR=self.ANGLE_FACTOR,
            MARK_COLOR0=self.MARK_COLOR[0],
            MARK_COLOR1=self.MARK_COLOR[1],
            MARK_COLOR2=self.MARK_COLOR[2]
        )

        if show and self.cloud_points_vis is not None:
            colors = colors + colors[:1]
            if self._current_frame != self.engine.episode_step:
                self.cloud_points_vis.reset()
            self._current_frame = self.engine.episode_step
            self.cloud_points_vis.draw_lines([[p[1] for p in colors]], [[LVecBase4(*p[-1], 1) for p in colors[1:]]])

        return detect_result(cloud_points=cloud_points.tolist(), detected_objects=detected_objects)

    def destroy(self):
        if self.cloud_points_vis:
            self.cloud_points_vis.removeNode()
        self.origin.removeNode()
        for np in self._node_path_list:
            np.detachNode()
            np.removeNode()

    def set_start_phase_offset(self, angle: float):
        """
        Change the start phase of lidar lasers
        :param angle: phasse offset in [degree]
        """
        self.start_phase_offset = np.deg2rad(angle)

    @staticmethod
    def _get_lidar_range(num_lasers, start_phase_offset):
        radian_unit = 2 * np.pi / num_lasers if num_lasers > 0 else None
        return np.arange(0, num_lasers) * radian_unit + start_phase_offset

    def __del__(self):
        logger.debug("Lidar is destroyed.")

    def detach_from_world(self):
        if isinstance(self.origin, NodePath):
            self.origin.detachNode()

    def attach_to_world(self, engine):
        if isinstance(self.origin, NodePath):
            self.origin.reparentTo(engine.render)


class SideDetector(DistanceDetector):
    def __init__(self, engine):
        super(SideDetector, self).__init__(engine)
        self.set_start_phase_offset(90)
        self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam | CamMask.SemanticCam)
        self.mask = CollisionGroup.ContinuousLaneLine | CollisionGroup.Sidewalk


class LaneLineDetector(SideDetector):
    MARK_COLOR = (1, 77 / 255, 77 / 255)

    def __init__(self, engine):
        super(SideDetector, self).__init__(engine)
        self.set_start_phase_offset(90)
        self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam | CamMask.SemanticCam)
        self.mask = CollisionGroup.ContinuousLaneLine | CollisionGroup.BrokenLaneLine | CollisionGroup.Sidewalk
