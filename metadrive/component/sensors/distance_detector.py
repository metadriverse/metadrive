import logging
from collections import namedtuple

import math
import numpy as np
from panda3d.bullet import BulletGhostNode, BulletSphereShape
from panda3d.core import NodePath

from metadrive.constants import CamMask, CollisionGroup
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import get_engine
from metadrive.utils.coordinates_shift import panda_vector
from metadrive.utils.math import panda_vector, get_laser_end

detect_result = namedtuple("detect_result", "cloud_points detected_objects")


def add_cloud_point_vis(
    point_x, point_y, height, num_lasers, laser_index, ANGLE_FACTOR, MARK_COLOR0, MARK_COLOR1, MARK_COLOR2
):
    f = laser_index / num_lasers if ANGLE_FACTOR else 1
    f *= 0.9
    f += 0.1
    return laser_index, (point_x, point_y, height), (f * MARK_COLOR0, f * MARK_COLOR1, f * MARK_COLOR2)


def d3_get_laser_end(lidar_range, perceive_distance, laser_index, heading_theta, 
                     vehicle_position_x, vehicle_position_y, vehicle_position_z,pitch):
    angle = lidar_range[laser_index] + heading_theta
    return (
        perceive_distance * math.cos(angle) * math.cos(pitch) + vehicle_position_x,
        perceive_distance * math.sin(angle) * math.cos(pitch) + vehicle_position_y,
        perceive_distance * math.sin(pitch) + vehicle_position_z
    )



def perceive(
    cloud_points, detector_mask, mask, lidar_range, perceive_distance, heading_theta, vehicle_position_x,
    vehicle_position_y, num_lasers, height, physics_world, extra_filter_node, require_colors, ANGLE_FACTOR, MARK_COLOR0,
    MARK_COLOR1, MARK_COLOR2,pitch
):
    cloud_points.fill(1.0)
    detected_objects = []
    colors = []
    pg_start_position = panda_vector(vehicle_position_x, vehicle_position_y, height)

    for laser_index in range(num_lasers):
        if (detector_mask is not None) and (not detector_mask[laser_index]):
            # update vis
            if require_colors:
                point_x, point_y, point_z = d3_get_laser_end(
                    lidar_range, perceive_distance, laser_index, heading_theta, vehicle_position_x, vehicle_position_y,height,pitch
                )
                point_x, point_y, point_z = panda_vector(point_x, point_y, point_z)
                colors.append(
                    add_cloud_point_vis(
                        point_x, point_y, point_z, num_lasers, laser_index, ANGLE_FACTOR, MARK_COLOR0, MARK_COLOR1,
                        MARK_COLOR2
                    )
                )
            continue

        # # coordinates problem here! take care
        point_x, point_y, point_z = d3_get_laser_end(
            lidar_range, perceive_distance, laser_index, heading_theta, vehicle_position_x, vehicle_position_y,height,pitch
        )
        laser_end = panda_vector(point_x, point_y, point_z)
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
                    laser_end[0], laser_end[1], laser_end[2], num_lasers, laser_index, ANGLE_FACTOR, MARK_COLOR0, MARK_COLOR1,
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

    def __init__(self, num_lasers: int = 16, distance: float = 50, enable_show:bool=True,
                  pitch:float = 0, vfov:float = 0, num_lasers_v:int = 1, generate = False):
        """
        pitch: in rad
        """
        # properties
        self._node_path_list = []
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

        self.pitch = pitch
        self.num_lasers_v = num_lasers_v
        self.vfov = np.deg2rad(vfov)
        self.enable_show = enable_show
        self.generate = generate


        # override these properties to decide which elements to detect and show
        self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam)
        self.mask = CollisionGroup.BrokenLaneLine
        self.cloud_points_vis = [] if show else None
        logging.debug("Load Vehicle Module: {}".format(self.__class__.__name__))
        if show:
            for laser_debug in range(self.num_lasers*self.num_lasers_v):
                ball = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                ball.setScale(0.25)
                ball.setColor(0., 0.5, 0.5, 1)
                ball.reparentTo(self.origin)
                self.cloud_points_vis.append(ball)
            # self.origin.flattenStrong()

        self.closest_observed_point = None

    def perceive(self, base_vehicle, physics_world, detector_mask: np.ndarray = None,position = None, heading = None, ):
        assert self.available
        assert self.num_lasers_v == 1, "You should use perceive3d!"
        extra_filter_node = set(base_vehicle.dynamic_nodes)
        vehicle_position = base_vehicle.position if position is None else position   #Added this conditional to make the lidar's spatial property configuration 
        heading_theta = base_vehicle.heading_theta if heading is None else heading   #Added this conditional to make the lidar's spatial property configuration
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
            MARK_COLOR2=self.MARK_COLOR[2],
            pitch = self.pitch
        )
        if self.cloud_points_vis is not None:
            for laser_index, pos, color in colors:
                self.cloud_points_vis[laser_index].setPos(pos)
                self.cloud_points_vis[laser_index].setColor(*color)
        return detect_result(cloud_points=cloud_points.tolist(), detected_objects=detected_objects)
    


    def perceive3d(self, base_vehicle, physics_world, detector_mask: np.ndarray = None,position = None, 
                   heading = None):
        assert self.available
        extra_filter_node = set(base_vehicle.dynamic_nodes)
        vehicle_position = base_vehicle.position if position is None else position   
        heading_theta = base_vehicle.heading_theta if heading is None else heading
        all_cloud_points = []
        all_objcts  = []
        base = 0
        pitch_space = np.linspace(self.pitch - self.vfov/2, self.pitch + self.vfov, self.num_lasers_v)
        minimum = 2
        minimum_pos = None
        for i in range(self.num_lasers_v):
            current_pitch = pitch_space[i]
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
                require_colors=self.cloud_points_vis is not None or self.generate,
                ANGLE_FACTOR=self.ANGLE_FACTOR,
                MARK_COLOR0=self.MARK_COLOR[0],
                MARK_COLOR1=self.MARK_COLOR[1],
                MARK_COLOR2=self.MARK_COLOR[2],
                pitch = current_pitch
            )
            if self.generate:
                candidate = min(cloud_points)
                if candidate <1 and candidate < minimum :
                    minimum = candidate
                    #Note: in order to retrieve the world coordinate of the hitted vertice, show_visualiation must be true. Otherwise, no
                    #points will be appended to colors, and the minimum_pos expression will raise index out of range exception.
                    minimum_pos = colors[np.argmin(cloud_points)][1]
            if self.cloud_points_vis is not None and self.enable_show:
                for laser_index, pos, color in colors:
                    self.cloud_points_vis[base + laser_index].setPos(pos)
                    self.cloud_points_vis[base + laser_index].setColor(*color)
                base += len(colors)
            all_cloud_points += cloud_points.tolist()
            all_objcts += detected_objects
        if self.generate:
            self.closest_observed_point = (minimum*self.perceive_distance, minimum_pos)
        return  detect_result(cloud_points=all_cloud_points, detected_objects=all_objcts)

    def _add_cloud_point_vis(self, laser_index, pos):
        self.cloud_points_vis[laser_index].setPos(pos)
        f = laser_index / self.num_lasers if self.ANGLE_FACTOR else 1
        self.cloud_points_vis[laser_index].setColor(
            f * self.MARK_COLOR[0], f * self.MARK_COLOR[1], f * self.MARK_COLOR[2]
        )

    def _get_laser_end(self, laser_index, heading_theta, vehicle_position):
        """
        This method seems not called anywhere. Consider removing it?
        """
        point_x = self.perceive_distance * math.cos(self._lidar_range[laser_index] + heading_theta) + \
                  vehicle_position[0]
        point_y = self.perceive_distance * math.sin(self._lidar_range[laser_index] + heading_theta) + \
                  vehicle_position[1]
        laser_end = panda_vector((point_x, point_y), self.height)
        return laser_end

    def destroy(self):
        if self.cloud_points_vis:
            for vis_laser in self.cloud_points_vis:
                vis_laser.removeNode()
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
        self._lidar_range = np.arange(0, self.num_lasers) * self.radian_unit + self.start_phase_offset

    def __del__(self):
        logging.debug("Lidar is destroyed.")

    def detach_from_world(self):
        if isinstance(self.origin, NodePath):
            self.origin.detachNode()

    def attach_to_world(self, engine):
        if isinstance(self.origin, NodePath):
            self.origin.reparentTo(engine.render)
    
    def set_height(self, new_height:float):
        self.height = new_height
    
    def set_pitch(self, new_pitch:float):
        """
        new_pitch, in radian relative to x-y plane in a right handed system.
        """
        self.pitch = new_pitch
    def set_vfov(self, new_vfov:float):
        self.vfov = new_vfov



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
