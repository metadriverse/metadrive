import logging

import numpy as np
from panda3d.bullet import BulletGhostNode, BulletSphereShape
from panda3d.core import BitMask32, NodePath

from pgdrive.constants import CamMask, CollisionGroup
from pgdrive.utils.asset_loader import AssetLoader
from pgdrive.utils.coordinates_shift import panda_position


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
        show = enable_show and (AssetLoader.loader is not None)
        self.dim = num_lasers
        self.num_lasers = num_lasers
        self.perceive_distance = distance
        self.height = self.DEFAULT_HEIGHT

        self.radian_unit = 2 * np.pi / num_lasers
        self.start_phase_offset = 0
        self.detection_results = []
        self.node_path = parent_node_np.attachNewNode("Could_points")

        # override these properties to decide which elements to detect and show
        self.node_path.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.Shadow | CamMask.DepthCam)
        self.mask = BitMask32.bit(CollisionGroup.BrokenLaneLine)

        self.cloud_points = [] if show else None
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
                self.cloud_points.append(laser_np)
                ball.getChildren().reparentTo(laser_np)
            # self.node_path.flattenStrong()

    def perceive(self, vehicle_position, heading_theta, pg_physics_world):
        """
        Call me to update the perception info
        """
        # coordinates problem here! take care
        pg_start_position = panda_position(vehicle_position, self.height)
        self.detection_results = []

        # lidar calculation use pg coordinates
        mask = self.mask
        laser_heading = np.arange(0, self.num_lasers) * self.radian_unit + heading_theta + self.start_phase_offset
        point_x = self.perceive_distance * np.cos(laser_heading) + vehicle_position[0]
        point_y = self.perceive_distance * np.sin(laser_heading) + vehicle_position[1]
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        for laser_index in range(self.num_lasers):
            # # coordinates problem here! take care
            laser_end = panda_position((point_x[laser_index], point_y[laser_index]), self.height)
            result = pg_physics_world.rayTestClosest(pg_start_position, laser_end, mask)
            self.detection_results.append(result)
            if self.cloud_points is not None:
                if result.hasHit():
                    curpos = result.getHitPos()
                else:
                    curpos = laser_end
                # if 0<=laser_index < 10 or 230 <= laser_index <=239:
                #     self.cloud_points[laser_index].setColor(1,0,0)
                self.cloud_points[laser_index].setPos(curpos)
                f = laser_index / self.num_lasers if self.ANGLE_FACTOR else 1
                self.cloud_points[laser_index].setColor(
                    f * self.MARK_COLOR[0], f * self.MARK_COLOR[1], f * self.MARK_COLOR[2]
                )

    def get_cloud_points(self):
        return [point.getHitFraction() for point in self.detection_results]

    def destroy(self):
        if self.cloud_points:
            for vis_laser in self.cloud_points:
                vis_laser.removeNode()
        self.node_path.removeNode()
        self.detection_results = None

    def set_start_phase_offset(self, angle: float):
        """
        Change the start phase of lidar lasers
        :param angle: phasse offset in [degree]
        :return: None
        """
        self.start_phase_offset = np.deg2rad(angle)

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
