import logging
from metadrive.component.road_network.node_road_network import NodeRoadNetwork

import numpy as np
from panda3d.core import TransparencyAttrib, LineSegs, NodePath

from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road
from metadrive.constants import RENDER_MODE_ONSCREEN, CamMask
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils import clip, norm, get_np_random
from metadrive.utils.coordinates_shift import panda_position
from metadrive.utils.scene_utils import ray_localization
from metadrive.utils.space import Parameter, BlockParameterSpace


class BaseNavigation:
    navigation_info_dim = 10
    NAVI_POINT_DIST = 50
    PRE_NOTIFY_DIST = 40
    MIN_ALPHA = 0.15
    CKPT_UPDATE_RANGE = 5
    FORCE_CALCULATE = False
    LINE_TO_DEST_HEIGHT = 0.6

    def __init__(
        self,
        engine,
        show_navi_mark: bool = False,
        random_navi_mark_color=False,
        show_dest_mark=False,
        show_line_to_dest=False,
        panda_color=None
    ):
        """
        This class define a helper for localizing vehicles and retrieving navigation information.
        It now only support from first block start to the end node, but can be extended easily.
        """
        self.map = None
        self.checkpoints = None
        self._target_checkpoints_index = None
        self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self.current_lane = None
        self._navi_info = np.zeros((self.navigation_info_dim, ), dtype=np.float32)  # navi information res

        # Vis
        self._show_navi_info = (engine.mode == RENDER_MODE_ONSCREEN and not engine.global_config["debug_physics_world"])
        self.origin = NodePath("navigation_sign") if self._show_navi_info else None
        self.navi_mark_color = (0.6, 0.8, 0.5) if not random_navi_mark_color else get_np_random().rand(3)
        if panda_color is not None:
            assert len(panda_color) == 3 and 0 <= panda_color[0] <= 1
            self.navi_mark_color = tuple(panda_color)
        self.navi_arrow_dir = [0, 0]
        self._dest_node_path = None
        self._goal_node_path = None

        self._line_to_dest = None
        self._show_line_to_dest = show_line_to_dest
        if self._show_navi_info:
            # nodepath
            self._line_to_dest = self.origin.attachNewNode("line")
            self._goal_node_path = self.origin.attachNewNode("target")
            self._dest_node_path = self.origin.attachNewNode("dest")

            if show_navi_mark:
                navi_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                navi_point_model.reparentTo(self._goal_node_path)
            if show_dest_mark:
                dest_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                dest_point_model.reparentTo(self._dest_node_path)
            if show_line_to_dest:
                line_seg = LineSegs("line_to_dest")
                line_seg.setColor(self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 1.0)
                line_seg.setThickness(4)
                self._dynamic_line_np = NodePath(line_seg.create(True))
                self._dynamic_line_np.reparentTo(self.origin)
                self._line_to_dest = line_seg

            self._goal_node_path.setTransparency(TransparencyAttrib.M_alpha)
            self._dest_node_path.setTransparency(TransparencyAttrib.M_alpha)

            self._goal_node_path.setColor(
                self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.7
            )
            self._dest_node_path.setColor(
                self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.7
            )
            self._goal_node_path.hide(CamMask.AllOn)
            self._dest_node_path.hide(CamMask.AllOn)
            self._goal_node_path.show(CamMask.MainCam)
            self._dest_node_path.show(CamMask.MainCam)
        logging.debug("Load Vehicle Module: {}".format(self.__class__.__name__))

    def reset(self, map: BaseMap, current_lane):
        self.map = map
        self.current_lane = current_lane

    def get_checkpoints(self):
        """Return next checkpoint and the next next checkpoint"""
        later_middle = (float(self.get_current_lane_num()) / 2 - 0.5) * self.get_current_lane_width()
        ref_lane1 = self.current_ref_lanes[0]
        checkpoint1 = ref_lane1.position(ref_lane1.length, later_middle)
        ref_lane2 = self.next_ref_lanes[0] if self.next_ref_lanes is not None else self.current_ref_lanes[0]
        checkpoint2 = ref_lane2.position(ref_lane2.length, later_middle)
        return checkpoint1, checkpoint2

    def set_route(self, current_lane_index: str, destination: str):
        """
        Find a shortest path from start road to end road
        :param current_lane_index: start road node
        :param destination: end road node or end lane index
        :return: None
        """
        raise NotImplementedError

    def update_localization(self, ego_vehicle):
        """
        It is called every step
        """
        raise NotImplementedError

    def _get_info_for_checkpoint(self, lanes_id, ref_lane, ego_vehicle):

        navi_information = []
        # Project the checkpoint position into the target vehicle's coordination, where
        # +x is the heading and +y is the right hand side.
        later_middle = (float(self.get_current_lane_num()) / 2 - 0.5) * self.get_current_lane_width()
        check_point = ref_lane.position(ref_lane.length, later_middle)
        dir_vec = check_point - ego_vehicle.position  # get the vector from center of vehicle to checkpoint
        dir_norm = norm(dir_vec[0], dir_vec[1])
        if dir_norm > self.NAVI_POINT_DIST:  # if the checkpoint is too far then crop the direction vector
            dir_vec = dir_vec / dir_norm * self.NAVI_POINT_DIST
        ckpt_in_heading, ckpt_in_rhs = ego_vehicle.projection(dir_vec)  # project to ego vehicle's coordination

        # Dim 1: the relative position of the checkpoint in the target vehicle's heading direction.
        navi_information.append(clip((ckpt_in_heading / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        # Dim 2: the relative position of the checkpoint in the target vehicle's right hand side direction.
        navi_information.append(clip((ckpt_in_rhs / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        if lanes_id == 0:
            lanes_heading = ref_lane.heading_theta_at(ref_lane.local_coordinates(ego_vehicle.position)[0])
        else:
            lanes_heading = ref_lane.heading_theta_at(min(self.PRE_NOTIFY_DIST, ref_lane.length))

        # Try to include the current lane's information into the navigation information
        bendradius = 0.0
        dir = 0.0
        angle = 0.0
        if isinstance(ref_lane, CircularLane):
            bendradius = ref_lane.radius / (
                BlockParameterSpace.CURVE[Parameter.radius].max +
                self.get_current_lane_num() * self.get_current_lane_width()
            )
            dir = ref_lane.direction
            if dir == 1:
                angle = ref_lane.end_phase - ref_lane.start_phase
            elif dir == -1:
                angle = ref_lane.start_phase - ref_lane.end_phase

        # Dim 3: The bending radius of current lane
        navi_information.append(clip(bendradius, 0.0, 1.0))

        # Dim 4: The bending direction of current lane (+1 for clockwise, -1 for counterclockwise)
        navi_information.append(clip((dir + 1) / 2, 0.0, 1.0))

        # Dim 5: The angular difference between the heading in lane ending position and
        # the heading in lane starting position
        navi_information.append(
            clip((np.rad2deg(angle) / BlockParameterSpace.CURVE[Parameter.angle].max + 1) / 2, 0.0, 1.0)
        )
        return navi_information, lanes_heading, check_point

    def _update_target_checkpoints(self, ego_lane_index, ego_lane_longitude):
        raise NotImplementedError

    def get_navi_info(self):
        return self._navi_info

    def destroy(self):
        if self._show_navi_info:
            try:
                self._line_to_dest.removeNode()
            except AttributeError:
                pass
            self._dest_node_path.removeNode()
            self._goal_node_path.removeNode()
        self.next_ref_lanes = None
        self.current_ref_lanes = None

    def set_force_calculate_lane_index(self, force: bool):
        self.FORCE_CALCULATE = force

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def get_current_lateral_range(self, current_position, engine) -> float:
        raise NotImplementedError

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return len(self.current_ref_lanes)

    def _get_current_lane(self, ego_vehicle):
        """
        Called in update_localization to find current lane information
        """
        possible_lanes, on_lane = ray_localization(
            ego_vehicle.heading, ego_vehicle.position, ego_vehicle.engine, return_all_result=True, return_on_lane=True
        )
        for lane, index, l_1_dist in possible_lanes:
            if lane in self.current_ref_lanes:
                return lane, index, on_lane
        nx_ckpt = self._target_checkpoints_index[-1]
        if nx_ckpt == self.checkpoints[-1] or self.next_ref_lanes is None:
            return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)

        if self.map.road_network_type == NodeRoadNetwork:
            nx_nx_ckpt = nx_ckpt + 1
            next_ref_lanes = self.map.road_network.graph[self.checkpoints[nx_ckpt]][self.checkpoints[nx_nx_ckpt]]
        else:
            next_ref_lanes = self.next_ref_lanes
        for lane, index, l_1_dist in possible_lanes:
            if lane in next_ref_lanes:
                return lane, index, on_lane
        return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)

    def _update_current_lane(self, ego_vehicle):
        lane, lane_index, on_lane = self._get_current_lane(ego_vehicle)
        ego_vehicle.on_lane = on_lane
        if lane is None:
            lane, lane_index = ego_vehicle.lane, ego_vehicle.lane_index
            if self.FORCE_CALCULATE:
                lane_index, _ = self.map.road_network.get_closest_lane_index(ego_vehicle.position)
                lane = self.map.road_network.get_lane(lane_index)
        self.current_lane = lane
        assert lane_index == lane.index, "lane index mismatch!"
        return lane, lane_index

    def _ray_lateral_range(self, engine, start_position, dir, length=50):
        """
        It is used to measure the lateral range of special blocks
        :param start_position: start_point
        :param dir: ray direction
        :param length: length of ray
        :return: lateral range [m]
        """
        end_position = start_position[0] + dir[0] * length, start_position[1] + dir[1] * length
        start_position = panda_position(start_position, z=0.15)
        end_position = panda_position(end_position, z=0.15)
        mask = FirstPGBlock.CONTINUOUS_COLLISION_MASK
        res = engine.physics_world.static_world.rayTestClosest(start_position, end_position, mask=mask)
        if not res.hasHit():
            return length
        else:
            return res.getHitFraction() * length

    def _draw_line_to_dest(self, start_position, end_position):
        if not self._show_line_to_dest:
            return
        line_seg = self._line_to_dest
        line_seg.moveTo(panda_position(start_position, self.LINE_TO_DEST_HEIGHT))
        line_seg.drawTo(panda_position(end_position, self.LINE_TO_DEST_HEIGHT))
        self._dynamic_line_np.removeNode()
        self._dynamic_line_np = NodePath(line_seg.create(False))
        self._dynamic_line_np.hide(CamMask.Shadow | CamMask.RgbCam)
        self._dynamic_line_np.reparentTo(self.origin)

    def detach_from_world(self):
        if isinstance(self.origin, NodePath):
            self.origin.detachNode()

    def attach_to_world(self, engine):
        if isinstance(self.origin, NodePath):
            self.origin.reparentTo(engine.render)
