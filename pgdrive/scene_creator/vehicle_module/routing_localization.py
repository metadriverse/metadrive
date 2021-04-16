import logging

import numpy as np
from panda3d.core import BitMask32, LQuaternionf, TransparencyAttrib

from pgdrive.constants import COLLISION_INFO_COLOR, RENDER_MODE_ONSCREEN, CamMask
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.lane.circular_lane import CircularLane
from pgdrive.scene_creator.map import Map
from pgdrive.scene_creator.road.road import Road
from pgdrive.utils import clip, norm, get_np_random
from pgdrive.utils.asset_loader import AssetLoader
from pgdrive.utils.pg_space import Parameter, BlockParameterSpace
from pgdrive.utils.scene_utils import ray_localization


class RoutingLocalizationModule:
    navigation_info_dim = 10
    NAVI_POINT_DIST = 50
    PRE_NOTIFY_DIST = 40
    MARK_COLOR = COLLISION_INFO_COLOR["green"][1]
    MIN_ALPHA = 0.15
    CKPT_UPDATE_RANGE = 5
    FORCE_CALCULATE = False

    def __init__(self, pg_world, show_navi_mark: bool = False):
        """
        This class define a helper for localizing vehicles and retrieving navigation information.
        It now only support from first block start to the end node, but can be extended easily.
        """
        self.map = None
        self.final_road = None
        self.checkpoints = None
        self.final_lane = None
        self.current_ref_lanes = None
        self._target_checkpoints_index = None
        self._navi_info = np.zeros((self.navigation_info_dim, ))  # navi information res

        # Vis
        self._is_showing = True  # store the state of navigation mark
        self._show_navi_point = (
            pg_world.mode == RENDER_MODE_ONSCREEN and not pg_world.world_config["debug_physics_world"]
        )
        self._goal_node_path = None
        self._arrow_node_path = None
        if self._show_navi_point:
            self._goal_node_path = pg_world.render.attachNewNode("target")
            self._arrow_node_path = pg_world.aspect2d.attachNewNode("arrow")
            navi_arrow_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "navi_arrow.gltf"))
            navi_arrow_model.setScale(0.1, 0.12, 0.2)
            navi_arrow_model.setPos(2, 1.15, -0.221)
            self._left_arrow = self._arrow_node_path.attachNewNode("left arrow")
            self._left_arrow.setP(180)
            self._right_arrow = self._arrow_node_path.attachNewNode("right arrow")
            self._left_arrow.setColor(self.MARK_COLOR)
            self._right_arrow.setColor(self.MARK_COLOR)
            navi_arrow_model.instanceTo(self._left_arrow)
            navi_arrow_model.instanceTo(self._right_arrow)
            self._arrow_node_path.setPos(0, 0, 0.08)
            self._arrow_node_path.hide(BitMask32.allOn())
            self._arrow_node_path.show(CamMask.MainCam)
            self._arrow_node_path.setQuat(LQuaternionf(np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)))

            # the transparency attribute of gltf model is invalid on windows
            # self._arrow_node_path.setTransparency(TransparencyAttrib.M_alpha)
            if show_navi_mark:
                navi_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                navi_point_model.reparentTo(self._goal_node_path)
            self._goal_node_path.setTransparency(TransparencyAttrib.M_alpha)
            self._goal_node_path.setColor(0.6, 0.8, 0.5, 0.7)
            self._goal_node_path.hide(BitMask32.allOn())
            self._goal_node_path.show(CamMask.MainCam)
        logging.debug("Load Vehicle Module: {}".format(self.__class__.__name__))

    def update(self, map: Map, start_road_node=None, final_road_node=None, random_seed=False):
        self.map = map
        if start_road_node is None:
            start_road_node = FirstBlock.NODE_1
        if final_road_node is None:
            random_seed = random_seed if random_seed is not False else map.random_seed
            socket = get_np_random(random_seed).choice(map.blocks[-1].get_socket_list())
            final_road_node = socket.positive_road.end_node
        self.set_route(start_road_node, final_road_node)

    def set_route(self, start_road_node: str, end_road_node: str):
        """
        Find a shorest path from start road to end road
        :param start_road_node: start road node
        :param end_road_node: end road node
        :return: None
        """
        self.checkpoints = self.map.road_network.shortest_path(start_road_node, end_road_node)
        assert len(self.checkpoints) >= 2
        # update routing info
        self.final_road = Road(self.checkpoints[-2], end_road_node)
        self.final_lane = self.final_road.get_lanes(self.map.road_network)[-1]
        self._target_checkpoints_index = [0, 1]
        self._navi_info.fill(0.0)
        target_road_1_start = self.checkpoints[0]
        target_road_1_end = self.checkpoints[1]
        self.current_ref_lanes = self.map.road_network.graph[target_road_1_start][target_road_1_end]

    def update_navigation_localization(self, ego_vehicle):
        position = ego_vehicle.position
        lane, lane_index = ray_localization(position, ego_vehicle.pg_world)
        if lane is None:
            lane, lane_index = ego_vehicle.lane, ego_vehicle.lane_index
            ego_vehicle.on_lane = False
            if self.FORCE_CALCULATE:
                lane_index, _ = self.map.road_network.get_closest_lane_index(position)
                lane = self.map.road_network.get_lane(lane_index)
        long, _ = lane.local_coordinates(position)
        self._update_target_checkpoints(lane_index, long)

        target_road_1_start = self.checkpoints[self._target_checkpoints_index[0]]
        target_road_1_end = self.checkpoints[self._target_checkpoints_index[0] + 1]
        target_road_2_start = self.checkpoints[self._target_checkpoints_index[1]]
        target_road_2_end = self.checkpoints[self._target_checkpoints_index[1] + 1]
        target_lanes_1 = self.map.road_network.graph[target_road_1_start][target_road_1_end]
        target_lanes_2 = self.map.road_network.graph[target_road_2_start][target_road_2_end]
        self.current_ref_lanes = target_lanes_1

        self._navi_info.fill(0.0)
        half = self.navigation_info_dim // 2
        self._navi_info[:half], lanes_heading1, checkpoint = self._get_info_for_checkpoint(
            lanes_id=0, lanes=target_lanes_1, ego_vehicle=ego_vehicle
        )

        self._navi_info[half:], lanes_heading2, _ = self._get_info_for_checkpoint(
            lanes_id=1, lanes=target_lanes_2, ego_vehicle=ego_vehicle
        )

        if self._show_navi_point:
            pos_of_goal = checkpoint
            self._goal_node_path.setPos(pos_of_goal[0], -pos_of_goal[1], 1.8)
            self._goal_node_path.setH(self._goal_node_path.getH() + 3)
            self._update_navi_arrow([lanes_heading1, lanes_heading2])

        return lane, lane_index

    def _get_info_for_checkpoint(self, lanes_id, lanes, ego_vehicle):
        ref_lane = lanes[0]
        later_middle = (float(self.get_current_lane_num()) / 2 - 0.5) * self.get_current_lane_width()
        check_point = ref_lane.position(ref_lane.length, later_middle)
        if lanes_id == 0:
            # calculate ego v lane heading
            lanes_heading = ref_lane.heading_at(ref_lane.local_coordinates(ego_vehicle.position)[0])
        else:
            lanes_heading = ref_lane.heading_at(min(self.PRE_NOTIFY_DIST, ref_lane.length))
        dir_vec = check_point - ego_vehicle.position
        dir_norm = norm(dir_vec[0], dir_vec[1])
        if dir_norm > self.NAVI_POINT_DIST:
            dir_vec = dir_vec / dir_norm * self.NAVI_POINT_DIST
        proj_heading, proj_side = ego_vehicle.projection(dir_vec)
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
        return (
            clip((proj_heading / self.NAVI_POINT_DIST + 1) / 2, 0.0,
                 1.0), clip((proj_side / self.NAVI_POINT_DIST + 1) / 2, 0.0,
                            1.0), clip(bendradius, 0.0, 1.0), clip((dir + 1) / 2, 0.0, 1.0),
            clip((np.rad2deg(angle) / BlockParameterSpace.CURVE[Parameter.angle].max + 1) / 2, 0.0, 1.0)
        ), lanes_heading, check_point

    def _update_navi_arrow(self, lanes_heading):
        lane_0_heading = lanes_heading[0]
        lane_1_heading = lanes_heading[1]
        if abs(lane_0_heading - lane_1_heading) < 0.01:
            if self._is_showing:
                self._left_arrow.detachNode()
                self._right_arrow.detachNode()
                self._is_showing = False
        else:
            dir_0 = np.array([np.cos(lane_0_heading), np.sin(lane_0_heading), 0])
            dir_1 = np.array([np.cos(lane_1_heading), np.sin(lane_1_heading), 0])
            cross_product = np.cross(dir_1, dir_0)
            left = False if cross_product[-1] < 0 else True
            if not self._is_showing:
                self._is_showing = True
            if left:
                if not self._left_arrow.hasParent():
                    self._left_arrow.reparentTo(self._arrow_node_path)
                if self._right_arrow.hasParent():
                    self._right_arrow.detachNode()
            else:
                if not self._right_arrow.hasParent():
                    self._right_arrow.reparentTo(self._arrow_node_path)
                if self._left_arrow.hasParent():
                    self._left_arrow.detachNode()

    def _update_target_checkpoints(self, ego_lane_index, ego_lane_longitude):
        """
        Return should_update: True or False
        """
        if self._target_checkpoints_index[0] == self._target_checkpoints_index[1]:  # on last road
            return False

        # arrive to second checkpoint
        current_road_start_point = ego_lane_index[0]
        if current_road_start_point in self.checkpoints[self._target_checkpoints_index[1]:] \
                and ego_lane_longitude < self.CKPT_UPDATE_RANGE:
            if current_road_start_point not in self.checkpoints[self._target_checkpoints_index[1]:-1]:
                return False
            idx = self.checkpoints.index(current_road_start_point, self._target_checkpoints_index[1], -1)
            self._target_checkpoints_index = [idx]
            if idx + 1 == len(self.checkpoints) - 1:
                self._target_checkpoints_index.append(idx)
            else:
                self._target_checkpoints_index.append(idx + 1)
            return True
        return False

    def get_navi_info(self):
        return self._navi_info

    def destroy(self):
        if self._show_navi_point:
            self._arrow_node_path.removeNode()
            self._goal_node_path.removeNode()

    def set_force_calculate_lane_index(self, force: bool):
        self.FORCE_CALCULATE = force

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def get_current_lateral_range(self) -> float:
        """Return the maximum lateral distance from left to right."""
        return self.get_current_lane_width() * self.get_current_lane_num()

    def get_current_lane_width(self) -> float:
        return self.map.config[self.map.LANE_WIDTH]

    def get_current_lane_num(self) -> float:
        return self.map.config[self.map.LANE_NUM]

    # def get_navigate_landmarks(self, distance):
    #     ret = []
    #     for L in range(len(self.checkpoints) - 1):
    #         start = self.checkpoints[L]
    #         end = self.checkpoints[L + 1]
    #         target_lanes = self.map.road_network.graph[start][end]
    #         idx = self.get_current_lane_num() // 2 - 1
    #         ref_lane = target_lanes[idx]
    #         for tll in range(3, int(ref_lane.length), 3):
    #             check_point = ref_lane.position(tll, 0)
    #             ret.append([check_point[0], -check_point[1]])
    #     return ret
