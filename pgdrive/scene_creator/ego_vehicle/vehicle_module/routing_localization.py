import logging

import numpy as np
from panda3d.core import NodePath

from pgdrive.pg_config.parameter_space import BlockParameterSpace, Parameter
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.lanes.circular_lane import CircularLane
from pgdrive.scene_creator.map import Map
from pgdrive.utils.asset_loader import AssetLoader
from pgdrive.utils.math_utils import clip, norm


class RoutingLocalizationModule:
    Navi_obs_dim = 10
    """
    It is necessary to interactive with other traffic vehicles
    """
    NAVI_POINT_DIST = 50

    def __init__(self, show_navi_point: False):
        """
        It now only support from first block start to the end node, but can be extended easily
        """
        self.map = None
        self.final_road = None
        self.checkpoints = None
        self.final_lane = None
        self.target_checkpoints_index = None
        self.navi_point_vis = None
        self.ndoe_path = None
        self.navi_info = None
        self.current_ref_lanes = None
        self.show_navi_point = show_navi_point
        logging.debug("Load Vehicle Module: {}".format(self.__class__.__name__))

    def update(self, map: Map, parent_node_path):
        self.map = map
        self.final_road = np.random.RandomState(map.random_seed).choice(map.blocks[-1]._sockets).positive_road
        self.checkpoints = self.map.road_network.shortest_path(FirstBlock.NODE_1, self.final_road.end_node)
        self.final_lane = self.final_road.get_lanes(map.road_network)[-1]
        self.target_checkpoints_index = [0, 1]
        self.navi_point_vis = [] if self.show_navi_point else None
        self.ndoe_path = NodePath("navi_points") if self.show_navi_point else None
        self.navi_info = []
        target_road_1_start = self.checkpoints[0]
        target_road_1_end = self.checkpoints[1]
        self.current_ref_lanes = self.map.road_network.graph[target_road_1_start][target_road_1_end]

        if self.show_navi_point:
            for _ in self.target_checkpoints_index:
                navi_point_model = AssetLoader.loader.loadModel(
                    AssetLoader.file_path(AssetLoader.asset_path, "models", "box.egg")
                )
                navi_point_model.setScale(1)
                navi_point_model.setColor(0, 0.5, 0.5)
                navi_point_model.reparentTo(self.ndoe_path)
                self.navi_point_vis.append(navi_point_model)
            self.ndoe_path.reparentTo(parent_node_path)

    def get_navigate_landmarks(self, distance):
        ret = []
        for L in range(len(self.checkpoints) - 1):
            start = self.checkpoints[L]
            end = self.checkpoints[L + 1]
            target_lanes = self.map.road_network.graph[start][end]
            idx = self.map.lane_num // 2 - 1
            ref_lane = target_lanes[idx]
            for tll in range(3, int(ref_lane.length), 3):
                check_point = ref_lane.position(tll, 0)
                ret.append([check_point[0], -check_point[1]])
        return ret

    def update_navigation_localization(self, ego_vehicle):
        position = ego_vehicle.position
        lane_index = self.map.road_network.get_closest_lane_index(position)
        lane = self.map.road_network.get_lane(lane_index)
        self._update_target_checkpoints(lane_index)

        target_road_1_start = self.checkpoints[self.target_checkpoints_index[0]]
        target_road_1_end = self.checkpoints[self.target_checkpoints_index[0] + 1]
        target_road_2_start = self.checkpoints[self.target_checkpoints_index[1]]
        target_road_2_end = self.checkpoints[self.target_checkpoints_index[1] + 1]
        target_lanes_1 = self.map.road_network.graph[target_road_1_start][target_road_1_end]
        target_lanes_2 = self.map.road_network.graph[target_road_2_start][target_road_2_end]
        res = []
        self.current_ref_lanes = target_lanes_1

        for lanes_id, lanes in enumerate([target_lanes_1, target_lanes_2]):
            ref_lane = lanes[0]
            later_middle = (float(self.map.lane_num) / 2 - 0.5) * self.map.lane_width
            if isinstance(ref_lane, CircularLane) and ref_lane.direction == 1:
                ref_lane = lanes[-1]
                later_middle *= -1
            check_point = ref_lane.position(ref_lane.length, later_middle)
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
                    BlockParameterSpace.CURVE[Parameter.radius].max + self.map.lane_num * self.map.lane_width
                )
                dir = ref_lane.direction
                if dir == 1:
                    angle = ref_lane.end_phase - ref_lane.start_phase
                elif dir == -1:
                    angle = ref_lane.start_phase - ref_lane.end_phase
            res += [
                clip((proj_heading / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0),
                clip((proj_side / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0),
                clip(bendradius, 0.0, 1.0),
                clip((dir + 1) / 2, 0.0, 1.0),
                clip((np.rad2deg(angle) / BlockParameterSpace.CURVE[Parameter.angle].max + 1) / 2, 0.0, 1.0)
            ]
            if self.navi_point_vis is not None:
                self.navi_point_vis[lanes_id].setPos(check_point[0], -check_point[1], 0.5)
        self.navi_info = res
        return lane, lane_index

    def _update_target_checkpoints(self, ego_lane_index):
        current_road_start_point = ego_lane_index[0]
        # print(current_road_start_point, self.vehicle.lane_index[1])
        # print(self.checkpoints[self.target_checkpoints_index[0]], self.checkpoints[self.target_checkpoints_index[1]])
        if self.target_checkpoints_index[0] == self.target_checkpoints_index[1]:
            # on last road
            return

        # arrive to second checkpoint
        if current_road_start_point == self.checkpoints[self.target_checkpoints_index[1]]:
            last_checkpoint_idx = self.target_checkpoints_index.pop(0)
            next_checkpoint_idx = last_checkpoint_idx + 2
            if next_checkpoint_idx == len(self.checkpoints) - 1:
                self.target_checkpoints_index.append(next_checkpoint_idx - 1)
            else:
                self.target_checkpoints_index.append(next_checkpoint_idx)
            # print(self.target_checkpoints_index)

    def get_navi_info(self):
        return self.navi_info
