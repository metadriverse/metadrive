import logging

import numpy as np
from panda3d.core import BitMask32, LQuaternionf, TransparencyAttrib
from pgdrive.pg_config.cam_mask import CamMask
from pgdrive.pg_config.parameter_space import BlockParameterSpace, Parameter
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.lanes.circular_lane import CircularLane
from pgdrive.scene_creator.map import Map
from pgdrive.utils.asset_loader import AssetLoader
from pgdrive.utils.math_utils import clip, norm
from pgdrive.world import RENDER_MODE_ONSCREEN
from pgdrive.world.constants import COLLISION_INFO_COLOR


class RoutingLocalizationModule:
    Navi_obs_dim = 10
    """
    It is necessary to interactive with other traffic vehicles
    """
    NAVI_POINT_DIST = 50
    PRE_NOTIFY_DIST = 40
    MARK_COLOR = COLLISION_INFO_COLOR["green"][1]
    MIN_ALPHA = 0.15
    SHOW_NAVI_POINT = False

    def __init__(self, pg_world, show_navi_point: False):
        """
        It now only support from first block start to the end node, but can be extended easily
        """
        self.map = None
        self.final_road = None
        self.checkpoints = None
        self.final_lane = None
        self.target_checkpoints_index = None
        self.navi_info = None  # navi information res
        self.current_ref_lanes = None

        # Vis
        self.showing = True  # store the state of navigation mark
        self.show_navi_point = show_navi_point and pg_world.mode == RENDER_MODE_ONSCREEN and not pg_world.pg_config[
            "debug_physics_world"]
        self.goal_node_path = pg_world.render.attachNewNode("target") if self.show_navi_point else None
        self.arrow_node_path = pg_world.aspect2d.attachNewNode("arrow") if self.show_navi_point else None
        if self.show_navi_point:
            navi_arrow_model = AssetLoader.loader.loadModel(
                AssetLoader.file_path(AssetLoader.asset_path, "models", "navi_arrow.gltf")
            )
            navi_arrow_model.setScale(0.1, 0.12, 0.2)
            navi_arrow_model.setPos(2, 1.15, -0.221)
            self.left_arrow = self.arrow_node_path.attachNewNode("left arrow")
            self.left_arrow.setP(180)
            self.right_arrow = self.arrow_node_path.attachNewNode("right arrow")
            self.left_arrow.setColor(self.MARK_COLOR)
            self.right_arrow.setColor(self.MARK_COLOR)
            navi_arrow_model.instanceTo(self.left_arrow)
            navi_arrow_model.instanceTo(self.right_arrow)
            self.arrow_node_path.setPos(0, 0, 0.08)
            self.arrow_node_path.hide(BitMask32.allOn())
            self.arrow_node_path.show(CamMask.MainCam)
            self.arrow_node_path.setQuat(LQuaternionf(np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)))
            self.arrow_node_path.setTransparency(TransparencyAttrib.M_alpha)
            if self.SHOW_NAVI_POINT:
                navi_point_model = AssetLoader.loader.loadModel(
                    AssetLoader.file_path(AssetLoader.asset_path, "models", "box.egg")
                )
                navi_point_model.reparentTo(self.goal_node_path)
            self.goal_node_path.setTransparency(TransparencyAttrib.M_alpha)
            self.goal_node_path.setColor(0.6, 0.8, 0.5, 0.7)
            self.goal_node_path.hide(BitMask32.allOn())
            self.goal_node_path.show(CamMask.MainCam)
        logging.debug("Load Vehicle Module: {}".format(self.__class__.__name__))

    def update(self, map: Map):
        self.map = map

        # TODO(pzh): I am not sure whether we should change the random state here.
        #  If so, then the vehicle may have different final road in single map, this will avoid it from over-fitting
        #  the map and memorize the routes.
        self.final_road = np.random.RandomState(map.random_seed).choice(map.blocks[-1]._sockets).positive_road
        self.checkpoints = self.map.road_network.shortest_path(FirstBlock.NODE_1, self.final_road.end_node)
        self.final_lane = self.final_road.get_lanes(map.road_network)[-1]
        self.target_checkpoints_index = [0, 1]
        self.navi_info = []
        target_road_1_start = self.checkpoints[0]
        target_road_1_end = self.checkpoints[1]
        self.current_ref_lanes = self.map.road_network.graph[target_road_1_start][target_road_1_end]

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
        ckpts = []
        lanes_heading = []
        for lanes_id, lanes in enumerate([target_lanes_1, target_lanes_2]):
            ref_lane = lanes[0]
            later_middle = (float(self.map.lane_num) / 2 - 0.5) * self.map.lane_width
            if isinstance(ref_lane, CircularLane) and ref_lane.direction == 1:
                ref_lane = lanes[-1]
                later_middle *= -1
            check_point = ref_lane.position(ref_lane.length, later_middle)
            if lanes_id == 0:
                # calculate ego v lane heading
                lanes_heading.append(ref_lane.heading_at(ref_lane.local_coordinates(ego_vehicle.position)[0]))
            else:
                lanes_heading.append(ref_lane.heading_at(min(self.PRE_NOTIFY_DIST, ref_lane.length)))
            ckpts.append(check_point)
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

        if self.show_navi_point:
            pos_of_goal = ckpts[0]
            self.goal_node_path.setPos(pos_of_goal[0], -pos_of_goal[1], 1.8)
            self.goal_node_path.setH(self.goal_node_path.getH() + 3)
            self.update_navi_arrow(lanes_heading)

        self.navi_info = res
        return lane, lane_index

    def update_navi_arrow(self, lanes_heading):
        lane_0_heading = lanes_heading[0]
        lane_1_heading = lanes_heading[1]
        if abs(lane_0_heading - lane_1_heading) < 0.01:
            if self.showing:
                self.left_arrow.setAlphaScale(self.MIN_ALPHA)
                self.right_arrow.setAlphaScale(self.MIN_ALPHA)
                self.showing = False
        else:
            dir_0 = np.array([np.cos(lane_0_heading), np.sin(lane_0_heading), 0])
            dir_1 = np.array([np.cos(lane_1_heading), np.sin(lane_1_heading), 0])
            cross_product = np.cross(dir_1, dir_0)
            left = False if cross_product[-1] < 0 else True
            if not self.showing:
                self.showing = True
            if left:
                self.left_arrow.setAlphaScale(1)
                self.right_arrow.setAlphaScale(self.MIN_ALPHA)
            else:
                self.right_arrow.setAlphaScale(1)
                self.left_arrow.setAlphaScale(self.MIN_ALPHA)

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

    def destory(self):
        if self.show_navi_point:
            self.arrow_node_path.removeNode()
            self.goal_node_path.removeNode()

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))
