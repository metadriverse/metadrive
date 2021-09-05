import logging

import numpy as np
from panda3d.core import TransparencyAttrib, LineSegs, NodePath

from metadrive.component.blocks.bottleneck import Merge, Split
from metadrive.component.blocks.first_block import FirstPGBlock
from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.road.road import Road
from metadrive.constants import RENDER_MODE_ONSCREEN, CamMask
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils import clip, norm, get_np_random
from metadrive.utils.coordinates_shift import panda_position
from metadrive.utils.scene_utils import ray_localization
from metadrive.utils.space import Parameter, BlockParameterSpace


class Navigation:
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
        show_line_to_dest=False
    ):
        """
        This class define a helper for localizing vehicles and retrieving navigation information.
        It now only support from first block start to the end node, but can be extended easily.
        """
        self.map = None
        self.final_road = None
        self.final_lane = None
        self.checkpoints = None
        self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.current_road = None
        self.next_road = None
        self._target_checkpoints_index = None
        self._navi_info = np.zeros((self.navigation_info_dim, ))  # navi information res

        # Vis
        self._show_navi_info = (engine.mode == RENDER_MODE_ONSCREEN and not engine.global_config["debug_physics_world"])
        self.origin = NodePath("navigation_sign") if self._show_navi_info else None
        self.navi_mark_color = (0.6, 0.8, 0.5) if not random_navi_mark_color else get_np_random().rand(3)
        self.navi_arrow_dir = None
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
                line_seg.setColor(self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.7)
                line_seg.setThickness(2)
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

    def update(self, map: BaseMap, current_lane_index, final_road_node=None, random_seed=None):
        # TODO(pzh): We should not determine the destination of a vehicle in the navigation module.
        start_road_node = current_lane_index[0]
        self.map = map
        if start_road_node is None:
            start_road_node = FirstPGBlock.NODE_1
        if final_road_node is None:
            current_road_negative = Road(*current_lane_index[:-1]).is_negative_road()
            # choose first block when born on negative road
            block = map.blocks[0] if current_road_negative else map.blocks[-1]
            sockets = block.get_socket_list()
            socket = get_np_random(random_seed).choice(sockets)
            while True:
                if not socket.is_socket_node(start_road_node) or len(sockets) == 1:
                    break
                else:
                    sockets.remove(socket)
                    if len(sockets) == 0:
                        raise ValueError("Can not set a destination!")
            # choose negative road end node when current road is negative road
            final_road_node = socket.negative_road.end_node if current_road_negative else socket.positive_road.end_node
        self.set_route(current_lane_index, final_road_node)

    def set_route(self, current_lane_index: str, end_road_node: str):
        """
        Find a shortest path from start road to end road
        :param current_lane_index: start road node
        :param end_road_node: end road node
        :return: None
        """
        start_road_node = current_lane_index[0]
        self.checkpoints = self.map.road_network.shortest_path(start_road_node, end_road_node)
        self._target_checkpoints_index = [0, 1]
        # update routing info
        if len(self.checkpoints) <= 2:
            self.checkpoints = [current_lane_index[0], current_lane_index[1]]
            self._target_checkpoints_index = [0, 0]
        assert len(self.checkpoints) >= 2, "Can not find a route from {} to {}".format(start_road_node, end_road_node)
        self.final_road = Road(self.checkpoints[-2], self.checkpoints[-1])
        final_lanes = self.final_road.get_lanes(self.map.road_network)
        self.final_lane = final_lanes[-1]
        self._navi_info.fill(0.0)
        target_road_1_start = self.checkpoints[0]
        target_road_1_end = self.checkpoints[1]
        self.current_ref_lanes = self.map.road_network.graph[target_road_1_start][target_road_1_end]
        self.next_ref_lanes = self.map.road_network.graph[self.checkpoints[1]][self.checkpoints[2]
                                                                               ] if len(self.checkpoints) > 2 else None
        self.current_road = Road(target_road_1_start, target_road_1_end)
        self.next_road = Road(self.checkpoints[1], self.checkpoints[2]) if len(self.checkpoints) > 2 else None
        if self._dest_node_path is not None:
            ref_lane = final_lanes[0]
            later_middle = (float(self.get_current_lane_num()) / 2 - 0.5) * self.get_current_lane_width()
            check_point = ref_lane.position(ref_lane.length, later_middle)
            self._dest_node_path.setPos(check_point[0], -check_point[1], 1.8)

    def update_localization(self, ego_vehicle):
        position = ego_vehicle.position
        lane, lane_index = self.get_current_lane(ego_vehicle)
        if lane is None:
            lane, lane_index = ego_vehicle.lane, ego_vehicle.lane_index
            ego_vehicle.on_lane = False
            if self.FORCE_CALCULATE:
                lane_index, _ = self.map.road_network.get_closest_lane_index(position)
                lane = self.map.road_network.get_lane(lane_index)
        long, _ = lane.local_coordinates(position)
        self._update_target_checkpoints(lane_index, long)

        assert len(self.checkpoints) >= 2

        # target_road_1 is the road segment the vehicle is driving on.
        target_road_1_start = self.checkpoints[self._target_checkpoints_index[0]]
        target_road_1_end = self.checkpoints[self._target_checkpoints_index[0] + 1]
        target_lanes_1 = self.map.road_network.graph[target_road_1_start][target_road_1_end]
        self.current_ref_lanes = target_lanes_1
        self.current_road = Road(target_road_1_start, target_road_1_end)

        # target_road_2 is next road segment the vehicle should drive on.
        target_road_2_start = self.checkpoints[self._target_checkpoints_index[1]]
        target_road_2_end = self.checkpoints[self._target_checkpoints_index[1] + 1]
        target_lanes_2 = self.map.road_network.graph[target_road_2_start][target_road_2_end]

        if target_road_1_start == target_road_2_start:
            # When we are in the final road segment that there is no further road to drive on
            self.next_road = None
            self.next_ref_lanes = None
        else:
            self.next_road = Road(target_road_2_start, target_road_2_end)
            self.next_ref_lanes = target_lanes_2

        self._navi_info.fill(0.0)
        half = self.navigation_info_dim // 2
        self._navi_info[:half], lanes_heading1, checkpoint = self._get_info_for_checkpoint(
            lanes_id=0, lanes=target_lanes_1, ego_vehicle=ego_vehicle
        )

        self._navi_info[half:], lanes_heading2, _ = self._get_info_for_checkpoint(
            lanes_id=1, lanes=target_lanes_2, ego_vehicle=ego_vehicle
        )

        if self._show_navi_info:  # Whether to visualize little boxes in the scene denoting the checkpoints
            pos_of_goal = checkpoint
            self._goal_node_path.setPos(pos_of_goal[0], -pos_of_goal[1], 1.8)
            self._goal_node_path.setH(self._goal_node_path.getH() + 3)
            self.navi_arrow_dir = [lanes_heading1, lanes_heading2]
            dest_pos = self._dest_node_path.getPos()
            self._draw_line_to_dest(
                engine=ego_vehicle.engine,
                start_position=ego_vehicle.position,
                end_position=(dest_pos[0], -dest_pos[1])
            )

        return lane, lane_index

    def _get_info_for_checkpoint(self, lanes_id, lanes, ego_vehicle):

        navi_information = []

        # Project the checkpoint position into the target vehicle's coordination, where
        # +x is the heading and +y is the right hand side.
        ref_lane = lanes[0]
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
        """
        Return should_update: True or False
        """
        if self._target_checkpoints_index[0] == self._target_checkpoints_index[1]:  # on last road
            return

        # arrive to second checkpoint
        current_road_start_point = ego_lane_index[0]
        if current_road_start_point in self.checkpoints[self._target_checkpoints_index[1]:] \
                and ego_lane_longitude < self.CKPT_UPDATE_RANGE:
            if current_road_start_point not in self.checkpoints[self._target_checkpoints_index[1]:-1]:
                return
            idx = self.checkpoints.index(current_road_start_point, self._target_checkpoints_index[1], -1)
            self._target_checkpoints_index = [idx]
            if idx + 1 == len(self.checkpoints) - 1:
                self._target_checkpoints_index.append(idx)
            else:
                self._target_checkpoints_index.append(idx + 1)
            return
        return

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
        self.next_road = None
        self.current_road = None
        self.next_ref_lanes = None
        self.current_ref_lanes = None

    def set_force_calculate_lane_index(self, force: bool):
        self.FORCE_CALCULATE = force

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def get_current_lateral_range(self, current_position, engine) -> float:
        """Return the maximum lateral distance from left to right."""
        # special process for special block
        try:
            current_block_id = self.current_road.block_ID()
        except AttributeError:
            return self.get_current_lane_width() * self.get_current_lane_num()
        if current_block_id == Split.ID or current_block_id == Merge.ID:
            left_lane = self.current_ref_lanes[0]
            assert isinstance(left_lane, StraightLane), "Reference lane should be straight lane here"
            long, lat = left_lane.local_coordinates(current_position)
            current_position = left_lane.position(long, -left_lane.width / 2)
            return self._ray_lateral_range(engine, current_position, self.current_ref_lanes[0].direction_lateral)
        else:
            return self.get_current_lane_width() * self.get_current_lane_num()

    def get_current_lane_width(self) -> float:
        return self.map._config[self.map.LANE_WIDTH]

    def get_current_lane_num(self) -> float:
        return len(self.current_ref_lanes)

    def get_current_lane(self, ego_vehicle):
        possible_lanes = ray_localization(
            ego_vehicle.heading, ego_vehicle.position, ego_vehicle.engine, return_all_result=True
        )
        for lane, index, l_1_dist in possible_lanes:
            if lane in self.current_ref_lanes:
                return lane, index
        nx_ckpt = self._target_checkpoints_index[-1]
        if nx_ckpt == self.checkpoints[-1] or self.next_road is None:
            return possible_lanes[0][:-1] if len(possible_lanes) > 0 else (None, None)

        nx_nx_ckpt = nx_ckpt + 1
        next_ref_lanes = self.map.road_network.graph[self.checkpoints[nx_ckpt]][self.checkpoints[nx_nx_ckpt]]
        for lane, index, l_1_dist in possible_lanes:
            if lane in next_ref_lanes:
                return lane, index
        return possible_lanes[0][:-1] if len(possible_lanes) > 0 else (None, None)

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

    def _draw_line_to_dest(self, engine, start_position, end_position):
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
