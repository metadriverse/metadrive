import numpy as np

from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.pg_space import Parameter, BlockParameterSpace
from metadrive.component.pgblock.bottleneck import Merge, Split
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.navigation_module.base_navigation import BaseNavigation
from metadrive.utils import clip, norm, get_np_random
from metadrive.utils.math import panda_vector
from metadrive.utils.pg.utils import ray_localization


class NodeNetworkNavigation(BaseNavigation):
    def __init__(
        self,
        show_navi_mark: bool = False,
        show_dest_mark=False,
        show_line_to_dest=False,
        panda_color=None,
        name=None,
        vehicle_config=None
    ):
        """
        This class define a helper for localizing vehicles and retrieving navigation information.
        It now only support from first block start to the end node, but can be extended easily.
        """
        super(NodeNetworkNavigation, self).__init__(
            show_navi_mark=show_navi_mark,
            show_dest_mark=show_dest_mark,
            show_line_to_dest=show_line_to_dest,
            panda_color=panda_color,
            name=name,
            vehicle_config=vehicle_config
        )
        self.final_road = None
        self.spawn_road = None
        self.current_road = None
        self.next_road = None

    def reset(self, vehicle, dest=None, random_seed=None):
        possible_lanes = ray_localization(vehicle.heading, vehicle.spawn_place, self.engine, use_heading_filter=False)
        possible_lane_indexes = [lane_index for lane, lane_index, dist in possible_lanes]

        if len(possible_lanes) == 0 and vehicle.config["spawn_lane_index"] is None:
            from metadrive.utils.error_class import NavigationError
            raise NavigationError("Can't find valid lane for navigation.")

        if vehicle.config["spawn_lane_index"] is not None and vehicle.config["spawn_lane_index"
                                                                             ] in possible_lane_indexes:
            idx = possible_lane_indexes.index(vehicle.config["spawn_lane_index"])
            lane, new_l_index = possible_lanes[idx][:-1]
        else:
            assert len(possible_lanes) > 0
            lane, new_l_index = possible_lanes[0][:-1]

        if dest is None:
            dest = vehicle.config["destination"]

        current_lane = lane
        destination = dest if dest is not None else None
        random_seed = self.engine.global_random_seed if random_seed is None else random_seed
        assert current_lane is not None, "spawn place is not on road!"
        super(NodeNetworkNavigation, self).reset(current_lane)
        assert self.map.road_network_type == NodeRoadNetwork, "This Navigation module only support NodeRoadNetwork type"
        destination = self.auto_assign_task(self.map, current_lane.index, destination, random_seed)
        self.set_route(current_lane.index, destination)

    @staticmethod
    def auto_assign_task(map, current_lane_index, final_road_node=None, random_seed=None):
        # TODO we will assign the route in the task manager in the future
        start_road_node = current_lane_index[0]
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
        return final_road_node

    def set_route(self, current_lane_index: str, destination: str):
        """
        Find the shortest path from start road to the end road.

        Args:
            current_lane_index: start road node
            destination: end road node or end lane index

        Returns:
            None
        """
        self.spawn_road = current_lane_index[:-1]
        self.checkpoints = self.map.road_network.shortest_path(current_lane_index, destination)
        self._target_checkpoints_index = [0, 1]
        # update routing info
        if len(self.checkpoints) <= 2:
            self.checkpoints = [current_lane_index[0], current_lane_index[1]]
            self._target_checkpoints_index = [0, 0]
        assert len(self.checkpoints
                   ) >= 2, "Can not find a route from {} to {}".format(current_lane_index[0], destination)
        self.final_road = Road(self.checkpoints[-2], self.checkpoints[-1])
        final_lanes = self.final_road.get_lanes(self.map.road_network)
        self.final_lane = final_lanes[-1]
        self._navi_info.fill(0.0)
        target_road_1_start = self.checkpoints[0]
        target_road_1_end = self.checkpoints[1]
        self.current_ref_lanes = self.map.road_network.graph[target_road_1_start][target_road_1_end]

        for l in self.current_ref_lanes:
            assert l.index is not None, self.current_ref_lanes

        self.next_ref_lanes = self.map.road_network.graph[self.checkpoints[1]][self.checkpoints[2]
                                                                               ] if len(self.checkpoints) > 2 else None
        self.current_road = Road(target_road_1_start, target_road_1_end)
        self.next_road = Road(self.checkpoints[1], self.checkpoints[2]) if len(self.checkpoints) > 2 else None
        if self._dest_node_path is not None:
            ref_lane = final_lanes[0]
            later_middle = (float(self.get_current_lane_num()) / 2 - 0.5) * self.get_current_lane_width()
            check_point = ref_lane.position(ref_lane.length, later_middle)
            self._dest_node_path.setPos(panda_vector(check_point[0], check_point[1], self.MARK_HEIGHT))

        # Compute the total length of the route for computing route completion
        self.total_length = 0.0
        self.travelled_length = 0.0
        self._last_long_in_ref_lane = 0.0
        for ckpt1, ckpt2 in zip(self.checkpoints[:-1], self.checkpoints[1:]):
            self.total_length += self.map.road_network.graph[ckpt1][ckpt2][0].length

    def update_localization(self, ego_vehicle):
        """
        Update current position, route completion and checkpoints according to current position.

        Args:
            ego_vehicle: a vehicle object

        Returns:
            None
        """
        position = ego_vehicle.position
        lane, lane_index = self._update_current_lane(ego_vehicle)
        long, _ = lane.local_coordinates(position)
        need_update = self._update_target_checkpoints(lane_index, long)
        assert len(self.checkpoints) >= 2

        # Update travelled_length for route completion
        long_in_ref_lane, _ = self.current_ref_lanes[0].local_coordinates(position)
        travelled = long_in_ref_lane - self._last_long_in_ref_lane
        self.travelled_length += travelled
        self._last_long_in_ref_lane = long_in_ref_lane
        # print(f"{self.travelled_length=}, {travelled=}, {long_in_ref_lane=}, "
        #       f"{self.route_completion=}, {self._last_long_in_ref_lane=}")

        # target_road_1 is the road segment the vehicle is driving on.
        if need_update:
            target_road_1_start = self.checkpoints[self._target_checkpoints_index[0]]
            target_road_1_end = self.checkpoints[self._target_checkpoints_index[0] + 1]
            target_lanes_1 = self.map.road_network.graph[target_road_1_start][target_road_1_end]
            self.current_ref_lanes = target_lanes_1
            self.current_road = Road(target_road_1_start, target_road_1_end)

            self._last_long_in_ref_lane = self.current_ref_lanes[0].local_coordinates(position)[0]

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
        half = self.CHECK_POINT_INFO_DIM
        # Put the next checkpoint's information into the first half of the navi_info
        self._navi_info[:half], lanes_heading1, next_checkpoint = self._get_info_for_checkpoint(
            lanes_id=0, ref_lane=self.current_ref_lanes[0], ego_vehicle=ego_vehicle
        )

        # Put the next of the next checkpoint's information into the first half of the navi_info
        self._navi_info[half:], lanes_heading2, next_next_checkpoint = self._get_info_for_checkpoint(
            lanes_id=1,
            ref_lane=self.next_ref_lanes[0] if self.next_ref_lanes is not None else self.current_ref_lanes[0],
            ego_vehicle=ego_vehicle
        )

        self.navi_arrow_dir = [lanes_heading1, lanes_heading2]
        if self._show_navi_info:
            # Whether to visualize little boxes in the scene denoting the checkpoints
            pos_of_goal = next_checkpoint
            self._goal_node_path.setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
            self._goal_node_path.setH(self._goal_node_path.getH() + 3)

            pos_of_goal = next_next_checkpoint
            self._goal_node_path2.setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
            self._goal_node_path2.setH(self._goal_node_path2.getH() + 3)

            dest_pos = self._dest_node_path.getPos()
            self._draw_line_to_dest(start_position=ego_vehicle.position, end_position=(dest_pos[0], dest_pos[1]))
            navi_pos = self._goal_node_path.getPos()
            next_navi_pos = self._goal_node_path2.getPos()
            self._draw_line_to_navi(
                start_position=ego_vehicle.position,
                end_position=(navi_pos[0], navi_pos[1]),
                next_checkpoint=(next_navi_pos[0], next_navi_pos[1])
            )

    def _update_target_checkpoints(self, ego_lane_index, ego_lane_longitude) -> bool:
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

    def _get_current_lane(self, ego_vehicle):
        """
        Called in update_localization to find current lane information. If the vehicle is in the current reference lane,
        meaning it is not yet moving to the next road segment, then return the current reference lane. Otherwise, return
        the next reference lane. If the vehicle is not in any of the reference lanes, then return the closest lane.
        """
        possible_lanes, on_lane = ray_localization(
            ego_vehicle.heading, ego_vehicle.position, ego_vehicle.engine, return_on_lane=True
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

    def _get_info_for_checkpoint(self, lanes_id, ref_lane, ego_vehicle):
        """
        Return the information of checkpoints for state observation.

        Args:
            lanes_id: the lane index of current lane. (lanes is a list so each lane has an index in this list)
            ref_lane: the reference lane.
            ego_vehicle: the vehicle object.

        Returns:
            navi_information, lanes_heading, check_point
        """
        navi_information = []
        # Project the checkpoint position into the target vehicle's coordination, where
        # +x is the heading and +y is the right hand side.
        later_middle = (float(self.get_current_lane_num()) / 2 - 0.5) * self.get_current_lane_width()
        check_point = ref_lane.position(ref_lane.length, later_middle)
        dir_vec = check_point - ego_vehicle.position  # get the vector from center of vehicle to checkpoint
        dir_norm = norm(dir_vec[0], dir_vec[1])
        if dir_norm > self.NAVI_POINT_DIST:  # if the checkpoint is too far then crop the direction vector
            dir_vec = dir_vec / dir_norm * self.NAVI_POINT_DIST
        ckpt_in_heading, ckpt_in_rhs = ego_vehicle.convert_to_local_coordinates(
            dir_vec, 0.0
        )  # project to ego vehicle's coordination

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
            dir = -ref_lane.direction
            angle = ref_lane.angle

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

    def _update_current_lane(self, ego_vehicle):
        lane, lane_index, on_lane = self._get_current_lane(ego_vehicle)
        ego_vehicle.on_lane = on_lane
        if lane is None:
            lane, lane_index = ego_vehicle.lane, ego_vehicle.lane_index
            if self.FORCE_CALCULATE:
                lane_index, _ = self.map.road_network.get_closest_lane_index(ego_vehicle.position)
                lane = self.map.road_network.get_lane(lane_index)
        self._current_lane = lane
        assert lane_index == lane.index, "lane index mismatch!"
        return lane, lane_index

    def get_state(self):
        """Return the navigation information for recording/replaying."""
        final_road = self.final_road
        return {"spawn_road": self.spawn_road, "destination": (final_road.start_node, final_road.end_node)}

    @property
    def route_completion(self):
        """Return the route completion at this moment."""
        return self.travelled_length / self.total_length
