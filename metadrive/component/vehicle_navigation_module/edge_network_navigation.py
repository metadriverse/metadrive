from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.component.vehicle_navigation_module.base_navigation import BaseNavigation


class EdgeNetworkNavigation(BaseNavigation):
    """
   This class define a helper for localizing vehicles and retrieving navigation information.
   It now only support EdgeRoadNetwork
   """

    def __init__(
            self,
            engine,
            show_navi_mark: bool = False,
            random_navi_mark_color=False,
            show_dest_mark=False,
            show_line_to_dest=False
    ):
        super(EdgeNetworkNavigation,
              self).__init__(engine, show_navi_mark, random_navi_mark_color, show_dest_mark, show_line_to_dest)

    def reset(self, map, current_lane, destination=None, random_seed=None):
        super(EdgeNetworkNavigation, self).reset(map, current_lane)
        assert self.map.road_network_type == EdgeRoadNetwork, "This Navigation module only support EdgeRoadNetwork type"
        self.set_route(current_lane.index, destination)

    def set_route(self, current_lane_index: str, destination: str):
        """
        Find a shortest path from start road to end road
        :param current_lane_index: start road node
        :param destination: end road node or end lane index
        :return: None
        """
        self.checkpoints = self.map.road_network.shortest_path(current_lane_index, destination)
        # if len(self.checkpoints) == 0:
        #     self.checkpoints.append(current_lane_index)
        #     self.checkpoints.append(current_lane_index)
        self._target_checkpoints_index = [0, 1]
        # update routing info
        assert len(self.checkpoints) > 0, "Can not find a route from {} to {}".format(current_lane_index, destination)
        self.final_lane = self.map.road_network.get_lane(self.checkpoints[-1])
        self._navi_info.fill(0.0)
        self.current_ref_lanes = self.map.road_network.get_peer_lanes_from_index(self.current_checkpoint_lane_index)
        self.next_ref_lanes = self.map.road_network.get_peer_lanes_from_index(self.next_checkpoint_lane_index)
        if self._dest_node_path is not None:
            ref_lane = self.final_lane
            later_middle = (float(self.get_current_lane_num()) / 2 - 0.5) * self.get_current_lane_width()
            check_point = ref_lane.position(ref_lane.length, later_middle)
            self._dest_node_path.setPos(check_point[0], -check_point[1], 1.8)

    def update_localization(self, ego_vehicle):
        position = ego_vehicle.position
        lane, lane_index = self._update_current_lane(ego_vehicle)
        long, _ = lane.local_coordinates(position)
        need_update = self._update_target_checkpoints(lane_index, long)

        # target_road_1 is the road segment the vehicle is driving on.
        if need_update:
            self.current_ref_lanes = self.map.road_network.get_peer_lanes_from_index(self.current_checkpoint_lane_index)

            # target_road_2 is next road segment the vehicle should drive on.
            self.next_ref_lanes = self.map.road_network.get_peer_lanes_from_index(self.next_checkpoint_lane_index)

            if self.current_checkpoint_lane_index == self.next_checkpoint_lane_index:
                # When we are in the final road segment that there is no further road to drive on
                self.next_ref_lanes = None

        self._navi_info.fill(0.0)
        half = self.navigation_info_dim // 2
        self._navi_info[:half], lanes_heading1, checkpoint = self._get_info_for_checkpoint(
            lanes_id=0, ref_lane=self.current_lane, ego_vehicle=ego_vehicle
        )

        self._navi_info[half:], lanes_heading2, _ = self._get_info_for_checkpoint(
            lanes_id=1,
            ref_lane=self.map.road_network.get_lane(self.next_checkpoint_lane_index),
            ego_vehicle=ego_vehicle
        )

        if self._show_navi_info:  # Whether to visualize little boxes in the scene denoting the checkpoints
            pos_of_goal = checkpoint
            self._goal_node_path.setPos(pos_of_goal[0], -pos_of_goal[1], 1.8)
            self._goal_node_path.setH(self._goal_node_path.getH() + 3)
            self.navi_arrow_dir = [lanes_heading1, lanes_heading2]
            dest_pos = self._dest_node_path.getPos()
            self._draw_line_to_dest(start_position=ego_vehicle.position, end_position=(dest_pos[0], -dest_pos[1]))

    def _update_target_checkpoints(self, ego_lane_index, ego_lane_longitude) -> bool:
        """
        update the checkpoint, return True if updated else False
        """
        if self.current_checkpoint_lane_index == self.next_checkpoint_lane_index:  # on last road
            return False

        # arrive to second checkpoint
        new_index = ego_lane_index
        if new_index in self.checkpoints[self._target_checkpoints_index[1]:] \
                and ego_lane_longitude < self.CKPT_UPDATE_RANGE:
            idx = self.checkpoints.index(new_index, self._target_checkpoints_index[1])
            self._target_checkpoints_index = [idx]
            if idx + 1 == len(self.checkpoints):
                self._target_checkpoints_index.append(idx)
            else:
                self._target_checkpoints_index.append(idx + 1)
            return True
        return False

    def get_current_lateral_range(self, current_position, engine) -> float:
        return self.current_lane.width * len(self.current_ref_lanes)

    @property
    def current_checkpoint_lane_index(self):
        return self.checkpoints[self._target_checkpoints_index[0]]

    @property
    def next_checkpoint_lane_index(self):
        return self.checkpoints[self._target_checkpoints_index[1]]
