import numpy as np

from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.component.vehicle_navigation_module.base_navigation import BaseNavigation
from metadrive.utils import clip, norm
from metadrive.utils.scene_utils import ray_localization
from metadrive.utils.space import Parameter, BlockParameterSpace
from metadrive.utils.scene_utils import ray_localization


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
        show_line_to_dest=False,
        panda_color=None
    ):
        super(EdgeNetworkNavigation, self).__init__(
            engine, show_navi_mark, random_navi_mark_color, show_dest_mark, show_line_to_dest, panda_color=panda_color
        )

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
        need_update = self._update_target_checkpoints(lane_index)

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
            lanes_id=0,
            ref_lane=self.map.road_network.get_lane(self.current_checkpoint_lane_index),
            ego_vehicle=ego_vehicle
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

    def _update_target_checkpoints(self, ego_lane_index) -> bool:
        """
        update the checkpoint, return True if updated else False
        """
        if self.current_checkpoint_lane_index == self.next_checkpoint_lane_index:  # on last road
            return False

        # arrive to second checkpoint
        new_index = ego_lane_index
        if new_index in self.checkpoints[self._target_checkpoints_index[1]:]:
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

    def _get_current_lane(self, ego_vehicle):
        """
        Called in update_localization to find current lane information
        """
        possible_lanes, on_lane = ray_localization(
            ego_vehicle.heading,
            ego_vehicle.position,
            ego_vehicle.engine,
            return_all_result=True,
            use_heading_filter=False,
            return_on_lane=True
        )
        for lane, index, l_1_dist in possible_lanes:
            if lane in self.current_ref_lanes:
                return lane, index, on_lane
        nx_ckpt = self._target_checkpoints_index[-1]
        if nx_ckpt == self.checkpoints[-1] or self.next_ref_lanes is None:
            return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)

        next_ref_lanes = self.next_ref_lanes
        for lane, index, l_1_dist in possible_lanes:
            if lane in next_ref_lanes:
                return lane, index, on_lane
        return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)

    def _get_info_for_checkpoint(self, lanes_id, ref_lane, ego_vehicle):

        navi_information = []
        # Project the checkpoint position into the target vehicle's coordination, where
        # +x is the heading and +y is the right hand side.
        check_point = ref_lane.end
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
