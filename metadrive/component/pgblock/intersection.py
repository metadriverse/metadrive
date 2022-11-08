import copy
from collections import deque

import numpy as np

from metadrive.component.pgblock.create_pg_block_utils import CreateAdverseRoad, CreateRoadFrom, ExtendStraightLane, \
    create_bend_straight
from metadrive.component.pgblock.pg_block import PGBlock, PGBlockSocket
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.road_network import Road
from metadrive.constants import LineType
from metadrive.utils.scene_utils import check_lane_on_road
from metadrive.utils.space import ParameterSpace, Parameter, BlockParameterSpace


class InterSection(PGBlock):
    """
                                up(Goal:1)
                                   ||
                                   ||
                                   ||
                                   ||
                                   ||
                  _________________||_________________
    left(Goal:2)  -----------------||---------------- right(Goal:0)
                               __  ||
                              |    ||
             spawn_road    <--|    ||
                              |    ||
                              |__  ||
                                  down
    It's an Intersection with two lanes on same direction, 4 lanes on both roads
    """

    ID = "X"
    EXTRA_PART = "extra"
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.INTERSECTION)
    SOCKET_NUM = 3
    ANGLE = 90  # may support other angle in the future
    EXIT_PART_LENGTH = 35

    _enable_u_turn_flag = False

    # LEFT_TURN_NUM = 1 now it is useless

    def __init__(self, *args, **kwargs):
        if "radius" in kwargs:
            self.radius = kwargs.pop("radius")
        else:
            self.radius = None
        super(InterSection, self).__init__(*args, **kwargs)
        if self.radius is None:
            self.radius = self.get_config()[Parameter.radius]

    def _try_plug_into_previous_block(self) -> bool:
        para = self.get_config()
        decrease_increase = -1 if para[Parameter.decrease_increase] == 0 else 1
        if self.positive_lane_num <= 1:
            decrease_increase = 1
        elif self.positive_lane_num >= 4:
            decrease_increase = -1
        self.lane_num_intersect = self.positive_lane_num + decrease_increase * para[Parameter.change_lane_num]
        no_cross = True
        attach_road = self.pre_block_socket.positive_road
        _attach_road = self.pre_block_socket.negative_road
        attach_lanes = attach_road.get_lanes(self._global_network)
        # right straight left node name, rotate it to fit different part
        intersect_nodes = deque(
            [self.road_node(0, 0),
             self.road_node(1, 0),
             self.road_node(2, 0), _attach_road.start_node]
        )

        for i in range(4):
            right_lane, success = self._create_part(attach_lanes, attach_road, self.radius, intersect_nodes, i)
            no_cross = no_cross and success
            if i != 3:
                lane_num = self.positive_lane_num if i == 1 else self.lane_num_intersect
                exit_road = Road(self.road_node(i, 0), self.road_node(i, 1))
                no_cross = CreateRoadFrom(
                    right_lane,
                    lane_num,
                    exit_road,
                    self.block_network,
                    self._global_network,
                    ignore_intersection_checking=self.ignore_intersection_checking
                ) and no_cross
                no_cross = CreateAdverseRoad(
                    exit_road,
                    self.block_network,
                    self._global_network,
                    ignore_intersection_checking=self.ignore_intersection_checking
                ) and no_cross
                socket = PGBlockSocket(exit_road, -exit_road)
                self.add_respawn_roads(socket.negative_road)
                self.add_sockets(socket)
                attach_road = -exit_road
                attach_lanes = attach_road.get_lanes(self.block_network)
        return no_cross

    def _create_part(self, attach_lanes, attach_road: Road, radius: float, intersect_nodes: deque,
                     part_idx) -> (StraightLane, bool):
        lane_num = self.lane_num_intersect if part_idx == 0 or part_idx == 2 else self.positive_lane_num
        non_cross = True
        attach_left_lane = attach_lanes[0]
        # first left part
        assert isinstance(attach_left_lane, StraightLane), "Can't create a intersection following a circular lane"
        self._create_left_turn(radius, lane_num, attach_left_lane, attach_road, intersect_nodes, part_idx)

        # u-turn
        if self._enable_u_turn_flag:
            adverse_road = -attach_road
            self._create_u_turn(attach_road, part_idx)

        # go forward part
        lanes_on_road = copy.deepcopy(attach_lanes)
        straight_lane_len = 2 * radius + (2 * lane_num - 1) * lanes_on_road[0].width_at(0)
        for l in lanes_on_road:
            next_lane = ExtendStraightLane(l, straight_lane_len, (LineType.NONE, LineType.NONE))
            self.block_network.add_lane(attach_road.end_node, intersect_nodes[1], next_lane)

        # right part
        length = self.EXIT_PART_LENGTH
        right_turn_lane = lanes_on_road[-1]
        assert isinstance(right_turn_lane, StraightLane), "Can't create a intersection following a circular lane"
        right_bend, right_straight = create_bend_straight(
            right_turn_lane, length, radius, np.deg2rad(self.ANGLE), True, right_turn_lane.width_at(0),
            (LineType.NONE, LineType.SIDE)
        )

        non_cross = (
            not check_lane_on_road(
                self._global_network, right_bend, 1, ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and non_cross
        CreateRoadFrom(
            right_bend,
            min(self.positive_lane_num, self.lane_num_intersect),
            Road(attach_road.end_node, intersect_nodes[0]),
            self.block_network,
            self._global_network,
            toward_smaller_lane_index=True,
            side_lane_line_type=LineType.SIDE,
            inner_lane_line_type=LineType.NONE,
            center_line_type=LineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        intersect_nodes.rotate(-1)
        right_straight.line_types = [LineType.BROKEN, LineType.SIDE]
        return right_straight, non_cross

    def get_socket(self, index: int) -> PGBlockSocket:
        socket = super(InterSection, self).get_socket(index)
        if socket.negative_road in self.get_respawn_roads():
            self._respawn_roads.remove(socket.negative_road)
        return socket

    def _create_left_turn(self, radius, lane_num, attach_left_lane, attach_road, intersect_nodes, part_idx):
        left_turn_radius = radius + lane_num * attach_left_lane.width_at(0)
        diff = self.lane_num_intersect - self.positive_lane_num  # increase lane num
        if ((part_idx == 1 or part_idx == 3) and diff > 0) or ((part_idx == 0 or part_idx == 2) and diff < 0):
            diff = abs(diff)
            left_bend, extra_part = create_bend_straight(
                attach_left_lane, self.lane_width * diff, left_turn_radius, np.deg2rad(self.ANGLE), False,
                attach_left_lane.width_at(0), (LineType.NONE, LineType.NONE)
            )
            left_road_start = intersect_nodes[2]
            pre_left_road_start = left_road_start + self.EXTRA_PART
            CreateRoadFrom(
                left_bend,
                min(self.positive_lane_num, self.lane_num_intersect),
                Road(attach_road.end_node, pre_left_road_start),
                self.block_network,
                self._global_network,
                toward_smaller_lane_index=False,
                center_line_type=LineType.NONE,
                side_lane_line_type=LineType.NONE,
                inner_lane_line_type=LineType.NONE,
                ignore_intersection_checking=self.ignore_intersection_checking
            )

            CreateRoadFrom(
                extra_part,
                min(self.positive_lane_num, self.lane_num_intersect),
                Road(pre_left_road_start, left_road_start),
                self.block_network,
                self._global_network,
                toward_smaller_lane_index=False,
                center_line_type=LineType.NONE,
                side_lane_line_type=LineType.NONE,
                inner_lane_line_type=LineType.NONE,
                ignore_intersection_checking=self.ignore_intersection_checking
            )

        else:
            left_bend, _ = create_bend_straight(
                attach_left_lane, self.EXIT_PART_LENGTH, left_turn_radius, np.deg2rad(self.ANGLE), False,
                attach_left_lane.width_at(0), (LineType.NONE, LineType.NONE)
            )
            left_road_start = intersect_nodes[2]
            CreateRoadFrom(
                left_bend,
                min(self.positive_lane_num, self.lane_num_intersect),
                Road(attach_road.end_node, left_road_start),
                self.block_network,
                self._global_network,
                toward_smaller_lane_index=False,
                center_line_type=LineType.NONE,
                side_lane_line_type=LineType.NONE,
                inner_lane_line_type=LineType.NONE,
                ignore_intersection_checking=self.ignore_intersection_checking
            )

    def _create_u_turn(self, attach_road, part_idx):
        # set to CONTINUOUS to debug
        line_type = LineType.NONE
        lanes = attach_road.get_lanes(self.block_network) if part_idx != 0 else self.positive_lanes
        attach_left_lane = lanes[0]
        lane_num = len(lanes)
        left_turn_radius = self.lane_width / 2
        left_bend, _ = create_bend_straight(
            attach_left_lane, 0.1, left_turn_radius, np.deg2rad(180), False, attach_left_lane.width_at(0),
            (LineType.NONE, LineType.NONE)
        )
        left_road_start = (-attach_road).start_node
        CreateRoadFrom(
            left_bend,
            lane_num,
            Road(attach_road.end_node, left_road_start),
            self.block_network,
            self._global_network,
            toward_smaller_lane_index=False,
            center_line_type=line_type,
            side_lane_line_type=line_type,
            inner_lane_line_type=line_type,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

    def enable_u_turn(self, enable_u_turn: bool):
        self._enable_u_turn_flag = enable_u_turn

    def get_intermediate_spawn_lanes(self):
        """Override this function for intersection so that we won't spawn vehicles in the center of intersection."""
        respawn_lanes = self.get_respawn_lanes()
        return respawn_lanes
