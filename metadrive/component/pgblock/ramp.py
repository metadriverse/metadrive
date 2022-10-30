import math

import numpy as np

from metadrive.component.pgblock.create_pg_block_utils import ExtendStraightLane, CreateRoadFrom, CreateAdverseRoad, \
    create_bend_straight
from metadrive.component.pgblock.pg_block import PGBlock
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.road_network import Road
from metadrive.constants import Decoration, LineType
from metadrive.utils.scene_utils import check_lane_on_road
from metadrive.utils.space import ParameterSpace, Parameter, BlockParameterSpace


class Ramp(PGBlock):
    """
                    InRamp                                             OutRamp
    
     start ----------- end ------------------           start ----------- end ------------------
     start ----------- end ------------------           start ----------- end ------------------              
    (start ----------- end)[----------------]           start ----------- end [----------------]
                       end -----------------}          (start ---------- {end) 
                      //                                                      \\                                                         
    { ---------------//                                                        \\---------------}    
    """
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.RAMP_PARAMETER)
    SOCKET_NUM = 1

    # can be added to parameter space in the future
    RADIUS = 40  # meters
    ANGLE = 10  # degree
    LANE_TYPE = (LineType.CONTINUOUS, LineType.CONTINUOUS)
    SPEED_LIMIT = 12  # 12 m/s ~= 40 km/h
    CONNECT_PART_LEN = 20
    RAMP_LEN = 15


class InRampOnStraight(Ramp):
    ID = "r"
    EXTRA_PART = 10
    SOCKET_LEN = 20

    def _try_plug_into_previous_block(self) -> bool:
        acc_lane_len = self.get_config()[Parameter.length]
        no_cross = True

        # extend road and acc raod part, part 0
        self.set_part_idx(0)
        sin_angle = math.sin(np.deg2rad(self.ANGLE))
        cos_angle = math.cos(np.deg2rad(self.ANGLE))
        longitude_len = sin_angle * self.RADIUS * 2 + cos_angle * self.CONNECT_PART_LEN + self.RAMP_LEN

        extend_lane = ExtendStraightLane(
            self.positive_basic_lane, longitude_len + self.EXTRA_PART, [LineType.BROKEN, LineType.CONTINUOUS]
        )
        extend_road = Road(self.pre_block_socket.positive_road.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            extend_lane,
            self.positive_lane_num,
            extend_road,
            self.block_network,
            self._global_network,
            side_lane_line_type=LineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        extend_road.get_lanes(self.block_network)[-1].line_types = [
            LineType.BROKEN if self.positive_lane_num != 1 else LineType.CONTINUOUS, LineType.CONTINUOUS
        ]
        no_cross = CreateAdverseRoad(
            extend_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        _extend_road = -extend_road
        left_lane_line = LineType.NONE if self.positive_lane_num == 1 else LineType.BROKEN
        _extend_road.get_lanes(self.block_network)[-1].line_types = [left_lane_line, LineType.SIDE]

        # main acc part
        acc_side_lane = ExtendStraightLane(
            extend_lane, acc_lane_len + self.lane_width, [extend_lane.line_types[0], LineType.SIDE]
        )
        acc_road = Road(extend_road.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            acc_side_lane,
            self.positive_lane_num,
            acc_road,
            self.block_network,
            self._global_network,
            side_lane_line_type=LineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        no_cross = CreateAdverseRoad(
            acc_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        left_line_type = LineType.CONTINUOUS if self.positive_lane_num == 1 else LineType.BROKEN
        acc_road.get_lanes(self.block_network)[-1].line_types = [left_line_type, LineType.BROKEN]

        # socket part
        socket_side_lane = ExtendStraightLane(acc_side_lane, self.SOCKET_LEN, acc_side_lane.line_types)
        socket_road = Road(acc_road.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            socket_side_lane,
            self.positive_lane_num,
            socket_road,
            self.block_network,
            self._global_network,
            side_lane_line_type=LineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        no_cross = CreateAdverseRoad(
            socket_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        self.add_sockets(PGBlock.create_socket_from_positive_road(socket_road))

        # ramp part, part 1
        self.set_part_idx(1)
        lateral_dist = (1 - cos_angle) * self.RADIUS * 2 + sin_angle * self.CONNECT_PART_LEN
        end_point = extend_lane.position(self.EXTRA_PART + self.RAMP_LEN, lateral_dist + self.lane_width)
        start_point = extend_lane.position(self.EXTRA_PART, lateral_dist + self.lane_width)
        straight_part = StraightLane(
            start_point, end_point, self.lane_width, self.LANE_TYPE, speed_limit=self.SPEED_LIMIT
        )
        straight_road = Road(self.add_road_node(), self.add_road_node())
        self.block_network.add_lane(straight_road.start_node, straight_road.end_node, straight_part)
        no_cross = (
            not check_lane_on_road(
                self._global_network,
                straight_part,
                0.95,
                ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross
        self.add_respawn_roads(straight_road)

        # p1 road 0, 1
        bend_1, connect_part = create_bend_straight(
            straight_part,
            self.CONNECT_PART_LEN,
            self.RADIUS,
            np.deg2rad(self.ANGLE),
            False,
            self.lane_width,
            self.LANE_TYPE,
            speed_limit=self.SPEED_LIMIT
        )
        bend_1_road = Road(straight_road.end_node, self.add_road_node())
        connect_road = Road(bend_1_road.end_node, self.add_road_node())
        self.block_network.add_lane(bend_1_road.start_node, bend_1_road.end_node, bend_1)
        self.block_network.add_lane(connect_road.start_node, connect_road.end_node, connect_part)
        no_cross = (
            not check_lane_on_road(
                self._global_network, bend_1, 0.95, ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross
        no_cross = (
            not check_lane_on_road(
                self._global_network,
                connect_part,
                0.95,
                ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross

        # p1, road 2, 3
        bend_2, acc_lane = create_bend_straight(
            connect_part,
            acc_lane_len,
            self.RADIUS,
            np.deg2rad(self.ANGLE),
            True,
            self.lane_width,
            self.LANE_TYPE,
            speed_limit=self.SPEED_LIMIT
        )
        acc_lane.line_types = [LineType.BROKEN, LineType.CONTINUOUS]
        bend_2_road = Road(connect_road.end_node, self.road_node(0, 0))  # end at part1 road 0, extend road
        self.block_network.add_lane(bend_2_road.start_node, bend_2_road.end_node, bend_2)
        self.block_network.add_lane(acc_road.start_node, acc_road.end_node, acc_lane)
        no_cross = (
            not check_lane_on_road(
                self._global_network, bend_2, 0.95, ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross
        no_cross = (
            not check_lane_on_road(
                self._global_network, acc_lane, 0.95, ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross

        # p1, road 4, small circular to decorate
        merge_lane, _ = create_bend_straight(
            acc_lane, 10, self.lane_width / 2, np.pi / 2, False, self.lane_width,
            (LineType.BROKEN, LineType.CONTINUOUS)
        )
        self.block_network.add_lane(Decoration.start, Decoration.end, merge_lane)

        return no_cross

    def get_intermediate_spawn_lanes(self):
        """
        Remove lanes on socket
        """
        spawn_lanes = super(InRampOnStraight, self).get_intermediate_spawn_lanes()
        lane_on_socket = self.get_socket(0).get_positive_lanes(self.block_network)[0]
        filtered = []
        for lanes in spawn_lanes:
            if lane_on_socket in lanes:
                continue
            else:
                filtered.append(lanes)
        spawn_lanes = filtered
        assert sum([abs(l.length - InRampOnStraight.RAMP_LEN) <= 0.1 for ls in spawn_lanes for l in ls]) == 1
        return spawn_lanes


class OutRampOnStraight(Ramp):
    ID = "R"
    EXTRA_LEN = 15

    def _get_deacc_lane(self, att_lane: StraightLane):
        start = att_lane.position(self.lane_width, self.lane_width)
        end = att_lane.position(att_lane.length, self.lane_width)
        return StraightLane(start, end, self.lane_width, (LineType.BROKEN, LineType.CONTINUOUS))

    def _get_merge_part(self, side_lane: StraightLane):
        tool_lane = StraightLane(side_lane.end, side_lane.start, side_lane.width)
        merge_part, _ = create_bend_straight(
            tool_lane,
            10,
            self.lane_width / 2,
            np.pi / 2,
            True,
            width=self.lane_width,
            line_types=(LineType.CONTINUOUS, LineType.BROKEN)
        )
        return merge_part

    def _try_plug_into_previous_block(self) -> bool:
        no_cross = True
        sin_angle = math.sin(np.deg2rad(self.ANGLE))
        cos_angle = math.cos(np.deg2rad(self.ANGLE))
        longitude_len = sin_angle * self.RADIUS * 2 + cos_angle * self.CONNECT_PART_LEN + self.RAMP_LEN + self.EXTRA_LEN

        self.set_part_idx(0)
        # part 0 road 0
        dec_lane_len = self.get_config()[Parameter.length]
        dec_lane = ExtendStraightLane(
            self.positive_basic_lane, dec_lane_len + self.lane_width,
            [self.positive_basic_lane.line_types[0], LineType.SIDE]
        )
        dec_road = Road(self.pre_block_socket.positive_road.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            dec_lane,
            self.positive_lane_num,
            dec_road,
            self.block_network,
            self._global_network,
            side_lane_line_type=LineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        no_cross = CreateAdverseRoad(
            dec_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        dec_right_lane = dec_road.get_lanes(self.block_network)[-1]
        left_line_type = LineType.CONTINUOUS if self.positive_lane_num == 1 else LineType.BROKEN
        dec_right_lane.line_types = [left_line_type, LineType.BROKEN]

        # part 0 road 1
        extend_lane = ExtendStraightLane(
            dec_right_lane, longitude_len, [dec_right_lane.line_types[0], LineType.CONTINUOUS]
        )
        extend_road = Road(dec_road.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            extend_lane,
            self.positive_lane_num,
            extend_road,
            self.block_network,
            self._global_network,
            side_lane_line_type=LineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        no_cross = CreateAdverseRoad(
            extend_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        _extend_road = -extend_road
        _extend_road.get_lanes(self.block_network)[-1].line_types = [
            LineType.NONE if self.positive_lane_num == 1 else LineType.BROKEN, LineType.SIDE
        ]
        self.add_sockets(self.create_socket_from_positive_road(extend_road))

        # part 1 road 0
        self.set_part_idx(1)
        dec_side_right_lane = self._get_deacc_lane(dec_right_lane)
        self.block_network.add_lane(dec_road.start_node, dec_road.end_node, dec_side_right_lane)
        no_cross = (
            not check_lane_on_road(
                self._global_network,
                dec_side_right_lane,
                0.95,
                ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross

        bend_1, connect_part = create_bend_straight(
            dec_side_right_lane,
            self.CONNECT_PART_LEN,
            self.RADIUS,
            np.deg2rad(self.ANGLE),
            True,
            self.lane_width,
            self.LANE_TYPE,
            speed_limit=self.SPEED_LIMIT
        )
        bend_1_road = Road(dec_road.end_node, self.add_road_node())
        connect_road = Road(bend_1_road.end_node, self.add_road_node())
        self.block_network.add_lane(bend_1_road.start_node, bend_1_road.end_node, bend_1)
        self.block_network.add_lane(connect_road.start_node, connect_road.end_node, connect_part)
        no_cross = (
            not check_lane_on_road(
                self._global_network, bend_1, 0.95, ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross
        no_cross = (
            not check_lane_on_road(
                self._global_network,
                connect_part,
                0.95,
                ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross

        bend_2, straight_part = create_bend_straight(
            connect_part,
            self.RAMP_LEN,
            self.RADIUS,
            np.deg2rad(self.ANGLE),
            False,
            self.lane_width,
            self.LANE_TYPE,
            speed_limit=self.SPEED_LIMIT
        )
        bend_2_road = Road(connect_road.end_node, self.add_road_node())  # end at part1 road 0, extend road
        straight_road = Road(bend_2_road.end_node, self.add_road_node())
        self.block_network.add_lane(bend_2_road.start_node, bend_2_road.end_node, bend_2)
        self.block_network.add_lane(straight_road.start_node, straight_road.end_node, straight_part)
        no_cross = (
            not check_lane_on_road(
                self._global_network, bend_2, 0.95, ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross
        no_cross = (
            not check_lane_on_road(
                self._global_network,
                straight_part,
                0.95,
                ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross

        decoration_part = self._get_merge_part(dec_side_right_lane)
        self.block_network.add_lane(Decoration.start, Decoration.end, decoration_part)
        return no_cross
