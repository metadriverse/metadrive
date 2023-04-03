import copy

import numpy as np

from metadrive.component.pgblock.create_pg_block_utils import CreateAdverseRoad, CreateRoadFrom, ExtendStraightLane, \
    CreateTwoWayRoad, create_bend_straight
from metadrive.component.pgblock.pg_block import PGBlock, PGBlockSocket
from metadrive.component.road_network import Road
from metadrive.constants import PGLineType, PGLineColor
from metadrive.component.pg_space import ParameterSpace, Parameter, BlockParameterSpace


class ParkingLot(PGBlock):
    """
    Parking Lot
    """

    ID = "P"
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.PARKING_LOT_PARAMETER)
    ANGLE = np.deg2rad(90)
    SOCKET_LENGTH = 4  # m
    SOCKET_NUM = 1

    def _try_plug_into_previous_block(self) -> bool:
        self.spawn_roads = []
        self.dest_roads = []

        no_cross = True
        para = self.get_config()
        assert self.positive_lane_num == 1, "Lane number of previous block must be 1 in each direction"

        self.parking_space_length = para[Parameter.length]
        self.parking_space_width = self.lane_width
        parking_space_num = para[Parameter.one_side_vehicle_num]
        # parking_space_num = 10
        radius = para[Parameter.radius]

        main_straight_road_length = 2 * radius + (parking_space_num - 1) * self.parking_space_width
        main_lane = ExtendStraightLane(
            self.positive_lanes[0], main_straight_road_length, [PGLineType.BROKEN, PGLineType.NONE]
        )
        road = Road(self.pre_block_socket.positive_road.end_node, self.road_node(0, 0))

        # main straight part
        no_cross = CreateRoadFrom(
            main_lane,
            self.positive_lane_num,
            road,
            self.block_network,
            self._global_network,
            center_line_type=PGLineType.BROKEN,
            inner_lane_line_type=PGLineType.BROKEN,
            side_lane_line_type=PGLineType.NONE,
            center_line_color=PGLineColor.GREY,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        no_cross = CreateAdverseRoad(
            road,
            self.block_network,
            self._global_network,
            center_line_type=PGLineType.BROKEN,
            inner_lane_line_type=PGLineType.BROKEN,
            side_lane_line_type=PGLineType.NONE,
            center_line_color=PGLineColor.GREY,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross

        # socket part
        parking_lot_out_lane = ExtendStraightLane(main_lane, self.SOCKET_LENGTH, [PGLineType.BROKEN, PGLineType.NONE])
        parking_lot_out_road = Road(self.road_node(0, 0), self.road_node(0, 1))

        # out socket part
        no_cross = CreateRoadFrom(
            parking_lot_out_lane,
            self.positive_lane_num,
            parking_lot_out_road,
            self.block_network,
            self._global_network,
            center_line_type=PGLineType.BROKEN,
            inner_lane_line_type=PGLineType.BROKEN,
            side_lane_line_type=PGLineType.SIDE,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross

        no_cross = CreateAdverseRoad(
            parking_lot_out_road,
            self.block_network,
            self._global_network,
            center_line_type=PGLineType.BROKEN,
            inner_lane_line_type=PGLineType.BROKEN,
            side_lane_line_type=PGLineType.SIDE,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross

        socket = self.create_socket_from_positive_road(parking_lot_out_road)
        self.add_sockets(socket)

        # add parking space
        for i in range(int(parking_space_num)):
            no_cross = self._add_one_parking_space(
                copy.copy(self.get_socket_list()[0]).get_socket_in_reverse(),
                self.pre_block_socket.get_socket_in_reverse(), i + 1, radius, i * self.parking_space_width,
                (parking_space_num - i - 1) * self.parking_space_width
            ) and no_cross

        for i in range(parking_space_num, 2 * parking_space_num):
            index = i + 1
            i -= parking_space_num
            no_cross = self._add_one_parking_space(
                self.pre_block_socket, copy.copy(self.get_socket_list()[0]), index, radius,
                i * self.parking_space_width, (parking_space_num - i - 1) * self.parking_space_width
            ) and no_cross

        return no_cross

    def _add_one_parking_space(
        self, in_socket: PGBlockSocket, out_socket: PGBlockSocket, part_idx: int, radius, dist_to_in, dist_to_out
    ) -> bool:
        no_cross = True

        # lane into parking space and parking space, 1
        if in_socket.is_same_socket(self.pre_block_socket) or in_socket.is_same_socket(
                self.pre_block_socket.get_socket_in_reverse()):
            net = self._global_network
        else:
            net = self.block_network
        in_lane = in_socket.positive_road.get_lanes(net)[0]
        start_node = in_socket.positive_road.end_node
        if dist_to_in > 1e-3:
            # a straight part will be added
            in_lane = ExtendStraightLane(in_lane, dist_to_in, [PGLineType.NONE, PGLineType.NONE])
            in_road = Road(in_socket.positive_road.end_node, self.road_node(part_idx, 0))
            CreateRoadFrom(
                in_lane,
                self.positive_lane_num,
                in_road,
                self.block_network,
                self._global_network,
                center_line_type=PGLineType.NONE,
                inner_lane_line_type=PGLineType.NONE,
                side_lane_line_type=PGLineType.NONE,
                ignore_intersection_checking=self.ignore_intersection_checking
            )
            start_node = self.road_node(part_idx, 0)

        bend, straight = create_bend_straight(
            in_lane, self.parking_space_length, radius, self.ANGLE, True, self.parking_space_width
        )
        bend_road = Road(start_node, self.road_node(part_idx, 1))
        bend_no_cross = CreateRoadFrom(
            bend,
            self.positive_lane_num,
            bend_road,
            self.block_network,
            self._global_network,
            center_line_type=PGLineType.NONE,
            inner_lane_line_type=PGLineType.NONE,
            side_lane_line_type=PGLineType.SIDE if dist_to_in < 1e-3 else PGLineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )
        if dist_to_in < 1e-3:
            no_cross = no_cross and bend_no_cross

        straight_road = Road(self.road_node(part_idx, 1), self.road_node(part_idx, 2))
        self.dest_roads.append(straight_road)
        no_cross = no_cross and CreateRoadFrom(
            straight,
            self.positive_lane_num,
            straight_road,
            self.block_network,
            self._global_network,
            center_line_type=PGLineType.CONTINUOUS,
            inner_lane_line_type=PGLineType.NONE,
            side_lane_line_type=PGLineType.SIDE if dist_to_in < 1e-3 else PGLineType.NONE,
            center_line_color=PGLineColor.GREY,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        # lane into parking space and parking space, 2
        neg_road: Road = out_socket.negative_road
        if out_socket.is_same_socket(self.pre_block_socket) or out_socket.is_same_socket(
                self.pre_block_socket.get_socket_in_reverse()):
            net = self._global_network
        else:
            net = self.block_network
        neg_lane = \
            neg_road.get_lanes(net)[0]
        start_node = neg_road.end_node
        if dist_to_out > 1e-3:
            # a straight part will be added
            neg_lane = ExtendStraightLane(neg_lane, dist_to_out, [PGLineType.NONE, PGLineType.NONE])
            neg_road = Road(neg_road.end_node, self.road_node(part_idx, 3))
            CreateRoadFrom(
                neg_lane,
                self.positive_lane_num,
                neg_road,
                self.block_network,
                self._global_network,
                center_line_type=PGLineType.NONE,
                inner_lane_line_type=PGLineType.NONE,
                side_lane_line_type=PGLineType.NONE,
                ignore_intersection_checking=self.ignore_intersection_checking
            )
            start_node = self.road_node(part_idx, 3)

        bend, straight = create_bend_straight(
            neg_lane, self.lane_width, radius, self.ANGLE, False, self.parking_space_width
        )
        bend_road = Road(start_node, self.road_node(part_idx, 4))
        CreateRoadFrom(
            bend,
            self.positive_lane_num,
            bend_road,
            self.block_network,
            self._global_network,
            center_line_type=PGLineType.NONE,
            inner_lane_line_type=PGLineType.NONE,
            side_lane_line_type=PGLineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        straight_road = Road(self.road_node(part_idx, 4), self.road_node(part_idx, 1))
        CreateRoadFrom(
            straight,
            self.positive_lane_num,
            straight_road,
            self.block_network,
            self._global_network,
            center_line_type=PGLineType.NONE,
            inner_lane_line_type=PGLineType.NONE,
            side_lane_line_type=PGLineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        # give it a new road name to make it be a two way road (1,2) = (5,6) = parking space !
        parking_road = Road(self.road_node(part_idx, 5), self.road_node(part_idx, 6))
        self.spawn_roads.append(parking_road)
        CreateTwoWayRoad(
            Road(self.road_node(part_idx, 1), self.road_node(part_idx, 2)),
            self.block_network,
            self._global_network,
            parking_road,
            center_line_type=PGLineType.NONE,
            inner_lane_line_type=PGLineType.NONE,
            side_lane_line_type=PGLineType.SIDE if dist_to_out < 1e-3 else PGLineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        # out part
        parking_lane = parking_road.get_lanes(self.block_network)[0]

        # out part 1
        bend, straight = create_bend_straight(
            parking_lane, 0.1 if dist_to_out < 1e-3 else dist_to_out, radius, self.ANGLE, True, parking_lane.width
        )
        out_bend_road = Road(
            self.road_node(part_idx, 6),
            self.road_node(part_idx, 7) if dist_to_out > 1e-3 else out_socket.positive_road.start_node
        )
        bend_success = CreateRoadFrom(
            bend,
            self.positive_lane_num,
            out_bend_road,
            self.block_network,
            self._global_network,
            center_line_type=PGLineType.NONE,
            inner_lane_line_type=PGLineType.NONE,
            side_lane_line_type=PGLineType.SIDE if dist_to_out < 1e-3 else PGLineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )
        if dist_to_out < 1e-3:
            no_cross = no_cross and bend_success

        if dist_to_out > 1e-3:
            out_straight_road = Road(self.road_node(part_idx, 7), out_socket.positive_road.start_node)
            no_cross = no_cross and CreateRoadFrom(
                straight,
                self.positive_lane_num,
                out_straight_road,
                self.block_network,
                self._global_network,
                center_line_type=PGLineType.NONE,
                inner_lane_line_type=PGLineType.NONE,
                side_lane_line_type=PGLineType.NONE,
                ignore_intersection_checking=self.ignore_intersection_checking
            )
        # out part 2
        extend_lane = ExtendStraightLane(parking_lane, self.lane_width, [PGLineType.NONE, PGLineType.NONE])
        CreateRoadFrom(
            extend_lane,
            self.positive_lane_num,
            Road(self.road_node(part_idx, 6), self.road_node(part_idx, 8)),
            self.block_network,
            self._global_network,
            center_line_type=PGLineType.NONE,
            inner_lane_line_type=PGLineType.NONE,
            side_lane_line_type=PGLineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        bend, straight = create_bend_straight(
            extend_lane, 0.1 if dist_to_in < 1e-3 else dist_to_in, radius, self.ANGLE, False, parking_lane.width
        )
        out_bend_road = Road(
            self.road_node(part_idx, 8),
            self.road_node(part_idx, 9) if dist_to_in > 1e-3 else in_socket.negative_road.start_node
        )
        CreateRoadFrom(
            bend,
            self.positive_lane_num,
            out_bend_road,
            self.block_network,
            self._global_network,
            center_line_type=PGLineType.NONE,
            inner_lane_line_type=PGLineType.NONE,
            side_lane_line_type=PGLineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )
        if dist_to_in > 1e-3:
            out_straight_road = Road(self.road_node(part_idx, 9), in_socket.negative_road.start_node)
            CreateRoadFrom(
                straight,
                self.positive_lane_num,
                out_straight_road,
                self.block_network,
                self._global_network,
                center_line_type=PGLineType.NONE,
                inner_lane_line_type=PGLineType.NONE,
                side_lane_line_type=PGLineType.NONE,
                ignore_intersection_checking=self.ignore_intersection_checking
            )

        return no_cross

    @staticmethod
    def in_direction_parking_space(road: Road):
        """
        Give a parking space in out-direction, return in direction road
        """
        start_node = copy.deepcopy(road.start_node)
        end_node = copy.deepcopy(road.end_node)
        assert start_node[-2] == "5" and end_node[
            -2] == "6", "It is not out-direction of this parking space, start_node:{}, end_node:{}".format(
                start_node, end_node
            )
        start_node = start_node[:-2] + "1" + PGBlock.DASH
        end_node = end_node[:-2] + "2" + PGBlock.DASH
        return Road(start_node, end_node)

    @staticmethod
    def out_direction_parking_space(road: Road):
        """
        Give a parking space in in-direction, return out-direction road
        """
        start_node = copy.deepcopy(road.start_node)
        end_node = copy.deepcopy(road.end_node)
        assert start_node[-2] == "1" and end_node[
            -2] == "2", "It is not in-direction of this parking space, start_node:{}, end_node:{}".format(
                start_node, end_node
            )
        start_node = start_node[:-2] + "5" + PGBlock.DASH
        end_node = end_node[:-2] + "6" + PGBlock.DASH
        return Road(start_node, end_node)

    @staticmethod
    def is_out_direction_parking_space(road: Road):
        start_node = road.start_node
        end_node = road.end_node
        assert (start_node[-2] == "1"
                and end_node[-2] == "2") or (start_node[-2] == "5" and end_node[-2]
                                             == "6"), "{} to {} is not parking space".format(start_node, end_node)
        if start_node[-2] == "5" and end_node[-2] == "6":
            return True
        else:
            return False

    @staticmethod
    def is_in_direction_parking_space(road: Road):
        start_node = road.start_node
        end_node = road.end_node
        assert (start_node[-2] == "1"
                and end_node[-2] == "2") or (start_node[-2] == "5" and end_node[-2]
                                             == "6"), "{} to {} is not parking space".format(start_node, end_node)
        if start_node[-2] == "1" and end_node[-2] == "2":
            return True
        else:
            return False
