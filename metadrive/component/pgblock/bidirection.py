from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.pgblock.create_pg_block_utils import get_lanes_on_road, CreateRoadFrom, ExtendStraightLane
from metadrive.component.pgblock.pg_block import PGBlock, PGBlockSocket
from metadrive.component.road_network import Road
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.constants import PGLineType, PGLineColor
from metadrive.component.pg_space import ParameterSpace, Parameter, BlockParameterSpace


def create_overlap_road(
    positive_road: "Road",
    roadnet_to_get_road: "NodeRoadNetwork",  # mostly, block network
    roadnet_to_check_cross: "NodeRoadNetwork",  # mostly, previous global network
    ignore_start: str = None,
    ignore_end: str = None,
    center_line_type=PGLineType.CONTINUOUS,  # Identical to Block.CENTER_LINE_TYPE
    side_lane_line_type=PGLineType.SIDE,
    inner_lane_line_type=PGLineType.BROKEN,
    center_line_color=PGLineColor.YELLOW,
    ignore_intersection_checking=None
) -> bool:
    """
    Create overlap lanes
    """
    adverse_road = -positive_road
    lanes = get_lanes_on_road(positive_road, roadnet_to_get_road)
    reference_lane = lanes[-1]
    num = len(lanes) * 2
    width = lanes[-1].width_at(0)
    if isinstance(reference_lane, StraightLane):
        start_point = reference_lane.position(lanes[-1].length, 0)
        end_point = reference_lane.position(0, 0)
        symmetric_lane = StraightLane(
            start_point, end_point, width, lanes[-1].line_types, reference_lane.forbidden, reference_lane.speed_limit,
            reference_lane.priority
        )
    else:
        raise ValueError("Creating other lanes is not supported")
    symmetric_lane.line_colors = [PGLineColor.GREY, PGLineColor.GREY]
    success = CreateRoadFrom(
        symmetric_lane,
        int(num / 2),
        adverse_road,
        roadnet_to_get_road,
        roadnet_to_check_cross,
        ignore_start=ignore_start,
        ignore_end=ignore_end,
        side_lane_line_type=side_lane_line_type,
        inner_lane_line_type=inner_lane_line_type,
        center_line_type=center_line_type,
        center_line_color=center_line_color,
        ignore_intersection_checking=ignore_intersection_checking
    )
    positive_road.get_lanes(roadnet_to_get_road)[0].line_colors = [center_line_color, PGLineColor.GREY]
    return success


class BidirectionSocket(PGBlockSocket):
    def __init__(self, fake_positive_lane, fake_negative_lane, positive_road: Road, negative_road: Road = None):
        super(BidirectionSocket, self).__init__(positive_road, negative_road)
        self.fake_positive_lane = fake_positive_lane
        self.fake_negative_lane = fake_negative_lane

    def get_positive_lanes(self, global_network):
        return [self.fake_positive_lane]

    def get_negative_lanes(self, global_network):
        return [self.fake_negative_lane]


class Bidirection(PGBlock):
    ID = "B"
    SOCKET_NUM = 1
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.BIDIRECTION)

    def _try_plug_into_previous_block(self) -> bool:
        self.set_part_idx(0)  # only one part in simple block like straight, and curve
        para = self.get_config()
        length = para[Parameter.length]
        basic_lane = self.positive_lanes[0]

        fake_positive_lane = ExtendStraightLane(basic_lane, length, [PGLineType.BROKEN, PGLineType.SIDE])
        reference_lane = fake_positive_lane
        start_point = reference_lane.position(fake_positive_lane.length, -reference_lane.width)
        end_point = reference_lane.position(0, -reference_lane.width)
        symmetric_lane = StraightLane(start_point, end_point, reference_lane.width)
        fake_negative_lane = symmetric_lane

        start_position = basic_lane.position(basic_lane.length, -basic_lane.width / 2)
        end_position = basic_lane.position(basic_lane.length + length, -basic_lane.width / 2)
        new_lane = StraightLane(start_position, end_position, basic_lane.width, [PGLineType.BROKEN, PGLineType.SIDE])

        start = self.pre_block_socket.positive_road.end_node
        end = self.add_road_node()
        socket = Road(start, end)
        _socket = -socket

        # create positive road
        no_cross = CreateRoadFrom(
            new_lane,
            1,
            socket,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        )
        # create negative road
        no_cross = create_overlap_road(
            socket,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        new_lane.line_colors = [PGLineColor.GREY, PGLineColor.GREY]
        self.add_sockets(BidirectionSocket(fake_positive_lane, fake_negative_lane, socket, _socket))
        return no_cross
