from metadrive.component.pgblock.create_pg_block_utils import ExtendStraightLane, CreateRoadFrom, CreateAdverseRoad
from metadrive.component.pgblock.pg_block import PGBlock, PGBlockSocket
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.road_network import Road
from metadrive.constants import LineType
from metadrive.utils.space import ParameterSpace, Parameter, BlockParameterSpace


class Straight(PGBlock):
    """
    Straight Road
    ----------------------------------------
    ----------------------------------------
    ----------------------------------------
    """
    ID = "S"
    SOCKET_NUM = 1
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.STRAIGHT)

    def _try_plug_into_previous_block(self) -> bool:
        self.set_part_idx(0)  # only one part in simple block like straight, and curve
        para = self.get_config()
        length = para[Parameter.length]
        basic_lane = self.positive_basic_lane
        assert isinstance(basic_lane, StraightLane), "Straight road can only connect straight type"
        new_lane = ExtendStraightLane(basic_lane, length, [LineType.BROKEN, LineType.SIDE])
        start = self.pre_block_socket.positive_road.end_node
        end = self.add_road_node()
        socket = Road(start, end)
        _socket = -socket

        # create positive road
        no_cross = CreateRoadFrom(
            new_lane,
            self.positive_lane_num,
            socket,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        )
        # create negative road
        no_cross = CreateAdverseRoad(
            socket,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        self.add_sockets(PGBlockSocket(socket, _socket))
        return no_cross
