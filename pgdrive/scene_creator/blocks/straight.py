from pgdrive.pg_config.parameter_space import BlockParameterSpace, Parameter
from pgdrive.pg_config.pg_space import PGSpace
from pgdrive.scene_creator.blocks.block import Block, BlockSocket
from pgdrive.scene_creator.blocks.create_block_utils import ExtendStraightLane, CreateRoadFrom, CreateAdverseRoad
from pgdrive.scene_creator.lane.abs_lane import LineType
from pgdrive.scene_creator.lane.straight_lane import StraightLane
from pgdrive.scene_creator.road.road import Road


class Straight(Block):
    """
    Straight Road
    ----------------------------------------
    ----------------------------------------
    ----------------------------------------
    """
    ID = "S"
    SOCKET_NUM = 1
    PARAMETER_SPACE = PGSpace(BlockParameterSpace.STRAIGHT)

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
        no_cross = CreateRoadFrom(new_lane, self.positive_lane_num, socket, self.block_network, self._global_network)
        # create negative road
        no_cross = CreateAdverseRoad(socket, self.block_network, self._global_network) and no_cross
        self.add_sockets(BlockSocket(socket, _socket))
        return no_cross
