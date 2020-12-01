from pg_drive.pg_config.parameter_space import BlockParameterSpace, Parameter
from pg_drive.pg_config.pg_space import PgSpace
from pg_drive.scene_creator.basic_utils import ExtendStraightLane, CreateRoadFrom, CreateAdverseRoad
from pg_drive.scene_creator.blocks.block import Block, BlockSocket
from pg_drive.scene_creator.lanes.lane import LineType
from pg_drive.scene_creator.lanes.straight_lane import StraightLane
from pg_drive.scene_creator.road.road import Road


class Straight(Block):
    """
    Straight Road
    ----------------------------------------
    ----------------------------------------
    ----------------------------------------
    """
    ID = "S"
    SOCKET_NUM = 1
    PARAMETER_SPACE = PgSpace(BlockParameterSpace.STRAIGHT)

    def _try_plug_into_previous_block(self) -> bool:
        self.set_part_idx(0)  # only one part in simple block like straight, and curve
        para = self.get_config()
        length = para[Parameter.length]
        basic_lane = self.positive_basic_lane
        assert isinstance(basic_lane, StraightLane), "Straight road can only connect straight type"
        new_lane = ExtendStraightLane(basic_lane, length, [LineType.STRIPED, LineType.SIDE])
        start = self._pre_block_socket.positive_road.end_node
        end = self.add_road_node()
        socket = Road(start, end)
        _socket = -socket

        # create positive road
        no_cross = CreateRoadFrom(new_lane, self.positive_lane_num, socket, self.block_network, self._global_network)
        # create negative road
        no_cross = CreateAdverseRoad(socket, self.block_network, self._global_network) and no_cross
        self.add_sockets(BlockSocket(socket, _socket))
        return no_cross
