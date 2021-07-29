import numpy as np

from pgdrive.scene_creator.blocks.pg_block import PGBlock
from pgdrive.scene_creator.blocks.create_block_utils import CreateAdverseRoad, CreateRoadFrom, create_bend_straight
from pgdrive.constants import LineType
from pgdrive.scene_creator.road.road import Road
from pgdrive.utils.pg_space import PGSpace, Parameter, BlockParameterSpace


class Curve(PGBlock):
    """
        2 - - - - - - - - - -
       / 3 - - - - - - - - - -
      / /
     / /
    0 1
    """
    ID = "C"
    SOCKET_NUM = 1
    PARAMETER_SPACE = PGSpace(BlockParameterSpace.CURVE)

    def _try_plug_into_previous_block(self) -> bool:
        parameters = self.get_config()
        basic_lane = self.positive_basic_lane
        lane_num = self.positive_lane_num

        # part 1
        start_node = self.pre_block_socket.positive_road.end_node
        end_node = self.add_road_node()
        positive_road = Road(start_node, end_node)
        length = parameters[Parameter.length]
        direction = parameters[Parameter.dir]
        angle = parameters[Parameter.angle]
        radius = parameters[Parameter.radius]
        curve, straight = create_bend_straight(
            basic_lane, length, radius, np.deg2rad(angle), direction, basic_lane.width,
            (LineType.BROKEN, LineType.SIDE)
        )
        no_cross = CreateRoadFrom(curve, lane_num, positive_road, self.block_network, self._global_network)
        no_cross = CreateAdverseRoad(positive_road, self.block_network, self._global_network) and no_cross

        # part 2
        start_node = end_node
        end_node = self.add_road_node()
        positive_road = Road(start_node, end_node)
        no_cross = CreateRoadFrom(
            straight, lane_num, positive_road, self.block_network, self._global_network
        ) and no_cross
        no_cross = CreateAdverseRoad(positive_road, self.block_network, self._global_network) and no_cross

        # common properties
        self.add_sockets(self.create_socket_from_positive_road(positive_road))
        return no_cross
