from typing import Dict

from panda3d.bullet import BulletWorld
from panda3d.core import NodePath
from scene_creator.basic_utils import Decoration
from pg_config.pg_space import PgSpace
from scene_creator.blocks.block import BlockSocket, Block
from scene_creator.lanes.lane import LineType
from scene_creator.lanes.straight_lane import StraightLane
from scene_creator.road.road import Road
from scene_creator.road.road_network import RoadNetwork
from ..basic_utils import CreateRoadFrom, CreateAdverseRoad, ExtendStraightLane


class FirstBlock(Block):
    """
    A special Block type, only used to create the first block. One scene has only one first block!!!
    """
    NODE_1 = ">"
    NODE_2 = ">>"
    NODE_3 = ">>>"
    PARAMETER_SPACE = PgSpace({})
    ID = "I"
    SOCKET_NUM = 1

    def __init__(
        self, global_network: RoadNetwork, lane_width: float, lane_num: int, render_root_np: NodePath,
        bullet_physics_world: BulletWorld, random_seed
    ):
        place_holder = BlockSocket(Road(Decoration.start, Decoration.end), Road(Decoration.start, Decoration.end))
        super(FirstBlock, self).__init__(0, place_holder, global_network, random_seed)
        basic_lane = StraightLane(
            [0, lane_width], [10, lane_width], line_types=(LineType.STRIPED, LineType.SIDE), width=lane_width
        )
        ego_v_born_road = Road(self.NODE_1, self.NODE_2)
        CreateRoadFrom(basic_lane, lane_num, ego_v_born_road, self.block_network, self._global_network)
        CreateAdverseRoad(ego_v_born_road, self.block_network, self._global_network)

        next_lane = ExtendStraightLane(basic_lane, 40, [LineType.STRIPED, LineType.SIDE])
        other_v_born_road = Road(self.NODE_2, self.NODE_3)
        CreateRoadFrom(next_lane, lane_num, other_v_born_road, self.block_network, self._global_network)
        CreateAdverseRoad(other_v_born_road, self.block_network, self._global_network)

        self._create_in_bullet()
        global_network += self.block_network
        socket = self.create_socket_from_positive_road(other_v_born_road)
        self.add_sockets(socket)
        self.add_to_render_module(render_root_np)
        self.add_to_physics_world(bullet_physics_world)
        self._reborn_roads = [other_v_born_road]
