from panda3d.core import NodePath

from pgdrive.pg_config.pg_space import PGSpace
from pgdrive.scene_creator.blocks.block import Block, BlockSocket
from pgdrive.scene_creator.blocks.create_block_utils import CreateRoadFrom, CreateAdverseRoad, ExtendStraightLane
from pgdrive.scene_creator.lanes.lane import LineType
from pgdrive.scene_creator.lanes.straight_lane import StraightLane
from pgdrive.scene_creator.road.road import Road
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.utils.constans import Decoration
from pgdrive.world.pg_physics_world import PGPhysicsWorld


class FirstBlock(Block):
    """
    A special Set, only used to create the first block. One scene has only one first block!!!
    """
    NODE_1 = ">"
    NODE_2 = ">>"
    NODE_3 = ">>>"
    PARAMETER_SPACE = PGSpace({})
    ID = "I"
    SOCKET_NUM = 1

    def __init__(
        self, global_network: RoadNetwork, lane_width: float, lane_num: int, render_root_np: NodePath,
        pg_physics_world: PGPhysicsWorld, random_seed
    ):
        place_holder = BlockSocket(Road(Decoration.start, Decoration.end), Road(Decoration.start, Decoration.end))
        super(FirstBlock, self).__init__(0, place_holder, global_network, random_seed)
        basic_lane = StraightLane(
            [0, lane_width * (lane_num - 1)], [10, lane_width * (lane_num - 1)],
            line_types=(LineType.STRIPED, LineType.SIDE),
            width=lane_width
        )
        ego_v_born_road = Road(self.NODE_1, self.NODE_2)
        CreateRoadFrom(basic_lane, lane_num, ego_v_born_road, self.block_network, self._global_network)
        CreateAdverseRoad(ego_v_born_road, self.block_network, self._global_network)

        next_lane = ExtendStraightLane(basic_lane, 40, [LineType.STRIPED, LineType.SIDE])
        other_v_born_road = Road(self.NODE_2, self.NODE_3)
        CreateRoadFrom(next_lane, lane_num, other_v_born_road, self.block_network, self._global_network)
        CreateAdverseRoad(other_v_born_road, self.block_network, self._global_network)

        self._create_in_world()
        global_network += self.block_network
        socket = self.create_socket_from_positive_road(other_v_born_road)
        socket.index = 0
        self.add_sockets(socket)
        self.attach_to_pg_world(render_root_np, pg_physics_world)
        self._reborn_roads = [other_v_born_road]
