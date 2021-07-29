from panda3d.core import NodePath

from pgdrive.constants import Decoration, LineType
from pgdrive.scene_creator.blocks.pg_block import PGBlock, PGBlockSocket
from pgdrive.scene_creator.blocks.create_block_utils import CreateRoadFrom, CreateAdverseRoad, ExtendStraightLane
from pgdrive.scene_creator.lane.straight_lane import StraightLane
from pgdrive.scene_creator.road.road import Road
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.utils.pg_space import PGSpace
from pgdrive.engine.core.pg_physics_world import PGPhysicsWorld


class FirstPGBlock(PGBlock):
    """
    A special Set, only used to create the first block. One scene has only one first block!!!
    """
    NODE_1 = ">"
    NODE_2 = ">>"
    NODE_3 = ">>>"
    PARAMETER_SPACE = PGSpace({})
    ID = "I"
    SOCKET_NUM = 1
    ENTRANCE_LENGTH = 10

    def __init__(
        self,
        global_network: RoadNetwork,
        lane_width: float,
        lane_num: int,
        render_root_np: NodePath,
        pg_physics_world: PGPhysicsWorld,
        length: float = 50
    ):
        place_holder = PGBlockSocket(Road(Decoration.start, Decoration.end), Road(Decoration.start, Decoration.end))
        super(FirstPGBlock, self).__init__(0, place_holder, global_network, random_seed=0)
        assert length > self.ENTRANCE_LENGTH
        self._block_objects = []
        basic_lane = StraightLane(
            [0, lane_width * (lane_num - 1)], [self.ENTRANCE_LENGTH, lane_width * (lane_num - 1)],
            line_types=(LineType.BROKEN, LineType.SIDE),
            width=lane_width
        )
        ego_v_spawn_road = Road(self.NODE_1, self.NODE_2)
        CreateRoadFrom(basic_lane, lane_num, ego_v_spawn_road, self.block_network, self._global_network)
        CreateAdverseRoad(ego_v_spawn_road, self.block_network, self._global_network)

        next_lane = ExtendStraightLane(basic_lane, length - self.ENTRANCE_LENGTH, [LineType.BROKEN, LineType.SIDE])
        other_v_spawn_road = Road(self.NODE_2, self.NODE_3)
        CreateRoadFrom(next_lane, lane_num, other_v_spawn_road, self.block_network, self._global_network)
        CreateAdverseRoad(other_v_spawn_road, self.block_network, self._global_network)

        self.node_path = NodePath(self._block_name)
        self._create_in_world()

        # global_network += self.block_network
        global_network.add(self.block_network)

        socket = self.create_socket_from_positive_road(other_v_spawn_road)
        socket.set_index(self._block_name, 0)

        self.add_sockets(socket)
        self.attach_to_world(render_root_np, pg_physics_world)
        self._respawn_roads = [other_v_spawn_road]
