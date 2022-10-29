from panda3d.core import NodePath

from metadrive.component.pgblock.create_pg_block_utils import CreateRoadFrom, CreateAdverseRoad, ExtendStraightLane
from metadrive.component.pgblock.pg_block import PGBlock, PGBlockSocket
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.road_network import Road
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.constants import Decoration, LineType
from metadrive.engine.core.physics_world import PhysicsWorld
from metadrive.utils.space import ParameterSpace


class FirstPGBlock(PGBlock):
    """
    A special Set, only used to create the first block. One scene has only one first block!!!
    """
    NODE_1 = ">"
    NODE_2 = ">>"
    NODE_3 = ">>>"
    PARAMETER_SPACE = ParameterSpace({})
    ID = "I"
    SOCKET_NUM = 1
    ENTRANCE_LENGTH = 10

    def __init__(
        self,
        global_network: NodeRoadNetwork,
        lane_width: float,
        lane_num: int,
        render_root_np: NodePath,
        physics_world: PhysicsWorld,
        length: float = 30,
        ignore_intersection_checking=False
    ):
        place_holder = PGBlockSocket(Road(Decoration.start, Decoration.end), Road(Decoration.start, Decoration.end))
        super(FirstPGBlock, self).__init__(
            0, place_holder, global_network, random_seed=0, ignore_intersection_checking=ignore_intersection_checking
        )
        if length < self.ENTRANCE_LENGTH:
            print("Warning: first block length is two small", length, "<", self.ENTRANCE_LENGTH)
        self._block_objects = []
        basic_lane = StraightLane(
            [0, lane_width * (lane_num - 1)], [self.ENTRANCE_LENGTH, lane_width * (lane_num - 1)],
            line_types=(LineType.BROKEN, LineType.SIDE),
            width=lane_width
        )
        ego_v_spawn_road = Road(self.NODE_1, self.NODE_2)
        CreateRoadFrom(
            basic_lane,
            lane_num,
            ego_v_spawn_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        )
        CreateAdverseRoad(
            ego_v_spawn_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        next_lane = ExtendStraightLane(basic_lane, length - self.ENTRANCE_LENGTH, [LineType.BROKEN, LineType.SIDE])
        other_v_spawn_road = Road(self.NODE_2, self.NODE_3)
        CreateRoadFrom(
            next_lane,
            lane_num,
            other_v_spawn_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        )
        CreateAdverseRoad(
            other_v_spawn_road,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        self._create_in_world()

        # global_network += self.block_network
        global_network.add(self.block_network)

        socket = self.create_socket_from_positive_road(other_v_spawn_road)
        socket.set_index(self.name, 0)

        self.add_sockets(socket)
        self.attach_to_world(render_root_np, physics_world)
        self._respawn_roads = [other_v_spawn_road]

    def _try_plug_into_previous_block(self) -> bool:
        raise ValueError("BIG Recursive calculation error! Can not find a right sequence for building map! Check BIG")

    def destruct_block(self, physics_world: PhysicsWorld):
        """This block can not be destructed"""
        pass
