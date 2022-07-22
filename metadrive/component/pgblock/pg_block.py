import copy
import logging
from collections import OrderedDict
from typing import Union, List

from metadrive.component.block.base_block import BaseBlock
from metadrive.component.road_network import Road
from metadrive.component.road_network.node_road_network import NodeRoadNetwork


class PGBlockSocket:
    """
    A pair of roads in reverse direction
    Positive_road is right road, and Negative road is left road on which cars drive in reverse direction
    BlockSocket is a part of block used to connect other blocks
    """
    def __init__(self, positive_road: Road, negative_road: Road = None):
        self.positive_road = positive_road
        self.negative_road = negative_road if negative_road else None
        self.index = None

    def set_index(self, block_name: str, index: int):
        self.index = self.get_real_index(block_name, index)

    @classmethod
    def get_real_index(cls, block_name: str, index: int):
        return "{}-socket{}".format(block_name, index)

    def is_socket_node(self, road_node):
        if road_node == self.positive_road.start_node or road_node == self.positive_road.end_node or \
                road_node == self.negative_road.start_node or road_node == self.negative_road.end_node:
            return True
        else:
            return False

    def get_socket_in_reverse(self):
        """
        Return a new socket whose positive road=self.negative_road, negative_road=self.positive_road
        """
        new_socket = copy.deepcopy(self)
        new_socket.positive_road, new_socket.negative_road = self.negative_road, self.positive_road
        return new_socket

    def is_same_socket(self, other):
        return True if self.positive_road == other.positive_road and self.negative_road == other.negative_road else False

    def get_positive_lanes(self, global_network):
        return self.positive_road.get_lanes(global_network)

    def get_negative_lanes(self, global_network):
        return self.negative_road.get_lanes(global_network)


class PGBlock(BaseBlock):
    """
    Abstract class of Block,
    BlockSocket: a part of previous block connecting this block

    <----------------------------------------------
    road_2_end <---------------------- road_2_start
    <~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
    road_1_start ----------------------> road_1_end
    ---------------------------------------------->
    BlockSocket = tuple(road_1, road_2)

    When single-direction block created, road_2 in block socket is useless.
    But it's helpful when a town is created.
    """
    def __init__(
        self,
        block_index: int,
        pre_block_socket: PGBlockSocket,
        global_network: NodeRoadNetwork,
        random_seed,
        ignore_intersection_checking=False
    ):

        self.name = str(block_index) + self.ID
        super(PGBlock, self).__init__(
            block_index, global_network, random_seed, ignore_intersection_checking=ignore_intersection_checking
        )
        # block information
        assert self.SOCKET_NUM is not None, "The number of Socket should be specified when define a new block"
        if block_index == 0:
            from metadrive.component.pgblock.first_block import FirstPGBlock
            assert isinstance(self, FirstPGBlock), "only first block can use block index 0"
        elif block_index < 0:
            logging.debug("It is recommended that block index should > 1")
        self.number_of_sample_trial = 0

        # own sockets, one block derives from a socket, but will have more sockets to connect other blocks
        self._sockets = OrderedDict()

        # used to connect previous blocks, save its info here
        self.pre_block_socket = pre_block_socket
        self.pre_block_socket_index = pre_block_socket.index

        # used to create this block, but for first block it is nonsense
        if block_index != 0:
            self.positive_lanes = self.pre_block_socket.get_positive_lanes(self._global_network)
            self.negative_lanes = self.pre_block_socket.get_negative_lanes(self._global_network)
            self.positive_lane_num = len(self.positive_lanes)
            self.negative_lane_num = len(self.negative_lanes)
            self.positive_basic_lane = self.positive_lanes[-1]  # most right or outside lane is the basic lane
            self.negative_basic_lane = self.negative_lanes[-1]  # most right or outside lane is the basic lane
            self.lane_width = self.positive_basic_lane.width_at(0)

    def _sample_topology(self) -> bool:
        """
        Sample a new topology, clear the previous settings at first
        """
        self.number_of_sample_trial += 1
        no_cross = self._try_plug_into_previous_block()
        return no_cross

    def get_socket(self, index: Union[str, int]) -> PGBlockSocket:
        if isinstance(index, int):
            if index < 0 or index >= len(self._sockets):
                raise ValueError("Socket of {}: index out of range".format(self.class_name))
            socket_index = list(self._sockets)[index]
        else:
            assert index.startswith(self.name)
            socket_index = index
        assert socket_index in self._sockets, (socket_index, self._sockets.keys())
        return self._sockets[socket_index]

    def add_respawn_roads(self, respawn_roads: Union[List[Road], Road]):
        """
        Use this to add spawn roads instead of modifying the list directly
        """
        if isinstance(respawn_roads, List):
            for road in respawn_roads:
                self._add_one_respawn_road(road)
        elif isinstance(respawn_roads, Road):
            self._add_one_respawn_road(respawn_roads)
        else:
            raise ValueError("Only accept List[Road] or Road in this func")

    def add_sockets(self, sockets: Union[List[PGBlockSocket], PGBlockSocket]):
        """
        Use this to add sockets instead of modifying the list directly
        """
        if isinstance(sockets, PGBlockSocket):
            self._add_one_socket(sockets)
        elif isinstance(sockets, List):
            for socket in sockets:
                self._add_one_socket(socket)

    def _add_one_socket(self, socket: PGBlockSocket):
        assert isinstance(socket, PGBlockSocket), "Socket list only accept BlockSocket Type"
        if socket.index is not None and not socket.index.startswith(self.name):
            logging.warning(
                "The adding socket has index {}, which is not started with this block name {}. This is dangerous! "
                "Current block has sockets: {}.".format(socket.index, self.name, self.get_socket_indices())
            )
        if socket.index is None:
            # if this socket is self block socket
            socket.set_index(self.name, len(self._sockets))
        self._sockets[socket.index] = socket

    def _clear_topology(self):
        super(PGBlock, self)._clear_topology()
        self._sockets.clear()

    def _try_plug_into_previous_block(self) -> bool:
        """
        Try to plug this Block to previous block's socket, return True for success, False for road cross
        """
        raise NotImplementedError

    @staticmethod
    def create_socket_from_positive_road(road: Road) -> PGBlockSocket:
        """
        We usually create road from positive road, thus this func can get socket easily.
        Note: it is not recommended to generate socket from negative road
        """
        assert road.start_node[0] != Road.NEGATIVE_DIR and road.end_node[0] != Road.NEGATIVE_DIR, \
            "Socket can only be created from positive road"
        positive_road = Road(road.start_node, road.end_node)
        return PGBlockSocket(positive_road, -positive_road)

    def get_socket_indices(self):
        return list(self._sockets.keys())

    def get_socket_list(self):
        return list(self._sockets.values())

    def set_part_idx(self, x):
        """
        It is necessary to divide block to some parts in complex block and give them unique id according to part idx
        """
        self.PART_IDX = x
        self.ROAD_IDX = 0  # clear the road idx when create new part

    def add_road_node(self):
        """
        Call me to get a new node name of this block.
        It is more accurate and recommended to use road_node() to get a node name
        """
        self.ROAD_IDX += 1
        return self.road_node(self.PART_IDX, self.ROAD_IDX - 1)

    def road_node(self, part_idx: int, road_idx: int) -> str:
        """
        return standard road node name
        """
        return self.node(self.block_index, part_idx, road_idx)

    @classmethod
    def node(cls, block_idx: int, part_idx: int, road_idx: int) -> str:
        return str(block_idx) + cls.ID + str(part_idx) + cls.DASH + str(road_idx) + cls.DASH

    def get_intermediate_spawn_lanes(self):
        trigger_lanes = self.block_network.get_positive_lanes()
        respawn_lanes = self.get_respawn_lanes()
        for lanes in respawn_lanes:
            if lanes not in trigger_lanes:
                trigger_lanes.append(lanes)
        return trigger_lanes

    @property
    def block_network_type(self):
        return NodeRoadNetwork

    def create_in_world(self):
        graph = self.block_network.graph
        for _from, to_dict in graph.items():
            for _to, lanes in to_dict.items():
                for _id, lane in enumerate(lanes):
                    lane.construct_lane_in_block(self, (_from, _to, _id))
                    pos_road = not Road(_from, _to).is_negative_road()
                    lane.construct_lane_line_in_block(self, [True, True] if _id == 0 and pos_road else [False, True])
