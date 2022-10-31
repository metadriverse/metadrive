from typing import Union

from metadrive.component.algorithm.BIG import BIG
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.pg_block import PGBlock
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.engine.core.physics_world import PhysicsWorld
from panda3d.core import NodePath


class NextStep:
    back = 0
    forward = 1
    search_sibling = 3
    destruct_current = 4


class BigGenerateMethod:
    BLOCK_SEQUENCE = "block_sequence"
    BLOCK_NUM = "block_num"


class CityBIG(BIG):
    MAX_TRIAL = 2

    def __init__(
        self,
        lane_num: int,
        lane_width: float,
        global_network: NodeRoadNetwork,
        render_node_path: NodePath,
        physics_world: PhysicsWorld,
        # block_type_version: str,
        random_seed=None
    ):
        super(CityBIG, self).__init__(
            lane_num,
            lane_width,
            global_network,
            render_node_path,
            physics_world,
            # block_type_version,
            random_seed=random_seed
        )

    def generate(self, generate_method: BigGenerateMethod, parameter: Union[str, int]):
        """
        In order to embed it to the show_base loop, we implement BIG in a more complex way
        """
        if generate_method == BigGenerateMethod.BLOCK_NUM:
            assert isinstance(parameter, int), "When generating map by assigning block num, the parameter should be int"
            self.block_num = parameter + 1
        elif generate_method == BigGenerateMethod.BLOCK_SEQUENCE:
            assert isinstance(parameter, str), "When generating map from block sequence, the parameter should be a str"
            self.block_num = len(parameter) + 1
            self._block_sequence = FirstPGBlock.ID + parameter
        while True:
            if self.big_helper_func():
                break
        return self._global_network

    def sample_block(self) -> PGBlock:
        """
        Sample a random block type
        """
        if self._block_sequence is None:
            block_types = PGBlockDistConfig.all_blocks()
            block_probabilities = PGBlockDistConfig.block_probability()
            block_type = self.np_random.choice(block_types, p=block_probabilities)
        else:
            type_id = self._block_sequence[len(self.blocks)]
            block_type = PGBlockDistConfig.get_block(type_id)

        # exclude first block
        socket_used = set([block.pre_block_socket for block in self.blocks[1:]])
        socket_available = []
        for b in self.blocks:
            socket_available += b.get_socket_list()
        socket_available = set(socket_available).difference(socket_used)
        socket = self.np_random.choice(sorted(list(socket_available), key=lambda x: x.index))

        block = block_type(
            len(self.blocks),
            socket,
            self._global_network,
            self.np_random.randint(0, 10000),
            ignore_intersection_checking=False
        )
        return block


class CityMap(BaseMap):
    def _generate(self):
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        big_map = CityBIG(
            self._config[self.LANE_NUM],
            self._config[self.LANE_WIDTH],
            self.road_network,
            parent_node_path,
            physics_world,
            # self._config["block_type_version"]
        )
        big_map.generate(self._config[self.GENERATE_TYPE], self._config[self.GENERATE_CONFIG])
        self.blocks = big_map.blocks

    @property
    def road_network_type(self):
        return NodeRoadNetwork
