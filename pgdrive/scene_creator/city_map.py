import logging
from typing import Union

from panda3d.core import NodePath
from pgdrive.scene_creator.algorithm.BIG import BIG
from pgdrive.scene_creator.blocks.block import Block
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.map import Map
from pgdrive.scene_creator.pg_blocks import PGBlock
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.engine.world.pg_physics_world import PGPhysicsWorld


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
        self, lane_num: int, lane_width: float, global_network: RoadNetwork, render_node_path: NodePath,
        pg_physics_world: PGPhysicsWorld, random_seed: int, block_type_version: str
    ):
        super(CityBIG, self).__init__(
            lane_num, lane_width, global_network, render_node_path, pg_physics_world, random_seed, block_type_version
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
            self._block_sequence = FirstBlock.ID + parameter
        while True:
            if self.big_helper_func():
                break
        return self._global_network

    def big_helper_func(self):
        if len(self.blocks) >= self.block_num and self.next_step == NextStep.forward:
            return True
        if self.next_step == NextStep.forward:
            self._forward()
        elif self.next_step == NextStep.destruct_current:
            self._destruct_current()
        elif self.next_step == NextStep.search_sibling:
            self._search_sibling()
        elif self.next_step == NextStep.back:
            self._go_back()
        return False

    def sample_block(self, block_seed: int) -> Block:
        """
        Sample a random block type
        """
        if self._block_sequence is None:
            block_types = PGBlock.all_blocks(self.block_type_version)
            block_probabilities = PGBlock.block_probability(self.block_type_version)
            block_type = self.np_random.choice(block_types, p=block_probabilities)
        else:
            type_id = self._block_sequence[len(self.blocks)]
            block_type = PGBlock.get_block(type_id, self.block_type_version)

        # exclude first block
        socket_used = set([block.pre_block_socket for block in self.blocks[1:]])
        socket_available = []
        for b in self.blocks:
            socket_available += b.get_socket_list()
        socket_available = set(socket_available).difference(socket_used)
        socket = self.np_random.choice(list(socket_available))

        block = block_type(len(self.blocks), socket, self._global_network, block_seed)
        return block

    def destruct(self, block):
        block.destruct_block(self._physics_world)

    def construct(self, block) -> bool:
        return block.construct_block(self._render_node_path, self._physics_world)

    def _forward(self):
        logging.debug("forward")
        block = self.sample_block(self.np_random.randint(0, 1000))
        self.blocks.append(block)
        success = self.construct(block)
        self.next_step = NextStep.forward if success else NextStep.destruct_current

    def _go_back(self):
        logging.debug("back")
        self.blocks.pop()
        last_block = self.blocks[-1]
        self.destruct(last_block)
        self.next_step = NextStep.search_sibling

    def _search_sibling(self):
        logging.debug("sibling")
        block = self.blocks[-1]
        if block.number_of_sample_trial < self.MAX_TRIAL:
            success = self.construct(block)
            self.next_step = NextStep.forward if success else NextStep.destruct_current
        else:
            self.next_step = NextStep.back

    def _destruct_current(self):
        logging.debug("destruct")
        block = self.blocks[-1]
        self.destruct(block)
        self.next_step = NextStep.search_sibling if block.number_of_sample_trial < self.MAX_TRIAL else NextStep.back

    def __del__(self):
        logging.debug("Destroy Big")


class CityMap(Map):
    def _generate(self):
        parent_node_path, pg_physics_world = self.pgdrive_engine.worldNP, self.pgdrive_engine.physics_world
        big_map = CityBIG(
            self.config[self.LANE_NUM], self.config[self.LANE_WIDTH], self.road_network, parent_node_path,
            pg_physics_world, self.random_seed, self.config["block_type_version"]
        )
        big_map.generate(self.config[self.GENERATE_TYPE], self.config[self.GENERATE_CONFIG])
        self.blocks = big_map.blocks
