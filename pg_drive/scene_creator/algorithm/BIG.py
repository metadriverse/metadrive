import logging

from numpy.random import RandomState
from panda3d.bullet import BulletWorld
from panda3d.core import NodePath
from typing import Union
from pg_drive.scene_creator.blocks.block import Block
from pg_drive.scene_creator.blocks.first_block import FirstBlock
from pg_drive.pg_config.pg_blocks import PgBlock
from pg_drive.scene_creator.road.road_network import RoadNetwork


class NextStep:
    back = 0
    forward = 1
    search_sibling = 3
    destruct_current = 4


class BigGenerateMethod:
    BLOCK_SEQUENCE = "block_sequence"
    BLOCK_NUM = "block_num"


class BIG:
    MAX_TRIAL = 2

    def __init__(
        self, lane_num: int, lane_width: float, global_network: RoadNetwork, render_node_path: NodePath,
        bullet_physics_world: BulletWorld, random_seed: int
    ):
        self._block_sequence = None
        self._random_seed = random_seed
        self.np_random = RandomState(random_seed)
        self._lane_num = lane_num
        self._lane_width = lane_width
        self.block_num = None
        self._render_node_path = render_node_path
        self._bullet_world = bullet_physics_world
        self._global_network = global_network
        self.blocks = []
        first_block = FirstBlock(
            self._global_network, self._lane_width, self._lane_num, self._render_node_path, self._bullet_world,
            self._random_seed
        )
        self.blocks.append(first_block)
        self.next_step = NextStep.forward

    def generate(self, generate_method: BigGenerateMethod, parameter: Union[str, int]):
        if generate_method == BigGenerateMethod.BLOCK_NUM:
            assert isinstance(parameter, int), "When generating map by assigning block num, the parameter should be int"
            self.block_num = parameter + 1
        elif generate_method == BigGenerateMethod.BLOCK_SEQUENCE:
            assert isinstance(parameter, str), "When generating map from block sequence, the parameter should be a str"
            self.block_num = len(parameter) + 1
            self._block_sequence = FirstBlock.ID + parameter
        """
        In order to embed it to the show_base loop, we implement BIG in a more complex way
        """
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
        if self._block_sequence is None:
            block_types = PgBlock.all_blocks()
            block_probabilities = PgBlock.block_probability()
            block_type = self.np_random.choice(block_types, p=block_probabilities)
        else:
            type_id = self._block_sequence[len(self.blocks)]
            block_type = PgBlock.get_block(type_id)
        sockets = [i for i in range(self.blocks[-1].SOCKET_NUM)]
        block = block_type(
            len(self.blocks), self.blocks[-1].get_socket(self.np_random.choice(sockets)), self._global_network,
            block_seed
        )
        return block

    def destruct(self, block):
        block.destruct_block_in_world(self._bullet_world)

    def construct(self, block) -> bool:
        return block.construct_block_in_world(self._render_node_path, self._bullet_world)

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
