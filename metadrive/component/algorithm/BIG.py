import logging
from typing import Union

from panda3d.core import NodePath

from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.pg_block import PGBlock
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.engine.core.physics_world import PhysicsWorld
from metadrive.utils import get_np_random


class NextStep:
    back = 0
    forward = 1
    search_sibling = 3
    destruct_current = 4


class BigGenerateMethod:
    BLOCK_SEQUENCE = "block_sequence"
    BLOCK_NUM = "block_num"
    SINGLE_BLOCK = "single_block"


class BIG:
    MAX_TRIAL = 5

    def __init__(
        self,
        lane_num: int,
        lane_width: float,
        global_network: NodeRoadNetwork,
        render_node_path: NodePath,
        physics_world: PhysicsWorld,
        # block_type_version: str,
        exit_length=50,
        random_seed=None,
        block_dist_config=PGBlockDistConfig
    ):
        super(BIG, self).__init__()
        self.block_dist_config = block_dist_config
        self._block_sequence = None
        self.random_seed = random_seed
        self.np_random = get_np_random(random_seed)
        # Don't change this right now, since we need to make maps identical to old one
        self._lane_num = lane_num
        self._lane_width = lane_width
        self.block_num = None
        self._render_node_path = render_node_path
        self._physics_world = physics_world
        self._global_network = global_network
        self.blocks = []
        self._exit_length = exit_length
        first_block = FirstPGBlock(
            self._global_network,
            self._lane_width,
            self._lane_num,
            self._render_node_path,
            self._physics_world,
            length=self._exit_length
        )
        self.blocks.append(first_block)
        self.next_step = NextStep.forward
        # assert block_type_version in ["v1", "v2"]
        # self.block_type_version = block_type_version

    def generate(self, generate_method: str, parameter: Union[str, int]):
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

    def sample_block(self) -> PGBlock:
        """
        Sample a random block type
        """
        if self._block_sequence is None:
            block_types = self.block_dist_config.all_blocks()
            block_probabilities = self.block_dist_config.block_probability()
            block_type = self.np_random.choice(block_types, p=block_probabilities)
        else:
            type_id = self._block_sequence[len(self.blocks)]
            block_type = self.block_dist_config.get_block(type_id)

        socket = self.np_random.choice(self.blocks[-1].get_socket_indices())
        block = block_type(
            len(self.blocks),
            self.blocks[-1].get_socket(socket),
            self._global_network,
            self.np_random.randint(0, 10000),
            ignore_intersection_checking=False
        )
        return block

    def destruct(self, block):
        block.destruct_block(self._physics_world)

    def construct(self, block) -> bool:
        success = block.construct_block(self._render_node_path, self._physics_world)
        lane_num = max([len(socket.get_positive_lanes(self._global_network)) for socket in block._sockets.values()])
        if lane_num < self.block_dist_config.MIN_LANE_NUM or lane_num > self.block_dist_config.MAX_LANE_NUM:
            success = False
        return success

    def _forward(self):
        logging.debug("forward")
        block = self.sample_block()
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
        if len(self.blocks) == 1:
            self.next_step = NextStep.forward
            return
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
