from typing import List

from panda3d.core import NodePath

from metadrive.component.algorithm.BIG import BigGenerateMethod, BIG
from metadrive.component.algorithm.blocks_prob_dist import PGBlockConfig
from metadrive.component.blocks import FirstPGBlock
from metadrive.component.map.base_map import BaseMap, MapGenerateMethod
from metadrive.engine.core.physics_world import PhysicsWorld


class PGMap(BaseMap):
    def _generate(self):
        """
        We can override this function to introduce other methods!
        """
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        generate_type = self._config[self.GENERATE_TYPE]
        if generate_type == BigGenerateMethod.BLOCK_NUM or generate_type == BigGenerateMethod.BLOCK_SEQUENCE:
            self._big_generate(parent_node_path, physics_world)

        elif generate_type == MapGenerateMethod.PG_MAP_FILE:
            # other config such as lane width, num and seed will be valid, since they will be read from file
            blocks_config = self.read_map(self._config[self.GENERATE_CONFIG])
            self._config_generate(blocks_config, parent_node_path, physics_world)
        else:
            raise ValueError("Map can not be created by {}".format(generate_type))

    def _big_generate(self, parent_node_path: NodePath, physics_world: PhysicsWorld):
        big_map = BIG(
            self._config[self.LANE_NUM],
            self._config[self.LANE_WIDTH],
            self.road_network,
            parent_node_path,
            physics_world,
            # self._config["block_type_version"],
            exit_length=self._config["exit_length"],
            random_seed=self.engine.global_random_seed,
        )
        big_map.generate(self._config[self.GENERATE_TYPE], self._config[self.GENERATE_CONFIG])
        self.blocks = big_map.blocks

    def _config_generate(self, blocks_config: List, parent_node_path: NodePath, physics_world: PhysicsWorld):
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"
        last_block = FirstPGBlock(
            global_network=self.road_network,
            lane_width=self._config[self.LANE_WIDTH],
            lane_num=self._config[self.LANE_NUM],
            render_root_np=parent_node_path,
            physics_world=physics_world,
            length=self._config["exit_length"],
            ignore_intersection_checking=True
        )
        self.blocks.append(last_block)
        for block_index, b in enumerate(blocks_config[1:], 1):
            block_type = PGBlockConfig.get_block(b.pop(self.BLOCK_ID))
            pre_block_socket_index = b.pop(self.PRE_BLOCK_SOCKET_INDEX)
            last_block = block_type(
                block_index,
                last_block.get_socket(pre_block_socket_index),
                self.road_network,
                random_seed=self.engine.global_random_seed,
                ignore_intersection_checking=True
            )
            last_block.construct_from_config(b, parent_node_path, physics_world)
            self.blocks.append(last_block)
