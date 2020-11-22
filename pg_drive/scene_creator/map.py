import logging
from panda3d.bullet import BulletWorld
from pg_drive.pg_config.pg_config import PgConfig
from panda3d.core import NodePath

from pg_drive.scene_creator.algorithm.BIG import BIG, BigGenerateMethod
from pg_drive.scene_creator.road.road_network import RoadNetwork


class Map:
    def __init__(self, config: dict = None):
        """
        Scene can be stored and recover to save time when we access scenes encountered before
        Scene should contain road_network, blocks and vehicles
        """
        self.road_network = RoadNetwork()
        self.blocks = []
        self.lane_width = None
        self.lane_num = None
        self.random_seed = None
        self.config = self.default_config()
        if config:
            self.config.update(config)

    @staticmethod
    def default_config():
        return PgConfig({"type": BigGenerateMethod.BLOCK_NUM, "config": None})

    def big_generate(
        self, lane_width: float, lane_num: int, seed: int, parent_node_path: NodePath, physics_world: BulletWorld
    ):
        map = BIG(lane_num, lane_width, self.road_network, parent_node_path, physics_world, seed)
        map.generate(self.config["type"], self.config["config"])
        self.random_seed = seed
        self.blocks = map.blocks
        self.lane_num = lane_num
        self.lane_width = lane_width
        # TODO wrap this to support more generating methods
        self.road_network.update_indices()
        self.road_network.build_helper()

    def custom_generate(self, blocks):
        """
        TODO Used to read maps. Not a final edition
        """
        for block in blocks:
            self.road_network += block.road_network
        self.blocks = blocks

    def re_generate(self, parent_node_path: NodePath, bt_physics_world: BulletWorld):
        """
        For convenience
        """
        self.add_to_bullet_physics_world(bt_physics_world)
        from pg_drive.utils.visualization_loader import VisLoader
        if VisLoader.loader is not None:
            self.add_to_render_module(parent_node_path)

    def add_to_render_module(self, parent_node_path: NodePath):
        """
        If original node path is removed, this can re attach blocks to render module
        """
        for block in self.blocks:
            block.add_to_render_module(parent_node_path)

    def add_to_bullet_physics_world(self, bt_physics_world: BulletWorld):
        """
        If the original bullet physics world is deleted, call this to re-add road network
        """
        for block in self.blocks:
            block.add_to_physics_world(bt_physics_world)

    def remove_from_physics_world(self, bt_physics_world: BulletWorld):
        for block in self.blocks:
            block.remove_from_physics_world(bt_physics_world)

    def remove_from_render_module(self):
        for block in self.blocks:
            block.remove_from_render_module()

    def destroy_map(self, bt_physics_world: BulletWorld):
        for block in self.blocks:
            block.destroy(bt_physics_world)

    def __del__(self):
        describe = self.random_seed if self.random_seed is not None else "custom"
        logging.debug("Scene {} is destroyed".format(describe))
