import logging
from metadrive.component.map.base_map import BaseMap
from metadrive.component.nuplan_block.nuplan_block import NuPlanBlock
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork


class NuPlanMap(BaseMap):
    def __init__(self, map_index, nuplan_center, random_seed=None):
        self.map_index = map_index
        self._center = nuplan_center
        super(NuPlanMap, self).__init__(dict(id=map_index), random_seed=random_seed)

    @property
    def nuplan_center(self):
        return self._center

    def _generate(self):
        block = NuPlanBlock(0, self.road_network, 0, self.map_index, self.nuplan_center)
        block.construct_block(self.engine.worldNP, self.engine.physics_world, attach_to_world=True)
        self.blocks.append(block)

    def play(self):
        # For debug
        for b in self.blocks:
            b._create_in_world(skip=True)
            b.attach_to_world(self.engine.worldNP, self.engine.physics_world)
            b.detach_from_world(self.engine.physics_world)

    def metadrive_to_nuplan_position(self, pos):
        return pos[0] + self.nuplan_center[0], pos[1] + self.nuplan_center[1]

    def nuplan_to_metadrive_position(self, pos):
        return pos[0] - self.nuplan_center[0], pos[1] - self.nuplan_center[1]

    @property
    def road_network_type(self):
        return EdgeRoadNetwork

    def destroy(self):
        self.map_index = None
        super(NuPlanMap, self).destroy()

    def __del__(self):
        # self.destroy()
        logging.debug("Map is Released")
        print("[NuPlanMap] Map is Released")


if __name__ == "__main__":
    from metadrive.envs.real_data_envs.nuplan_env import NuPlanEnv
    from metadrive.manager.nuplan_data_manager import NuPlanDataManager
    from metadrive.engine.engine_utils import initialize_engine, set_global_random_seed

    default_config = NuPlanEnv.default_config()
    default_config["use_render"] = True
    default_config["debug"] = True
    default_config["debug_static_world"] = True
    engine = initialize_engine(default_config)
    set_global_random_seed(0)

    engine.data_manager = NuPlanDataManager()
    engine.data_manager.seed(0)
    map = NuPlanMap(map_index=0, nuplan_center=[664396.54429387, 3997613.41534655])
    map.attach_to_world()
    # engine.enableMouse()
    map.road_network.show_bounding_box(engine)

    # argoverse data set is as the same coordinates as panda3d
    pos = map.get_center_point()
    engine.main_camera.set_bird_view_pos(pos)

    while True:
        map.engine.step()
