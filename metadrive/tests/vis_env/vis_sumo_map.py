"""use netconvert --opendrive-files CARLA_town01.net.xml first"""


from metadrive.envs import BaseEnv
from metadrive.obs.observation_base import DummyObservation
import logging
from metadrive.manager.sumo_map_manager import SumoMapManager
from metadrive.engine.asset_loader import AssetLoader


class MyEnv(BaseEnv):

    def reward_function(self, agent):
        return 0, {}

    def cost_function(self, agent):
        return 0, {}

    def done_function(self, agent):
        return False, {}

    def get_single_observation(self):
        return DummyObservation()

    def setup_engine(self):
        super().setup_engine()
        map_path = AssetLoader.file_path("carla", "CARLA_town01.net.xml", unix_style=False)
        self.engine.register_manager("map_manager", SumoMapManager(map_path))


if __name__ == "__main__":
    # create env
    env = MyEnv(dict(use_render=True,
                     # if you have a screen and OpenGL suppor, you can set use_render=True to use 3D rendering
                     vehicle_config={"spawn_position_heading": [(0, 0), 0]},
                     manual_control=True,  # we usually manually control the car to test environment
                     use_mesh_terrain=True,
                     log_level=logging.CRITICAL))  # suppress logging message
    env.reset()
    for i in range(10000):
        # step
        obs, reward, termination, truncate, info = env.step(env.action_space.sample())
    env.close()
