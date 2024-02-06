from metadrive.envs import BaseEnv
from metadrive.obs.observation_base import DummyObservation
import logging

# ======================================== new content ===============================================
import cv2
from metadrive.component.map.pg_map import PGMap
from metadrive.manager.base_manager import BaseManager
from metadrive.component.pgblock.first_block import FirstPGBlock


class MyMapManager(BaseManager):
    PRIORITY = 0

    def __init__(self):
        super(MyMapManager, self).__init__()
        self.current_map = None
        self.all_maps = {idx: None for idx in range(3)}  # store the created map
        self._map_shape = ["X", "T", "O"]  # three types of maps

    def reset(self):
        idx = self.engine.global_random_seed % 3
        if self.all_maps[idx] is None:
            # create maps on the fly
            new_map = PGMap(map_config=dict(type=PGMap.BLOCK_SEQUENCE, config=self._map_shape[idx]))
            self.all_maps[idx] = new_map

        # attach map in the world
        map = self.all_maps[idx]
        map.attach_to_world()
        self.current_map = map
        return dict(current_map=self._map_shape[idx])

    def before_reset(self):
        if self.current_map is not None:
            self.current_map.detach_from_world()
            self.current_map = None

    def destroy(self):
        # clear all maps when this manager is destroyed
        super(MyMapManager, self).destroy()
        for map in self.all_maps.values():
            if map is not None:
                map.destroy()
        self.all_maps = None


# Expand the default config system, specify where to spawn the car
MY_CONFIG = dict(agent_configs={"default_agent": dict(spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0))})


class MyEnv(BaseEnv):
    @classmethod
    def default_config(cls):
        config = super(MyEnv, cls).default_config()
        config.update(MY_CONFIG)
        return config

    def setup_engine(self):
        super(MyEnv, self).setup_engine()
        self.engine.register_manager("map_manager", MyMapManager())

    # ======================================== new content ===============================================

    def reward_function(self, agent):
        return 0, {}

    def cost_function(self, agent):
        return 0, {}

    def done_function(self, agent):
        return False, {}

    def get_single_observation(self):
        return DummyObservation()


if __name__ == "__main__":
    frames = []

    # create env
    env = MyEnv(
        dict(
            use_render=False,
            # if you have a screen and OpenGL suppor, you can set use_render=True to use 3D rendering
            manual_control=True,  # we usually manually control the car to test environment
            num_scenarios=4,
            log_level=logging.CRITICAL
        )
    )  # suppress logging message
    for i in range(4):
        # reset
        o, info = env.reset(seed=i)
        print("Load map with shape: {}".format(info["current_map"]))
        # you can set window=True and remove generate_gif() if you have a screen.
        # Or just use 3D rendering and remove all stuff related to env.render()
        frame = env.render(
            mode="topdown",
            window=False,  # turn me on, if you have screen
            scaling=3,
            camera_position=(50, 0),
            screen_size=(400, 400)
        )
        frames.append(frame)
    cv2.imwrite("demo.png", cv2.cvtColor(cv2.hconcat(frames), cv2.COLOR_RGB2BGR))
    env.close()

from IPython.display import Image

Image(open("demo.png", 'rb').read())
