"""
This script demonstrates how to use the environment where traffic and road map are loaded from argoverse dataset.
"""
from metadrive.constants import HELP_MESSAGE
from metadrive.envs.real_data_envs.waymo_idm_env import WaymoIDMEnv
from metadrive.engine.asset_loader import AssetLoader
import random
import argparse


class DemoWaymoEnv(WaymoIDMEnv):
    def reset(self, force_seed=None):
        if self.engine is not None:
            seeds = [i for i in range(self.config["case_num"])]
            seeds.remove(self.current_seed)
            force_seed = random.choice(seeds)
        super(DemoWaymoEnv, self).reset(force_seed=force_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactive_traffic", action="store_true")
    args = parser.parse_args()
    asset_path = AssetLoader.asset_path
    print(HELP_MESSAGE)
    try:
        env = DemoWaymoEnv(
            {
                "manual_control": True,
                "replay": False if args.reactive_traffic else True,
                "use_render": True,
                "waymo_data_directory": AssetLoader.file_path(asset_path, "waymo", return_raw_style=False),
                "case_num": 3
            }
        )
        o = env.reset()

        for i in range(1, 100000):
            o, r, d, info = env.step([1.0, 0.])
            env.render(text={"Switch perspective": "Q or B", "Reset Episode": "R"})
    except:
        pass
    finally:
        env.close()
