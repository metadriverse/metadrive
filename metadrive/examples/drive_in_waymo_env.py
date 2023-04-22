"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
import random

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv


class DemoWaymoEnv(WaymoEnv):
    def reset(self, force_seed=None):
        if self.engine is not None:
            seeds = [i for i in range(self.config["num_scenarios"])]
            seeds.remove(self.current_seed)
            force_seed = random.choice(seeds)
        super(DemoWaymoEnv, self).reset(force_seed=force_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactive_traffic", action="store_true")
    parser.add_argument("--top_down", action="store_true")
    args = parser.parse_args()
    extra_args = dict(mode="top_down", film_size=(800, 800)) if args.top_down else {}
    asset_path = AssetLoader.asset_path
    print(HELP_MESSAGE)
    try:
        env = DemoWaymoEnv(
            {
                "manual_control": True,
                "reactive_traffic": True if args.reactive_traffic else False,
                "use_render": True if not args.top_down else False,
                "data_directory": AssetLoader.file_path(asset_path, "waymo", return_raw_style=False),
                "num_scenarios": 3
            }
        )
        o = env.reset()

        for i in range(1, 100000):
            o, r, d, info = env.step([1.0, 0.])
            env.render(text={"Switch perspective": "Q or B", "Reset Episode": "R"}, **extra_args)
            if d:
                env.reset()
    except Exception as e:
        raise e
    finally:
        env.close()
