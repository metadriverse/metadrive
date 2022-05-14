"""
This script demonstrates how to use the environment where traffic and road map are loaded from argoverse dataset.
"""
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.engine.asset_loader import AssetLoader
import random


class DemoWaymoEnv(WaymoEnv):
    def reset(self, force_seed=None):
        if self.engine is not None:
            seeds = [i for i in range(self.config["case_num"])]
            seeds.remove(self.current_seed)
            force_seed = random.choice(seeds)
        super(DemoWaymoEnv, self).reset(force_seed=force_seed)


if __name__ == "__main__":
    asset_path = AssetLoader.asset_path
    try:
        env = DemoWaymoEnv(
            {
                "manual_control": True,
                "replay": False,
                "use_render": True,
                "waymo_data_directory": AssetLoader.file_path(asset_path, "waymo", return_raw_style=False),
                "case_num": 3
            }
        )
        o = env.reset()

        for i in range(1, 100000):
            o, r, d, info = env.step([1.0, 0.])
            env.render(text={"Switch perspective": "Q or B", "Reset Episode": "r"})
    except:
        print("Something Wrong happen in this example, would you kindly report it to developers? Thanks!")
    finally:
        env.close()
