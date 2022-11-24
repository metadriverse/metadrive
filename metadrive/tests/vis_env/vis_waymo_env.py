"""
This script demonstrates how to use the environment where traffic and road map are loaded from argoverse dataset.
"""
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.component.vehicle_navigation_module.trajectory_navigation import WaymoTrajectoryNavigation
from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
import random


class DemoWaymoEnv(WaymoEnv):
    def reset(self, force_seed=None):
        if self.engine is not None and force_seed is None:
            seeds = [i for i in range(self.config["case_num"])]
            seeds.remove(self.current_seed)
            force_seed = random.choice(seeds)
        super(DemoWaymoEnv, self).reset(force_seed=force_seed)


if __name__ == "__main__":
    asset_path = AssetLoader.asset_path
    env = DemoWaymoEnv(
        {
            "manual_control": False,
            "agent_policy": ReplayEgoCarPolicy,
            "replay": False,
            "use_render": True,
            "waymo_data_directory": AssetLoader.file_path(asset_path, "waymo", return_raw_style=False),
            "case_num": 3,
            "start_case_index": 0,
            "crash_vehicle_done": False,
            "crash_vehicle_penalty": 0,
            "vehicle_config": {
                "navigation_module": WaymoTrajectoryNavigation,
                "show_side_detector": True,
                "show_lane_line_detector": True,
            }
        }
    )
    o = env.reset()

    for i in range(1, 100000):
        o, r, d, info = env.step([1.0, 0.])
        env.render(text={"seed": env.current_seed, "reward": r})
        if d:
            print(info["arrive_dest"])
            env.reset()
    env.close()
