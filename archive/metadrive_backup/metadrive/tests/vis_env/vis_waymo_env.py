"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import random

from metadrive.component.navigation_module.trajectory_navigation import TrajectoryNavigation
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


class DemoScenarioEnv(ScenarioEnv):
    def reset(self, seed=None):
        if self.engine is not None and seed is None:
            seeds = [i for i in range(self.config["num_scenarios"])]
            seeds.remove(self.current_seed)
            seed = random.choice(seeds)
        return super(DemoScenarioEnv, self).reset(seed=seed)


if __name__ == "__main__":
    asset_path = AssetLoader.asset_path
    env = DemoScenarioEnv(
        {
            "manual_control": False,
            "agent_policy": ReplayEgoCarPolicy,
            "use_render": True,
            "data_directory": AssetLoader.file_path(asset_path, "waymo", unix_style=False),
            "num_scenarios": 3,
            "start_scenario_index": 0,
            "crash_vehicle_done": False,
            "crash_vehicle_penalty": 0,
            "vehicle_config": {
                "navigation_module": TrajectoryNavigation,
                "show_side_detector": True,
                "show_lane_line_detector": True,
            }
        }
    )
    o, _ = env.reset(seed=0)

    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([1.0, 0.])
        # print(env.agent.height)
        env.render(text={"seed": env.current_seed, "reward": r})
        if tm or tc:
            # print(info["arrive_dest"])
            env.reset()
    env.close()
