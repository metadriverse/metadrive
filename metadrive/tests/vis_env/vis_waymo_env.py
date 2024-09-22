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
        # if self.engine is not None and seed is None:
        #     seeds = [i for i in range(self.config["num_scenarios"])]
        #     seeds.remove(self.current_seed)
        #     seed = random.choice(seeds)
        return super(DemoScenarioEnv, self).reset(seed=1)


if __name__ == "__main__":
    render = False
    asset_path = AssetLoader.asset_path
    env = DemoScenarioEnv(
        {
            "manual_control": True,
            "show_policy_mark": render,
            # "agent_policy": ReplayEgoCarPolicy,
            "reactive_traffic": True,
            "use_render": render,
            "data_directory": AssetLoader.file_path(asset_path, "waymo", unix_style=False),
            "num_scenarios": 1,
            "start_scenario_index": 1,
            "crash_vehicle_done": False,
            "crash_vehicle_penalty": 0,
            "vehicle_config": {
                "navigation_module": TrajectoryNavigation,
                "show_side_detector": True,
                "show_lane_line_detector": True,
            }
        }
    )
    o, _ = env.reset(seed=1)

    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([0.0, -1])
        # print(env.agent.height)
        env.render(text={"seed": env.current_seed, "reward": r},
                   # mode="topdown"
                   )
        if tm or tc:
            # print(info["arrive_dest"])
            env.reset()
    env.close()
