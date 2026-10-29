"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import random

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


class DemoScenarioEnv(ScenarioEnv):
    def reset(self, seed=None):
        if self.engine is not None:
            seeds = [i for i in range(self.config["num_scenarios"])]
            seeds.remove(self.current_seed)
            seed = random.choice(seeds)
        return super(DemoScenarioEnv, self).reset(seed=seed)


def test_top_down_semantics(render=False):
    asset_path = AssetLoader.asset_path
    try:
        env = DemoScenarioEnv(
            {
                "manual_control": False,
                "reactive_traffic": False,
                "no_light": True,
                "agent_policy": ReplayEgoCarPolicy,
                "use_render": False,
                "sequential_seed": True,
                "data_directory": AssetLoader.file_path(asset_path, "nuscenes", unix_style=False),
                "num_scenarios": 10
            }
        )
        o, _ = env.reset()
        for seed in range(100):
            env.reset(seed=seed)
            for i in range(1, 10):
                o, r, tm, tc, info = env.step([1.0, 0.])
                if render:
                    this_frame_fig = env.render(
                        mode="top_down",
                        semantic_map=True,
                        film_size=(8000, 8000),
                        num_stack=1,
                        scaling=10,
                    )
                # pygame.image.save(this_frame_fig, "{}.png".format(seed))
    finally:
        env.close()


if __name__ == '__main__':
    test_top_down_semantics(True)
