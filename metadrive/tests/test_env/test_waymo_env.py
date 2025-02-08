import numpy as np
import pytest

from metadrive.constants import get_color_palette
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


@pytest.mark.parametrize("policy", [TrajectoryIDMPolicy, ReplayEgoCarPolicy])
def test_waymo_env(policy, render=False, num_scenarios=3):
    TrajectoryIDMPolicy.NORMAL_SPEED = 30
    asset_path = AssetLoader.asset_path
    try:
        env = ScenarioEnv(
            {
                "manual_control": False,
                "no_traffic": True if policy == TrajectoryIDMPolicy else False,
                "use_render": render,
                "agent_policy": policy,
                "show_crosswalk": True,
                "show_sidewalk": True,
                "data_directory": AssetLoader.file_path(asset_path, "waymo", unix_style=False),
                "num_scenarios": num_scenarios
            }
        )
        for seed in range(0, num_scenarios):
            env.reset(seed=seed)
            for i in range(1000):
                o, r, tm, tc, info = env.step([1.0, 0.])
                assert env.observation_space.contains(o)
                if tm or tc:
                    assert info["arrive_dest"], "Can not arrive dest"
                    print("{} track_length: ".format(env.engine.global_seed), info["track_length"])
                    # assert info["arrive_dest"], "Can not arrive dest"
                    break

                if i == 999:
                    raise ValueError("Can not arrive dest")
            assert env.agent.panda_color == get_color_palette()[2]
    finally:
        env.close()


def test_store_map_memory_leakage(render=False):
    TrajectoryIDMPolicy.NORMAL_SPEED = 30
    asset_path = AssetLoader.asset_path
    env = ScenarioEnv(
        {
            "manual_control": False,
            "no_traffic": False,
            "store_map": True,
            "use_render": render,
            "agent_policy": ReplayEgoCarPolicy,
            "data_directory": AssetLoader.file_path(asset_path, "waymo", unix_style=False),
            "num_scenarios": 3
        }
    )
    try:

        memory = []
        for _ in range(10):
            # test twp times for testing loading stored map
            for seed in range(3):
                env.reset(seed=seed)
                for i in range(1000):
                    o, r, tm, tc, info = env.step([1.0, 0.])
                    assert env.observation_space.contains(o)
                    if tm or tc:
                        assert info["arrive_dest"], "Can not arrive dest"
                        assert env.episode_step > 60
                        break
                    if i == 999:
                        raise ValueError("Can not arrive dest")

            def process_memory():
                import psutil
                import os
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                return mem_info.rss

            memory.append(process_memory() / 1024 / 1024)  # in mb
        # print(memory)
        assert np.std(memory) < 2, "They should be almost the same"
    finally:
        env.close()


if __name__ == "__main__":
    test_store_map_memory_leakage(render=True)
    # test_waymo_env(policy=TrajectoryIDMPolicy, render=True)
