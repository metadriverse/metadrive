import pytest
import numpy as np

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.idm_policy import WaymoIDMPolicy
from metadrive.policy.replay_policy import WaymoReplayEgoCarPolicy


@pytest.mark.parametrize("policy", [WaymoIDMPolicy, WaymoReplayEgoCarPolicy])
def test_waymo_env(policy, render=False, num_scenarios=3):
    WaymoIDMPolicy.NORMAL_SPEED = 30
    asset_path = AssetLoader.asset_path
    try:
        env = WaymoEnv(
            {
                "manual_control": False,
                "no_traffic": True if policy == WaymoIDMPolicy else False,
                "use_render": render,
                "agent_policy": policy,
                "data_directory": AssetLoader.file_path(asset_path, "waymo", return_raw_style=False),
                "num_scenarios": num_scenarios
            }
        )
        for seed in range(0, num_scenarios):
            env.reset(force_seed=seed)
            for i in range(1000):
                o, r, d, info = env.step([1.0, 0.])
                if d:
                    assert info["arrive_dest"], "Can not arrive dest"
                    print("{} track_length: ".format(env.engine.global_seed), info["track_length"])
                    # assert info["arrive_dest"], "Can not arrive dest"
                    break

                if i == 999:
                    raise ValueError("Can not arrive dest")
    finally:
        env.close()


def test_store_map_memory_leakage(render=False):
    WaymoIDMPolicy.NORMAL_SPEED = 30
    asset_path = AssetLoader.asset_path
    env = WaymoEnv(
        {
            "manual_control": False,
            "no_traffic": False,
            "store_map": True,
            "use_render": render,
            "agent_policy": WaymoReplayEgoCarPolicy,
            "data_directory": AssetLoader.file_path(asset_path, "waymo", return_raw_style=False),
            "num_scenarios": 3
        }
    )
    try:

        memory = []
        for _ in range(10):
            # test twp times for testing loading stored map
            for seed in range(3):
                env.reset(force_seed=seed)
                for i in range(1000):
                    o, r, d, info = env.step([1.0, 0.])
                    if d:
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
    # test_waymo_env(policy=WaymoIDMPolicy, render=True)
