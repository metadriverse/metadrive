import argparse
import random
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.idm_policy import WaymoIDMPolicy
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
import pytest


@pytest.mark.parametrize("policy", [WaymoIDMPolicy, ReplayEgoCarPolicy])
def test_waymo_env(policy, render=False):
    class DemoWaymoEnv(WaymoEnv):
        def reset(self, force_seed=None):
            if self.engine is not None:
                seeds = [i for i in range(self.config["case_num"])]
                seeds.remove(self.current_seed)
                force_seed = random.choice(seeds)
            super(DemoWaymoEnv, self).reset(force_seed=force_seed)

    asset_path = AssetLoader.asset_path
    try:
        env = DemoWaymoEnv(
            {
                "manual_control": True,
                "replay": True,
                "no_traffic": True if policy == WaymoIDMPolicy else False,
                "use_render": render,
                "agent_policy": policy,
                "waymo_data_directory": AssetLoader.file_path(asset_path, "waymo", return_raw_style=False),
                "case_num": 3
            }
        )
        for seed in range(3):
            env.reset(force_seed=seed)
            for i in range(1000):
                o, r, d, info = env.step([1.0, 0.])
                if d:
                    assert info["arrive_dest"], "Can not arrive dest"
                    break
                if i == 999:
                    raise ValueError("Can not arrive dest")
    finally:
        env.close()


if __name__ == "__main__":
    test_waymo_env(policy=WaymoIDMPolicy, render=True)
