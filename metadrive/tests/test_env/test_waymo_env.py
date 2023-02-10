import pytest

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.idm_policy import WaymoIDMPolicy
from metadrive.policy.replay_policy import WaymoReplayEgoCarPolicy


@pytest.mark.parametrize("policy", [WaymoIDMPolicy, WaymoReplayEgoCarPolicy])
def test_waymo_env(policy, render=False):
    WaymoIDMPolicy.NORMAL_SPEED = 30

    asset_path = AssetLoader.asset_path
    try:
        env = WaymoEnv(
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
        for seed in range(2 if policy == WaymoIDMPolicy else 3):
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
