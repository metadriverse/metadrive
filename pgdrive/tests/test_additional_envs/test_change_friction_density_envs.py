import numpy as np

from pgdrive.envs.generation_envs.change_density_env import ChangeDensityEnv
from pgdrive.envs.generation_envs.change_friction_env import ChangeFrictionEnv
from pgdrive.envs.generation_envs.side_pass_env import SidePassEnv


def _run(env):
    env.seed(100000)
    for _ in range(5):
        obs = env.reset()
        for s in range(100):
            action = np.array([0.0, 1.0])
            o, r, d, i = env.step(action)
            if d:
                env.reset()
    env.close()


def test_change_friction():
    _run(ChangeFrictionEnv(config={"environment_num": 100, "start_seed": 1000, "change_friction": True}))
    _run(ChangeFrictionEnv(config={"environment_num": 100, "start_seed": 1000, "change_friction": False}))


def test_side_pass_env():
    _run(SidePassEnv({"target_vehicle_configs": {"default_agent": {"show_navi_mark": False}}}))


def test_change_density_env():
    _run(ChangeDensityEnv(config={"change_density": False}))
    _run(ChangeDensityEnv(config={"change_density": True}))


if __name__ == '__main__':
    # pytest.main(["-sv", "test_change_friction_density_envs.py"])
    # test_side_pass_env()
    test_change_friction()