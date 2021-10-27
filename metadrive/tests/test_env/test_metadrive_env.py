import copy
import os

import numpy as np
import pytest

from metadrive import MetaDriveEnv
from metadrive.constants import TerminationState

# Key: case name, value: environmental config
blackbox_test_configs = dict(
    default=dict(),
    random_traffic=dict(random_traffic=True),
    large_seed=dict(start_seed=1000000),
    traffic_density_0=dict(traffic_density=0),
    traffic_density_1=dict(traffic_density=1),
    decision_repeat_50=dict(decision_repeat=50),
    map_7=dict(map=7),
    map_30=dict(map=30),
    map_CCC=dict(map="CCC"),
    envs_100=dict(environment_num=100),
    envs_1000=dict(environment_num=1000),
    envs_10000=dict(environment_num=10000),
    envs_100000=dict(environment_num=100000),
    no_lidar0={"vehicle_config": dict(lidar=dict(num_lasers=0, distance=0, num_others=0))},
    no_lidar1={"vehicle_config": dict(lidar=dict(num_lasers=0, distance=10, num_others=0))},
    no_lidar2={"vehicle_config": dict(lidar=dict(num_lasers=10, distance=0, num_others=0))},
    no_lidar3={"vehicle_config": dict(lidar=dict(num_lasers=0, distance=0, num_others=10))},
    no_lidar4={"vehicle_config": dict(lidar=dict(num_lasers=10, distance=10, num_others=0))},
    no_lidar5={"vehicle_config": dict(lidar=dict(num_lasers=10, distance=0, num_others=10))},
    no_lidar6={"vehicle_config": dict(lidar=dict(num_lasers=0, distance=10, num_others=10))},
    no_lidar7={"vehicle_config": dict(lidar=dict(num_lasers=10, distance=10, num_others=10))},
)

pid_control_config = dict(environment_num=1, start_seed=5, map="CrXROSTR", traffic_density=0.0, use_render=False)

info_keys = [
    "cost", "velocity", "steering", "acceleration", "step_reward", TerminationState.CRASH_VEHICLE,
    TerminationState.OUT_OF_ROAD, TerminationState.SUCCESS
]

assert "__init__.py" not in os.listdir(os.path.dirname(__file__)), "Please remove __init__.py in tests directory."


def _act(env, action):
    assert env.action_space.contains(action)
    obs, reward, done, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert np.isscalar(reward)
    assert isinstance(info, dict)
    for k in info_keys:
        assert k in info


@pytest.mark.parametrize("config", list(blackbox_test_configs.values()), ids=list(blackbox_test_configs.keys()))
def test_pgdrive_env_blackbox(config):
    env = MetaDriveEnv(config=copy.deepcopy(config))
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()


if __name__ == '__main__':
    pytest.main(["-sv", "test_metadrive_env.py"])
