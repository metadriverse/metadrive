import os

import numpy as np
import pytest
from pgdrive import PGDriveEnv
from pgdrive.constants import DEFAULT_AGENT
from pgdrive.scene_creator.vehicle_module.PID_controller import PIDController, Target

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
    no_lidar0=dict(target_vehicle_configs={DEFAULT_AGENT: dict(lidar=dict(num_lasers=0, distance=0, num_others=0))}),
    no_lidar1=dict(target_vehicle_configs={DEFAULT_AGENT: dict(lidar=dict(num_lasers=0, distance=10, num_others=0))}),
    no_lidar2=dict(target_vehicle_configs={DEFAULT_AGENT: dict(lidar=dict(num_lasers=10, distance=0, num_others=0))}),
    no_lidar3=dict(target_vehicle_configs={DEFAULT_AGENT: dict(lidar=dict(num_lasers=0, distance=0, num_others=10))}),
    no_lidar4=dict(target_vehicle_configs={DEFAULT_AGENT: dict(lidar=dict(num_lasers=10, distance=10, num_others=0))}),
    no_lidar5=dict(target_vehicle_configs={DEFAULT_AGENT: dict(lidar=dict(num_lasers=10, distance=0, num_others=10))}),
    no_lidar6=dict(target_vehicle_configs={DEFAULT_AGENT: dict(lidar=dict(num_lasers=0, distance=10, num_others=10))}),
    no_lidar7=dict(target_vehicle_configs={DEFAULT_AGENT: dict(lidar=dict(num_lasers=10, distance=10, num_others=10))}),
)

pid_control_config = dict(environment_num=1, start_seed=5, map="CrXROSTR", traffic_density=0.0)

info_keys = [
    "cost", "velocity", "steering", "acceleration", "step_reward", "crash_vehicle", "out_of_road", "arrive_dest"
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
    env = PGDriveEnv(config=config)
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


def test_zombie():
    env = PGDriveEnv(pid_control_config)
    target = Target(0.375, 30)
    dest = [-288.88415527, -411.55871582]
    try:
        o = env.reset()
        steering_controller = PIDController(1.6, 0.0008, 27.3)
        acc_controller = PIDController(0.1, 0.001, 0.3)
        steering_error = o[0] - target.lateral
        steering = steering_controller.get_result(steering_error)
        acc_error = env.vehicles[env.DEFAULT_AGENT].speed - target.speed
        acc = acc_controller.get_result(acc_error)
        for i in range(1, 1000000):
            o, r, d, info = env.step([-steering, acc])
            steering_error = o[0] - target.lateral
            steering = steering_controller.get_result(steering_error)
            t_speed = target.speed if abs(o[12] - 0.5) < 0.01 else target.speed - 10
            acc_error = env.vehicles[env.DEFAULT_AGENT].speed - t_speed
            acc = acc_controller.get_result(acc_error)
            if d:
                assert info["arrive_dest"]
                assert abs(env.vehicles[env.DEFAULT_AGENT].position[0] - dest[0]) < 0.15 and \
                       abs(env.vehicles[env.DEFAULT_AGENT].position[1] - dest[1]) < 0.15
                break
    finally:
        env.close()


if __name__ == '__main__':
    pytest.main(["-s", "test_pgdrive_env.py"])
