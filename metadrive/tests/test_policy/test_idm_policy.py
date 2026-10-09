import numpy as np
import pytest

from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.engine.engine_utils import initialize_engine
from metadrive.envs import MetaDriveEnv
from metadrive.envs.base_env import BASE_DEFAULT_CONFIG
from metadrive.envs.metadrive_env import METADRIVE_DEFAULT_CONFIG
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import Config


def _create_vehicle():
    v_config = Config(BASE_DEFAULT_CONFIG["vehicle_config"]).update(METADRIVE_DEFAULT_CONFIG["vehicle_config"])
    v_config.update({"use_render": False, "image_observation": False})
    config = Config(BASE_DEFAULT_CONFIG)
    config.update(
        {
            "use_render": False,
            "pstats": False,
            "image_observation": False,
            "debug": False,
            "vehicle_config": v_config
        }
    )
    initialize_engine(config)
    v = DefaultVehicle(vehicle_config=v_config, random_seed=0)
    return v


@pytest.mark.parametrize("use_mesh_terrain", [True, False], ids=["plane", "mesh"])
def test_idm_policy_briefly(use_mesh_terrain):
    env = MetaDriveEnv({"use_mesh_terrain": use_mesh_terrain})
    try:
        env.reset()
        vehicles = env.engine.traffic_manager.traffic_vehicles
        for v in vehicles:
            policy = IDMPolicy(
                vehicle=v, traffic_manager=env.engine.traffic_manager, delay_time=1, random_seed=env.current_seed
            )
            action = policy.before_step(v, front_vehicle=None, rear_vehicle=None, current_map=env.engine.current_map)
            action = policy.step(dt=0.02)
            action = policy.after_step(v, front_vehicle=None, rear_vehicle=None, current_map=env.engine.current_map)
            env.engine.policy_manager.register_new_policy(
                IDMPolicy,
                vehicle=v,
                traffic_manager=env.engine.traffic_manager,
                delay_time=1,
                random_seed=env.current_seed
            )
        env.step(env.action_space.sample())
        env.reset()
    finally:
        env.close()


@pytest.mark.parametrize("use_mesh_terrain", [True, False], ids=["plane", "mesh"])
def test_idm_policy_is_moving(use_mesh_terrain, render=False, in_test=True):
    # config = {"traffic_mode": "hybrid", "map": "SS", "traffic_density": 1.0}
    config = {"use_mesh_terrain": use_mesh_terrain, "traffic_mode": "respawn", "map": "SS", "traffic_density": 1.0}
    if render:
        config.update({"use_render": True, "manual_control": True})
    env = MetaDriveEnv(config)
    try:
        env.reset(seed=0)
        last_pos = None
        for t in range(100):
            env.step(env.action_space.sample())
            vs = env.engine.traffic_manager.traffic_vehicles
            # # print("Position: ", {str(v)[:4]: v.position for v in vs})
            new_pos = np.array([v.position for v in vs])
            if t > 50 and last_pos is not None and in_test:
                assert np.any(new_pos != last_pos)
            last_pos = new_pos
        env.reset()
    finally:
        env.close()


def test_idm_desired_gap_uses_si_units():
    """The desired gap is in metres, so speeds must enter in m/s (not km/h)."""
    from types import SimpleNamespace
    policy = SimpleNamespace(DISTANCE_WANTED=10.0, TIME_WANTED=1.5, ACC_FACTOR=1.0, DEACC_FACTOR=-5)
    heading = np.array([1.0, 0.0])
    ego = SimpleNamespace(
        speed=10.0,
        speed_km_h=36.0,
        velocity=np.array([10.0, 0.0]),
        velocity_km_h=np.array([36.0, 0.0]),
        heading=heading,
    )
    front = SimpleNamespace(
        speed=10.0, speed_km_h=36.0, velocity=np.array([10.0, 0.0]), velocity_km_h=np.array([36.0, 0.0])
    )
    # same speed: d* = d0 + v * T = 10 m + 10 m/s * 1.5 s = 25 m
    assert np.isclose(IDMPolicy.desired_gap(policy, ego, front), 25.0)
    # approaching at 2 m/s adds v * dv / (2 * sqrt(a * b)) = 10 * 2 / (2 * sqrt(5)) m
    front.velocity = np.array([8.0, 0.0])
    front.speed = 8.0
    expected = 25.0 + 10.0 * 2.0 / (2 * np.sqrt(5.0))
    assert np.isclose(IDMPolicy.desired_gap(policy, ego, front), expected)


if __name__ == '__main__':
    # test_idm_policy_briefly()
    test_idm_policy_is_moving(False, render=True, in_test=False)
