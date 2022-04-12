import numpy as np

from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.engine.engine_utils import initialize_engine
from metadrive.envs import MetaDriveEnv
from metadrive.envs.base_env import BASE_DEFAULT_CONFIG
from metadrive.envs.metadrive_env import METADRIVE_DEFAULT_CONFIG
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import Config


def _create_vehicle():
    v_config = Config(BASE_DEFAULT_CONFIG["vehicle_config"]).update(METADRIVE_DEFAULT_CONFIG["vehicle_config"])
    v_config.update({"use_render": False, "offscreen_render": False})
    config = Config(BASE_DEFAULT_CONFIG)
    config.update(
        {
            "use_render": False,
            "pstats": False,
            "offscreen_render": False,
            "debug": False,
            "vehicle_config": v_config
        }
    )
    initialize_engine(config)
    v = DefaultVehicle(vehicle_config=v_config, random_seed=0)
    return v


def test_idm_policy_briefly():
    env = MetaDriveEnv()
    env.reset()
    try:
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


def test_idm_policy_is_moving(render=False, in_test=True):
    # config = {"traffic_mode": "hybrid", "map": "SS", "traffic_density": 1.0}
    config = {"traffic_mode": "respawn", "map": "SS", "traffic_density": 1.0}
    if render:
        config.update({"use_render": True, "manual_control": True})
    env = MetaDriveEnv(config)
    env.reset(force_seed=0)
    last_pos = None
    try:
        for t in range(100):
            env.step(env.action_space.sample())
            vs = env.engine.traffic_manager.traffic_vehicles
            # print("Position: ", {str(v)[:4]: v.position for v in vs})
            new_pos = np.array([v.position for v in vs])
            if t > 50 and last_pos is not None and in_test:
                assert np.any(new_pos != last_pos)
            last_pos = new_pos
        env.reset()
    finally:
        env.close()


if __name__ == '__main__':
    # test_idm_policy_briefly()
    test_idm_policy_is_moving(render=True, in_test=False)
