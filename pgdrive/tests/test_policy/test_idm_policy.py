import numpy as np

from pgdrive.component.vehicle.base_vehicle import BaseVehicle
from pgdrive.envs import PGDriveEnvV2
from pgdrive.envs.base_env import BASE_DEFAULT_CONFIG
from pgdrive.envs.pgdrive_env import PGDriveEnvV1_DEFAULT_CONFIG
from pgdrive.policy.idm_policy import IDMPolicy
from pgdrive.utils import Config
from pgdrive.engine.engine_utils import initialize_engine


def _create_vehicle():
    v_config = Config(BASE_DEFAULT_CONFIG["vehicle_config"]).update(PGDriveEnvV1_DEFAULT_CONFIG["vehicle_config"])
    v_config.update({"use_render": False, "use_image": False})
    config = Config(BASE_DEFAULT_CONFIG)
    config.update(
        {
            "use_render": False,
            "pstats": False,
            "use_image": False,
            "debug": False,
            "vehicle_config": v_config
        }
    )
    initialize_engine(config, None)
    v = BaseVehicle(vehicle_config=v_config, random_seed=0)
    return v


def test_idm_policy_briefly():
    env = PGDriveEnvV2()
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
    config = {"traffic_mode": "hybrid", "map": "SS"}
    if render:
        config.update({"use_render": True, "fast": True, "manual_control": True})
    env = PGDriveEnvV2(config)
    env.reset()
    last_pos = None
    try:
        for _ in range(1000):
            env.step(env.action_space.sample())
            vs = env.engine.traffic_manager.traffic_vehicles
            # print("Position: ", {str(v)[:4]: v.position for v in vs})
            new_pos = np.array([v.position for v in vs])
            if last_pos is not None and in_test:
                assert not np.all(new_pos == last_pos)
            last_pos = new_pos
        env.reset()
    finally:
        env.close()


if __name__ == '__main__':
    # test_idm_policy_briefly()
    test_idm_policy_is_moving(render=True, in_test=False)
