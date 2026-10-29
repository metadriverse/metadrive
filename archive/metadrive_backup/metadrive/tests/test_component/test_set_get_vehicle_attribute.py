import numpy as np
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.utils.math import wrap_to_pi

from metadrive.envs.metadrive_env import MetaDriveEnv


def test_set_get_vehicle_attribute(render=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "decision_repeat": 1,
            "map": "SSS",
            "use_render": render,
        }
    )
    try:
        o, _ = env.reset()
        for _ in range(10):
            env.agent.set_velocity([5, 0], in_local_frame=False)
            o, r, tm, tc, info = env.step([0, 0])
            assert abs(env.agent.speed - 5) < 0.01  # may encounter friction
            assert np.isclose(env.agent.velocity, np.array([5, 0]), rtol=1e-2, atol=1e-2).all()
            assert abs(env.agent.speed - env.agent.speed_km_h / 3.6) < 1e-4
            assert np.isclose(env.agent.velocity, env.agent.velocity_km_h / 3.6).all()

        for _ in range(10):
            o, r, tm, tc, info = env.step([0, 0])
            env.agent.set_velocity([0, 5], in_local_frame=False)
            assert abs(env.agent.speed - 5) < 0.1
            assert np.isclose(env.agent.velocity, np.array([0, 5]), rtol=1e-5, atol=1e-5).all()
            assert abs(env.agent.speed - env.agent.speed_km_h / 3.6) < 1e-4
            assert np.isclose(env.agent.velocity, env.agent.velocity_km_h / 3.6).all()

        for _ in range(10):
            o, r, tm, tc, info = env.step([0, 0])
            env.agent.set_velocity([5, 3], value=10, in_local_frame=False)
            assert abs(env.agent.speed - 10) < 0.1
            assert np.isclose(
                env.agent.velocity,
                np.array([5 / np.linalg.norm(np.array([5, 3])) * 10, 3 / np.linalg.norm(np.array([5, 3])) * 10]),
                rtol=1e-5,
                atol=1e-5
            ).all()
            assert abs(env.agent.speed - env.agent.speed_km_h / 3.6) < 1e-4
            assert np.isclose(env.agent.velocity, env.agent.velocity_km_h / 3.6).all()

        for _ in range(10):
            o, r, tm, tc, info = env.step([0, 0])
            env.agent.set_velocity([0.3, 0.1], value=10, in_local_frame=False)
            assert abs(env.agent.speed - 10) < 0.1
            assert np.isclose(
                env.agent.velocity,
                np.array(
                    [0.3 / np.linalg.norm(np.array([0.3, 0.1])) * 10, 0.1 / np.linalg.norm(np.array([0.3, 0.1])) * 10]
                ),
                rtol=1e-5,
                atol=1e-5
            ).all()
            assert abs(env.agent.speed - env.agent.speed_km_h / 3.6) < 0.0001
            assert np.isclose(env.agent.velocity, env.agent.velocity_km_h / 3.6).all()

    finally:
        env.close()


def test_coordinates(render=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "decision_repeat": 1,
            "vehicle_config": {
                "enable_reverse": True
            },
            "map": "SSS",
            "use_render": render,
        }
    )
    try:
        o, _ = env.reset()
        assert abs(env.agent.heading_theta) == 0
        assert np.isclose(env.agent.heading, [1.0, 0]).all()
        env.agent.set_velocity([5, 0], in_local_frame=True)
        for _ in range(10):
            env.agent.set_velocity([5, 0], in_local_frame=True)
            o, r, tm, tc, info = env.step([0, 0])
        assert abs(env.agent.velocity[0] - 5.) < 1e-2 and abs(env.agent.velocity[1]) < 0.001

        o, _ = env.reset()
        assert abs(env.agent.heading_theta) == 0
        assert np.isclose(env.agent.heading, [1.0, 0]).all()
        env.agent.set_velocity([5, 0], in_local_frame=False)
        for _ in range(10):
            o, r, tm, tc, info = env.step([0, 0])
        assert env.agent.velocity[0] > 3. and abs(env.agent.velocity[1]) < 0.001

        env.reset()
        env.agent.set_velocity([0, 5], in_local_frame=False)

        for _ in range(1):
            o, r, tm, tc, info = env.step([0, 0])
        assert env.agent.velocity[1] > 3. and abs(env.agent.velocity[0]) < 0.002

        env.reset()
        assert abs(env.agent.heading_theta) == 0
        assert np.isclose(env.agent.heading, [1.0, 0]).all()
        env.agent.set_velocity([-5, 0], in_local_frame=False)
        for _ in range(10):
            o, r, tm, tc, info = env.step([0, 0])
        assert env.agent.velocity[0] < -3. and abs(env.agent.velocity[1]) < 0.001

        env.agent.set_velocity([0, -5], in_local_frame=False)

        for _ in range(1):
            o, r, tm, tc, info = env.step([0, 0])
        assert env.agent.velocity[1] < -3. and abs(env.agent.velocity[0]) < 0.002

        # steering left
        env.reset()
        begining_pos = env.agent.position
        assert abs(env.agent.heading_theta) == 0
        assert np.isclose(env.agent.heading, [1.0, 0]).all()

        for _ in range(100):
            o, r, tm, tc, info = env.step([0.8, 0.8])
        assert env.agent.velocity[1] > 1. and abs(env.agent.velocity[0]) > 1
        assert env.agent.heading_theta > 0.3  # rad
        assert env.agent.position[0] > begining_pos[0] and env.agent.position[1] > begining_pos[1]

        # steering right
        env.reset()
        begining_pos = env.agent.position
        assert abs(env.agent.heading_theta) == 0
        assert np.isclose(env.agent.heading, [1.0, 0]).all()

        for _ in range(100):
            o, r, tm, tc, info = env.step([-0.8, 0.8])
        assert env.agent.velocity[1] < -1. and abs(env.agent.velocity[0]) > 1
        assert env.agent.position[0] > begining_pos[0] and env.agent.position[1] < begining_pos[1]
        assert env.agent.heading_theta < -0.3  # rad

        env.reset()
        env.agent.set_heading_theta(np.deg2rad(90))
        for _ in range(10):
            o, r, tm, tc, info, = env.step([-0., 0.])
        assert wrap_to_pi(abs(env.agent.heading_theta - np.deg2rad(90))) < 1
        assert np.isclose(env.agent.heading, np.array([0, 1]), 1e-4, 1e-4).all()

        env.reset()
        env.agent.set_heading_theta(np.deg2rad(45))
        for _ in range(10):
            o, r, tm, tc, info, = env.step([-0., 0.])
        assert wrap_to_pi(abs(env.agent.heading_theta - np.deg2rad(45))) < 1
        assert np.isclose(env.agent.heading, np.array([np.sqrt(2) / 2, np.sqrt(2) / 2]), 1e-4, 1e-4).all()

        env.reset()
        env.agent.set_heading_theta(np.deg2rad(-90))
        for _ in range(10):
            o, r, tm, tc, info, = env.step([-0., 0.])
        assert abs(env.agent.heading_theta + np.deg2rad(-90) + np.pi) < 0.01
        assert np.isclose(env.agent.heading, np.array([0, -1]), 1e-4, 1e-4).all()
    finally:
        env.close()


def test_set_angular_v_and_set_v_no_friction(render=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "decision_repeat": 5,
            "map": "SSS",
            "use_render": render,
            "vehicle_config": {
                "no_wheel_friction": True
            }
        }
    )
    try:
        o, _ = env.reset()
        for _ in range(100):
            # 10 s , np.pi/10 per second
            env.agent.set_angular_velocity(np.pi / 10)
            o, r, tm, tc, info = env.step([0, 0])
        assert abs(wrap_to_pi(env.agent.heading_theta) - np.pi) < 1e-2, env.agent.heading_theta
        # print(env.agent.heading_theta / np.pi * 180)

        o, _ = env.reset()
        for _ in range(100):
            # 10 s , np.pi/10 per second
            env.agent.set_angular_velocity(18, in_rad=False)
            o, r, tm, tc, info = env.step([0, 0])
        assert abs(wrap_to_pi(env.agent.heading_theta) - np.pi) < 1e-2, env.agent.heading_theta
        # print(env.agent.heading_theta / np.pi * 180)

        o, _ = env.reset()
        start = env.agent.position[0]
        for _ in range(100):
            # 10 s
            env.agent.set_velocity([1, 0], in_local_frame=True)
            o, r, tm, tc, info = env.step([0, 0])
        assert abs(env.agent.position[0] - start - 10) < 5e-2, env.agent.position

        o, _ = env.reset()
        start = env.agent.position[0]
        for _ in range(100):
            # 10 s
            env.agent.set_velocity([1, 0], in_local_frame=False)
            o, r, tm, tc, info = env.step([0, 0])
        assert abs(env.agent.position[0] - start - 10) < 5e-2, env.agent.position

        o, _ = env.reset()
        start = env.agent.position[1]
        for _ in range(10):
            # 10 s
            env.agent.set_velocity([0, 1], in_local_frame=False)
            o, r, tm, tc, info = env.step([0, 0])
        assert abs(env.agent.position[1] - start - 1) < 5e-2, env.agent.position

        o, _ = env.reset()
        start = env.agent.position[0]
        env.agent.set_heading_theta(-np.pi / 2)
        for _ in range(100):
            # 10 s
            env.agent.set_velocity([0, 1], in_local_frame=True)
            o, r, tm, tc, info = env.step([0, 0])
        assert abs(env.agent.position[0] - start - 10) < 5e-2, env.agent.position

        o, _ = env.reset()
        start = env.agent.position[0]
        env.agent.set_heading_theta(-np.pi / 2)
        for _ in range(100):
            # 10 s
            env.agent.set_velocity([1, 0], in_local_frame=False)
            o, r, tm, tc, info = env.step([0, 0])
        assert abs(env.agent.position[0] - start - 10) < 5e-2, env.agent.position
    finally:
        env.close()


def test_set_angular_v_and_set_v_no_friction_pedestrian(render=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "decision_repeat": 5,
            "map": "S",
            "use_render": render,
            "vehicle_config": {
                "no_wheel_friction": True
            }
        }
    )
    try:
        o, _ = env.reset()
        env.engine.terrain.dynamic_nodes[0].setFriction(0.)
        obj_1 = env.engine.spawn_object(Pedestrian, position=[10, 3], heading_theta=0, random_seed=1)
        for _ in range(10):
            # 10 s , np.pi/10 per second
            obj_1.set_angular_velocity(np.pi / 10)
            o, r, tm, tc, info = env.step([0, 0])
        assert abs(wrap_to_pi(obj_1.heading_theta) - np.pi / 10) < 1e-2, obj_1.heading_theta
        obj_1.destroy()

        o, _ = env.reset()
        env.engine.terrain.dynamic_nodes[0].setFriction(0.)
        obj_1 = env.engine.spawn_object(Pedestrian, position=[10, 3], heading_theta=0, random_seed=1)
        for _ in range(10):
            # obj_1.set_position([30,0], 10)
            # 10 s , np.pi/10 per second
            obj_1.set_angular_velocity(18, in_rad=False)
            o, r, tm, tc, info = env.step([0, 0])
        assert abs(wrap_to_pi(obj_1.heading_theta) - np.pi / 10) < 1e-2, obj_1.heading_theta
        # print(obj_1.heading_theta / np.pi * 180)
        obj_1.destroy()

        o, _ = env.reset()
        env.engine.terrain.dynamic_nodes[0].setFriction(0.)
        obj_1 = env.engine.spawn_object(Pedestrian, position=[10, 3], heading_theta=0, random_seed=1)
        start_p = obj_1.position[0]
        for _ in range(10):
            # obj_1.set_position([30,0], 10)
            # 10 s , np.pi/10 per second
            obj_1.set_velocity([1, 0])
            o, r, tm, tc, info = env.step([0, 0])
        assert abs(obj_1.position[0] - start_p) > 0.7
        # print(obj_1.heading_theta / np.pi * 180)
        obj_1.destroy()

        o_ = env.reset()
        env.engine.terrain.dynamic_nodes[0].setFriction(0.)
        obj_1 = env.engine.spawn_object(Pedestrian, position=[10, 3], heading_theta=0, random_seed=1)
        start_p = obj_1.position[1]
        for _ in range(10):
            # obj_1.set_position([30,0], 10)
            # 10 s , np.pi/10 per second
            obj_1.set_velocity([0, 1])
            o, r, tm, tc, info = env.step([0, 0])
        assert abs(obj_1.position[1] - start_p) > 0.7
        # print(obj_1.heading_theta / np.pi * 180)
        obj_1.destroy()

    finally:
        env.close()


if __name__ == "__main__":
    # test_set_angular_v_and_set_v_no_friction_pedestrian(True)
    test_set_angular_v_and_set_v_no_friction(False)
    # test_coordinates(True)
    # test_set_get_vehicle_attribute(True)
