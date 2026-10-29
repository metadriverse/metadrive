from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.lidar import Lidar
from metadrive.envs.metadrive_env import MetaDriveEnv


def test_sensor_config():
    env = MetaDriveEnv({
        "sensors": {
            "lidar": (Lidar, )
        },
    })
    try:
        env.reset()
        assert 50 in env.engine.get_sensor("lidar").broad_detectors
        for i in range(1, 10):
            o, r, tm, tc, info = env.step([0, 1])
    finally:
        env.close()

    env = MetaDriveEnv({
        "sensors": {
            "lidar_new": (Lidar, )
        },
    })
    try:
        env.reset()
        assert 50 in env.engine.get_sensor("lidar").broad_detectors
        assert len(env.engine.get_sensor("lidar").broad_detectors) == 1
        env.engine.get_sensor("lidar_new").perceive(
            env.agent,
            physics_world=env.engine.physics_world.dynamic_world,
            num_lasers=env.agent.config["lidar"]["num_lasers"],
            distance=100.5,
            detector_mask=None
        )
        assert 100 in env.engine.get_sensor("lidar_new").broad_detectors
        assert 50 not in env.engine.get_sensor("lidar_new").broad_detectors
        assert len(env.engine.get_sensor("lidar_new").broad_detectors) == 1
        for i in range(1, 10):
            o, r, tm, tc, info = env.step([0, 1])
    finally:
        env.close()

    try:
        env.reset()
        assert 50 in env.engine.get_sensor("lidar").broad_detectors
        assert len(env.engine.get_sensor("lidar").broad_detectors) == 1
        env.engine.get_sensor("lidar_new").perceive(
            env.agent,
            physics_world=env.engine.physics_world.dynamic_world,
            num_lasers=env.agent.config["lidar"]["num_lasers"],
            distance=100.5,
            detector_mask=None
        )
        assert 100 in env.engine.get_sensor("lidar_new").broad_detectors
        assert 50 not in env.engine.get_sensor("lidar_new").broad_detectors
        assert len(env.engine.get_sensor("lidar_new").broad_detectors) == 1
        for i in range(1, 10):
            o, r, tm, tc, info = env.step([0, 1])
    finally:
        env.close()


if __name__ == '__main__':
    test_sensor_config()
