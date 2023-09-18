from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.lidar import Lidar
from metadrive.envs.metadrive_env import MetaDriveEnv


def test_sensor_config():
    env = MetaDriveEnv({
        "sensors": {
            "lidar": (Lidar, 100)
        },
    })
    try:
        env.reset()
        assert env.engine.get_sensor("lidar").broad_phase_distance == 100
        for i in range(1, 10):
            o, r, tm, tc, info = env.step([0, 1])
    finally:
        env.close()

    env = MetaDriveEnv({
        "sensors": {
            "lidar_new": (Lidar, 100)
        },
    })
    try:
        env.reset()
        assert env.engine.get_sensor("lidar").broad_phase_distance == 50
        assert env.engine.get_sensor("lidar_new").broad_phase_distance == 100
        for i in range(1, 10):
            o, r, tm, tc, info = env.step([0, 1])
    finally:
        env.close()

    try:
        env.reset()
        assert env.engine.get_sensor("lidar").broad_phase_distance == 50
        assert env.engine.get_sensor("lidar_new").broad_phase_distance == 100
        for i in range(1, 10):
            o, r, tm, tc, info = env.step([0, 1])
    finally:
        env.close()


if __name__ == '__main__':
    test_sensor_config()
