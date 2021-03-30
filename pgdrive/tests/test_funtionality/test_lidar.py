from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import setup_logger


class TestEnv(PGDriveEnv):
    def __init__(self, config):
        super(TestEnv, self).__init__(config)


def test_lidar(render=False):
    setup_logger(debug=True)
    env = TestEnv(
        {
            "use_render": render,
            "manual_control": render,
            "environment_num": 1,
            "traffic_density": 0.3,
            "vehicle_config": {
                "show_lidar": True
            }
        }
    )
    try:
        env.reset()
        detect_vehicle = False
        for i in range(1, 100000):
            o, r, d, info = env.step([0, 1])
            if len(env.vehicle.lidar.get_surrounding_vehicles()) != 0:
                detect_vehicle = True

            if d:
                break
        assert detect_vehicle, "Lidar detection failed"
    finally:
        env.close()


if __name__ == "__main__":
    test_lidar(render=True)
