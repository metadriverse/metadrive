from pgdrive import PGDriveEnv
from pgdrive.tests.test_pgdrive_env import _act


def test_config_consistency():
    env = PGDriveEnv({"vehicle_config": {"lidar": {"num_lasers": 999}}})
    try:
        env.reset()
        assert env.vehicle.vehicle_config["lidar"]["num_lasers"] == 999
    finally:
        env.close()


if __name__ == '__main__':
    test_config_consistency()
