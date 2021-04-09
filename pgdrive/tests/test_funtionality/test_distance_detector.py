from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.constants import BodyName
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
                "show_lidar": True,
                "side_detector": dict(num_lasers=2, distance=50),
                "lane_line_detector": dict(num_lasers=2, distance=50),
            }
        }
    )
    try:
        env.reset()
        objs = env.vehicle.side_detector.get_detected_objects() + env.vehicle.lane_line_detector.get_detected_objects()
        yellow = 0
        for obj in objs:
            if obj.getNode().getName() == BodyName.Yellow_continuous_line:
                yellow += 1
        assert yellow == 2, "side detector and lane detector broken"
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
