from pgdrive.component.vehicle.vehicle_type import DefaultVehicle
from pgdrive.constants import BodyName
from pgdrive.constants import DEFAULT_AGENT
from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import setup_logger


class TestEnv(PGDriveEnv):
    def __init__(self, config):
        super(TestEnv, self).__init__(config)


def test_original_lidar(render=False):
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
            },
            "_disable_detector_mask": True,
            "map": "XXX"
        }
    )
    try:
        env.reset()
        v_config = env.config["vehicle_config"]
        v_config["spawn_longitude"] = 0
        v_config["spawn_lateral"] = 7.5
        another_v = DefaultVehicle(v_config, random_seed=0)
        another_v.reset()
        objs = env.vehicle.side_detector.perceive(env.vehicle, env.vehicle.engine.physics_world.static_world
                                                  ).detected_objects + env.vehicle.lane_line_detector.perceive(
                                                      env.vehicle, env.vehicle.engine.physics_world.static_world
                                                  ).detected_objects
        yellow = 0
        for obj in objs:
            if obj.getNode().getName() == BodyName.Yellow_continuous_line:
                yellow += 1
        assert yellow == 2, "side detector and lane detector broken"
        detect_traffic_vehicle = False
        detect_base_vehicle = False
        for i in range(1, 100000):
            o, r, d, info = env.step([0, 1])
            if len(env.vehicle.lidar.get_surrounding_vehicles(env.observations[DEFAULT_AGENT].detected_objects)) > 2:
                detect_traffic_vehicle = True
            for hit in env.observations[DEFAULT_AGENT].detected_objects:
                v = hit.getNode()
                if v.hasPythonTag(BodyName.Vehicle):
                    detect_base_vehicle = True
            if d:
                break
        if not (detect_traffic_vehicle and detect_base_vehicle):
            print("Lidar detection failed")
        assert detect_traffic_vehicle and detect_base_vehicle, "Lidar detection failed"
    finally:
        env.close()


def test_lidar_with_mask(render=False):
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
            },
            "_disable_detector_mask": False,
            "map": "XXX"
        }
    )
    try:
        env.reset()
        v_config = env.config["vehicle_config"]
        v_config["spawn_longitude"] = 0
        v_config["spawn_lateral"] = 7.5
        another_v = DefaultVehicle(v_config, random_seed=0)
        another_v.reset()
        # for test
        env.agent_manager._pending_objects[another_v.name] = another_v
        objs = env.vehicle.side_detector.perceive(env.vehicle, env.vehicle.engine.physics_world.static_world
                                                  ).detected_objects + env.vehicle.lane_line_detector.perceive(
                                                      env.vehicle, env.vehicle.engine.physics_world.static_world
                                                  ).detected_objects
        yellow = 0
        for obj in objs:
            if obj.getNode().getName() == BodyName.Yellow_continuous_line:
                yellow += 1
        assert yellow == 2, "side detector and lane detector broken"
        detect_traffic_vehicle = False
        detect_base_vehicle = False
        for i in range(1, 100000):
            o, r, d, info = env.step([0, 1])
            if len(env.vehicle.lidar.get_surrounding_vehicles(env.observations[DEFAULT_AGENT].detected_objects)) > 2:
                detect_traffic_vehicle = True
            for hit in env.observations[DEFAULT_AGENT].detected_objects:
                v = hit.getNode()
                if v.hasPythonTag(BodyName.Vehicle):
                    detect_base_vehicle = True
            if d:
                break
        if not (detect_traffic_vehicle and detect_base_vehicle):
            print("Lidar detection failed")
        assert detect_traffic_vehicle and detect_base_vehicle, "Lidar detection failed"
    finally:
        env.close()


if __name__ == "__main__":
    test_lidar_with_mask(render=False)
    test_original_lidar(render=False)
