from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.constants import MetaDriveType
from metadrive.constants import DEFAULT_AGENT
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger
from metadrive.component.sensors.lidar import Lidar


def test_original_lidar(render=False):
    setup_logger(debug=True)
    env = MetaDriveEnv(
        {
            "use_render": render,
            "manual_control": render,
            "num_scenarios": 1,
            "traffic_density": 0.3,
            "vehicle_config": {
                "show_lidar": True,
                "side_detector": dict(num_lasers=2, distance=50),
                "lane_line_detector": dict(num_lasers=2, distance=50),
            },
            "map": "XXX"
        }
    )
    Lidar._disable_detector_mask = True
    try:
        env.reset()
        v_config = env.config["vehicle_config"]
        v_config["spawn_longitude"] = 0
        v_config["spawn_lateral"] = 7.5
        v_config["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)
        another_v = DefaultVehicle(v_config, random_seed=0)
        another_v.reset()
        objs = env.engine.get_sensor("side_detector").perceive(env.agent,
                                                               env.agent.engine.physics_world.static_world,
                                                               num_lasers=2,
                                                               distance=50
                                                               ).detected_objects + \
               env.engine.get_sensor("lane_line_detector").perceive(
                   env.agent,
                   env.agent.engine.physics_world.static_world,
                   num_lasers=2,
                   distance=50
               ).detected_objects
        yellow = 0
        for obj in objs:
            if obj.getNode().getName() == MetaDriveType.LINE_SOLID_SINGLE_YELLOW:
                yellow += 1
        assert yellow == 2, "side detector and lane detector broken"
        detect_traffic_vehicle = False
        detect_base_vehicle = False
        for i in range(1, 1000):
            o, r, tm, tc, info = env.step([0, 1])
            if len(env.agent.lidar.get_surrounding_vehicles(env.observations[DEFAULT_AGENT].detected_objects)) > 2:
                detect_traffic_vehicle = True
            for hit in env.observations[DEFAULT_AGENT].detected_objects:
                if isinstance(hit, BaseVehicle):
                    detect_base_vehicle = True
            if tm or tc:
                break
        # if not (detect_traffic_vehicle and detect_base_vehicle):
        #     print("Lidar detection failed")
        assert detect_traffic_vehicle and detect_base_vehicle, "Lidar detection failed"
    finally:
        env.close()


def test_lidar_with_mask(render=False):
    setup_logger(debug=True)
    env = MetaDriveEnv(
        {
            "use_render": render,
            "manual_control": render,
            "num_scenarios": 1,
            "traffic_density": 0.3,
            "vehicle_config": {
                "show_lidar": True,
                "side_detector": dict(num_lasers=2, distance=50),
                "lane_line_detector": dict(num_lasers=2, distance=50),
            },
            "map": "XXX"
        }
    )
    try:
        env.reset()
        v_config = env.config["vehicle_config"]
        v_config["spawn_longitude"] = 0
        v_config["spawn_lateral"] = 7.5
        v_config["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)
        another_v = DefaultVehicle(v_config, random_seed=0)
        another_v.reset()
        # for test
        objs = env.engine.get_sensor("side_detector").perceive(env.agent,
                                                               env.agent.engine.physics_world.static_world,
                                                               num_lasers=2,
                                                               distance=50
                                                               ).detected_objects + \
               env.engine.get_sensor("lane_line_detector").perceive(
                   env.agent,
                   env.agent.engine.physics_world.static_world,
                   num_lasers=2,
                   distance=50
               ).detected_objects
        yellow = 0
        for obj in objs:
            if obj.getNode().getName() == MetaDriveType.LINE_SOLID_SINGLE_YELLOW:
                yellow += 1
        assert yellow == 2, "side detector and lane detector broken"
        detect_traffic_vehicle = False
        detect_base_vehicle = False
        for i in range(1, 1000):
            o, r, tm, tc, info = env.step([0, 1])
            if len(env.agent.lidar.get_surrounding_vehicles(env.observations[DEFAULT_AGENT].detected_objects)) > 2:
                detect_traffic_vehicle = True
            for hit in env.observations[DEFAULT_AGENT].detected_objects:
                if isinstance(hit, BaseVehicle):
                    detect_base_vehicle = True
            if tm or tc:
                break
        # if not (detect_traffic_vehicle and detect_base_vehicle):
        #     print("Lidar detection failed")
        assert detect_traffic_vehicle and detect_base_vehicle, "Lidar detection failed"
    finally:
        env.close()


if __name__ == "__main__":
    # test_lidar_with_mask(render=True)
    test_original_lidar(render=False)
