from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy


def test_traffic_light_state_check(render=False, manual_control=False, debug=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "manual_control": manual_control,
            "use_render": render,
            "debug": debug,
            "map": "X",
            "window_size": (1200, 800),
            "vehicle_config": {
                "enable_reverse": True,
                "show_dest_mark": True
            },
        }
    )
    env.reset()
    try:
        # green
        env.reset()
        light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        light.set_green()
        test_success = True
        for s in range(1, 100):
            env.step([0, 1])
            if env.agent.red_light or env.agent.yellow_light:
                test_success = False
                break
        assert test_success
        light.destroy()

        # red test
        env.reset()
        light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        light.set_red()
        test_success = False
        for s in range(1, 100):
            env.step([0, 1])
            if env.agent.red_light:
                test_success = True
                break
        assert test_success
        light.destroy()

        # yellow
        env.reset()
        light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        light.set_yellow()
        test_success = False
        for s in range(1, 100):
            env.step([0, 1])
            if env.agent.yellow_light:
                test_success = True
                break
        assert test_success
        light.destroy()

    finally:
        env.close()


def test_traffic_light_detection(render=False, manual_control=False, debug=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "manual_control": manual_control,
            "use_render": render,
            "debug": debug,
            "map": "X",
            "window_size": (1200, 800),
            "vehicle_config": {
                "enable_reverse": True,
                "show_dest_mark": True
            },
        }
    )
    env.reset()
    try:
        # green
        env.reset()
        light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        light.set_green()
        test_success = True
        for s in range(1, 100):
            env.step([0, 1])
            if min(env.observations["default_agent"].cloud_points) < 0.99:
                test_success = False
                break
            assert len(env.observations["default_agent"].detected_objects) == 0
        assert test_success
        light.destroy()

        # red test
        env.reset()
        light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        light.set_red()
        test_success = False
        for s in range(1, 100):
            env.step([0, 1])
            if min(env.observations["default_agent"].cloud_points) < 0.5:
                test_success = True
                assert list(env.observations["default_agent"].detected_objects)[0].status == BaseTrafficLight.LIGHT_RED
                break
        assert test_success
        light.destroy()

        # yellow
        env.reset()
        light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        light.set_yellow()
        test_success = False
        for s in range(1, 100):
            env.step([0, 1])
            if min(env.observations["default_agent"].cloud_points) < 0.5:
                test_success = True
                assert list(env.observations["default_agent"].detected_objects
                            )[0].status == BaseTrafficLight.LIGHT_YELLOW
                break
        assert test_success
        light.destroy()

    finally:
        env.close()


def test_idm_policy(render=False, debug=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "agent_policy": IDMPolicy,
            "use_render": render,
            "debug": debug,
            "map": "X",
            "window_size": (1200, 800),
            "show_coordinates": True,
            "vehicle_config": {
                "show_lidar": True,
                "enable_reverse": True,
                "show_dest_mark": True
            },
        }
    )
    env.reset()
    try:
        # green
        env.reset()
        light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        light.set_green()
        for s in range(1, 1000):
            if s == 30:
                light.set_yellow()
            elif s == 90:
                light.set_red()
            env.step([0, 1])
            if env.vehicle.red_light or env.vehicle.yellow_light:
                raise ValueError("Vehicle should not stop at red light!")
        assert env.vehicle.speed < 0.1

        # move
        light.set_green()
        test_success = False
        for s in range(1, 1000):
            o, r, d, t, i = env.step([0, 1])
            if i["arrive_dest"]:
                test_success = True
                break
        light.destroy()
        assert test_success
    finally:
        env.close()


if __name__ == "__main__":
    # test_traffic_light_state_check(True, manual_control=False)
    # test_traffic_light_detection(True, manual_control=False)
    test_idm_policy(True)
