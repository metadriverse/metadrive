from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.envs.metadrive_env import MetaDriveEnv


def test_traffic_light(render=False, manual_control=False, debug=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "manual_control": manual_control,
            "use_render": render,
            "debug": debug,
            "debug_static_world": debug,
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
        test_success = False
        for s in range(1, 100):
            env.step([0, 1])
            if env.agent.green_light:
                test_success = True
                break
        assert test_success
        light.destroy()

        # red test
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


if __name__ == "__main__":
    test_traffic_light(True, manual_control=True)
