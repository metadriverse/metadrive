from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.envs.metadrive_env import MetaDriveEnv


def vis_traffic_light(render=True, manual_control=False, debug=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "manual_control": manual_control,
            "use_render": render,
            "debug": debug,
            "debug_static_world": False,
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
        test_success = False
        for s in range(1, 1000):
            if s < 100:
                light.set_green()
            elif s < 300:
                light.set_red()
            else:
                light.set_yellow()
            env.step([0, 1])
        light.destroy()

        # red test
        light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        light.set_red()
        test_success = False
        for s in range(1, 1000):
            env.step([0, 1])
        light.destroy()
        # yellow
        env.reset()
        light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        light.set_yellow()
        test_success = False
        for s in range(1, 1000):
            env.step([0, 1])
        light.destroy()

    finally:
        env.close()


if __name__ == "__main__":
    vis_traffic_light(True, manual_control=True, debug=True)
