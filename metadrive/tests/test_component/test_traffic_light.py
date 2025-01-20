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
                "show_dest_mark": True,
                'use_traffic_light_controller': True
            },
            'use_traffic_light_controller': True
        }
    )
    env.reset()
    if debug:
        env.engine.toggleDebug()
    try:
        # green
        env.reset()
        light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        light.set_green()
        test_success = False
        for s in range(1, 100):
            env.step([0, 1])
            if env.agent.green_light:
                print('[Successfully detected] Green light')
                test_success = True
                print('Go through the green light')
                # break
        assert test_success
        light.destroy()

        # red test
        env.reset()
        light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        # dummy_vehicle = env.engine.spawn_object(DummyVehicle, vehicle_config={}, position=light.position)
        # dummy_vehicle.set_static(True)
        # if not hasattr(env.engine, "dummy_vehicle"):
        #     env.engine.dummy_vehicle = [dummy_vehicle.id]
        # else:
        #     env.engine.dummy_vehicle.append(dummy_vehicle.id)
        light.set_red()
        test_success = False
        for s in range(1, 100):
            env.step([0, 1])
        light.set_green()
        for s in range(1, 100):
            env.step([0, 1])
        light.destroy()
        # env.engine.dummy_vehicle.remove(dummy_vehicle.id)
        # env.engine.clear_objects([dummy_vehicle.id], force_destroy=True)

        # yellow
        # env.reset()
        # light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        # light.set_yellow()
        # test_success = False
        # for s in range(1, 100):
        #     env.step([0, 1])
        #     if env.agent.yellow_light:
        #         print('[Successfully detected] Yellow light')
        #         test_success = True
        #         # break
        # assert test_success
        # light.destroy()

    finally:
        env.close()


if __name__ == "__main__":
    test_traffic_light(render=True, manual_control=False, debug=False)
