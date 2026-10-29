from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.utils import setup_logger

if __name__ == "__main__":
    setup_logger(True)
    env = SafeMetaDriveEnv(
        {
            "use_render": True,
            "manual_control": True,
            "num_scenarios": 1,
            "show_coordinates": True,
            "map": "CCCC",
            "accident_prob": 1.0,
            "vehicle_config": {
                "show_lidar": True
            }
        }
    )

    o, _ = env.reset()
    # print("vehicle num", len(env.engine.traffic_manager.vehicles))
    for i in range(1, 100000):
        # Print TrafficBarrier pitch and roll.
        # from metadrive.component.static_object.traffic_object import TrafficBarrier
        # obj = [v for v in env.engine.object_manager.spawned_objects.values() if isinstance(v, TrafficBarrier)][0]
        # print(f"{obj.pitch=}, {obj.roll=}, {obj.heading_theta=}, {obj.position=}")

        o, r, tm, tc, info = env.step([0, 1])
        env.render(text={})
    env.close()
