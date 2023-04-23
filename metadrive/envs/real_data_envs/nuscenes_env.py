from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

NuScenesEnv = ScenarioEnv

if __name__ == "__main__":
    env = NuScenesEnv(
        {
            "use_render": True,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "show_interface": False,
            "show_logo": False,
            "reactive_traffic": True,
            "show_fps": False,
            "debug": False,
            # "no_traffic": True,
            # "no_light": False,
            # "debug":True,
            # "no_traffic":True,
            # "start_scenario_index": 192,
            # "start_scenario_index": 1000,
            "num_scenarios": 10,
            # "force_reuse_object_name": True,
            # "data_directory": "/home/shady/Downloads/test_processed",
            "horizon": 1000,
            "no_static_vehicles": True,
            "show_policy_mark": True,
            # "show_coordinates": True,
            "vehicle_config": dict(
                show_navi_mark=False,
                no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
            "data_directory": AssetLoader.file_path("nuscenes", return_raw_style=False),
        }
    )
    success = []
    for i in [0,1,2, 3, 5, 6, 7, 8, 9]:
        env.reset(force_seed=i)
        for t in range(10000):
            o, r, d, info = env.step([0, 0])
            assert env.observation_space.contains(o)
            c_lane = env.vehicle.lane
            long, lat, = c_lane.local_coordinates(env.vehicle.position)
            # if env.config["use_render"]:
            env.render(
                text={
                    # "obs_shape": len(o),
                    # "lateral": env.observations["default_agent"].lateral_dist,
                    "seed": i,
                    # "reward": r,
                }
                # mode="topdown"
            )

            if d and info["arrive_dest"]:
                print("seed:{}, success".format(env.engine.global_random_seed))
                break
