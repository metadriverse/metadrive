import time

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

NuScenesEnv = ScenarioEnv

if __name__ == "__main__":
    env = NuScenesEnv(
        {
            "use_render": False,
            "no_map": False,
            "agent_policy": ReplayEgoCarPolicy,
            # "manual_control": True,
            "show_interface": False,
            # "need_lane_localization": False,
            # "image_observation": True,
            "show_logo": False,
            "no_traffic": False,
            "store_data": False,
            "sequential_seed": True,
            # "debug_static_world": True,
            # "sequential_seed": True,
            "reactive_traffic": True,
            "curriculum_level": 1,
            "show_fps": False,
            # "debug": True,
            "no_static_vehicles": False,
            # "pstats": True,
            "render_pipeline": True,
            # "daytime": "22:01",
            # "no_traffic": True,
            # "no_light": False,
            # "debug":True,
            # Make video
            # "episodes_to_evaluate_curriculum": 5,
            "window_size": (1600, 900),
            "camera_dist": 9,
            # "camera_height": 0.5,
            # "camera_pitch": np.pi / 3,
            # "camera_fov": 60,
            # "force_render_fps": 10,
            "start_scenario_index": 0,
            "num_scenarios": 10,
            # "force_reuse_object_name": True,
            # "data_directory": "/home/shady/Downloads/test_processed",
            "horizon": 1000,
            # "no_static_vehicles": True,
            # "show_policy_mark": True,
            # "show_coordinates": True,
            # "force_destroy": True,
            # "default_vehicle_in_traffic": True,
            "vehicle_config": dict(
                # light=True,
                # random_color=True,
                show_navi_mark=False,
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=12, distance=50)
            ),
            "data_directory": "/home/shady/data/scenarionet/dataset/nuscenes"
        }
    )

    # 0,1,3,4,5,6

    success = []
    reset_num = 0
    start = time.time()
    reset_used_time = 0
    s = 0
    while True:
        # for i in range(10):
        start_reset = time.time()
        env.reset(seed=0)

        reset_used_time += time.time() - start_reset
        reset_num += 1
        for t in range(10000):
            o, r, tm, tc, info = env.step([1, 0.88])
            assert env.observation_space.contains(o)
            s += 1
            if env.config["use_render"]:
                env.render(
                    text={
                        "seed": env.current_seed,
                        "num_map": info["num_stored_maps"],
                        "data_coverage": info["data_coverage"],
                        "reward": r,
                        "heading_r": info["step_reward_heading"],
                        "lateral_r": info["step_reward_lateral"],
                        "smooth_action_r": info["step_reward_action_smooth"]
                    },
                    # mode="topdown"
                )
            if tm or tc:
                print(
                    "Time elapse: {:.4f}. Average FPS: {:.4f}, AVG_Reset_time: {:.4f}".format(
                        time.time() - start, s / (time.time() - start - reset_used_time), reset_used_time / reset_num
                    )
                )
                print("seed:{}, success".format(env.engine.global_random_seed))
                print(list(env.engine.curriculum_manager.recent_success.dict.values()))
                break
