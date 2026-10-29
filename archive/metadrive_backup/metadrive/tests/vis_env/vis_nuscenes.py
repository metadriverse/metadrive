import time
from panda3d.core import PNMImage
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.instance_camera import InstanceCamera

NuScenesEnv = ScenarioEnv

if __name__ == "__main__":
    env = NuScenesEnv(
        {
            "use_render": True,
            "agent_policy": ReplayEgoCarPolicy,
            # "manual_control": True,
            "show_interface": True,
            # "debug_static_world": True,
            # "need_lane_localization": False,
            # "image_observation": True,
            "show_logo": False,
            "no_traffic": False,
            "store_data": False,
            "sequential_seed": True,
            # "pstats": True,
            # "debug_static_world": True,
            # "sequential_seed": True,
            # "reactive_traffic": True,
            "curriculum_level": 1,
            "show_fps": True,
            "show_sidewalk": True,
            "show_crosswalk": True,
            # "show_coordinates": True,
            "sensors": {
                "semantic": (SemanticCamera, 400, 300),
                "depth": (DepthCamera, 400, 300),
                "rgb": (RGBCamera, 400, 300),
            },
            # "pstats": True,
            # "use_mesh_terrain": True,
            # "debug": True,
            # "no_static_vehicles": False,
            # "pstats": True,
            # "render_pipeline": True,
            # "window_size": (1600, 900),
            "camera_dist": 9,
            "interface_panel": ["semantic", "depth", "rgb"],
            "start_scenario_index": 0,
            "num_scenarios": 10,
            # "force_reuse_object_name": True,
            "horizon": 1000,
            "vehicle_config": dict(
                # light=True,
                # random_color=True,
                show_navi_mark=False,
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=12, distance=50)
            ),
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
            # "drivable_area_extension": 0,
        }
    )

    # 0,1,3,4,5,6

    success = []
    reset_num = 0
    start = time.time()
    reset_used_time = 0
    s = 0
    # while True:
    # for i in range(10):
    start_reset = time.time()
    env.reset(seed=0)

    def reload_shader():
        env.engine.pbrpipe._recompile_pbr()
        env.engine.pssm.set_shader_inputs(env.engine.render)

    env.engine.accept("`", reload_shader)
    env.engine.accept("9", env.engine.terrain.reload_terrain_shader)
    env.engine.accept("0", env.engine.bufferViewer.toggleEnable)

    reset_used_time += time.time() - start_reset
    reset_num += 1
    for t in range(1000000):
        o, r, tm, tc, info = env.step([1, 0.88])
        env.engine.terrain.origin.set_shader_input('is_terrain', 1)
        assert env.observation_space.contains(o)
        s += 1
        # if env.config["use_render"]:
        #     env.render(
        #         text={
        #             "seed": env.current_seed,
        #             "num_map": info["num_stored_maps"],
        #             "data_coverage": info["data_coverage"],
        #             "reward": r,
        #             "heading_r": info["step_reward_heading"],
        #             "lateral_r": info["step_reward_lateral"],
        #             "smooth_action_r": info["step_reward_action_smooth"]
        #         },
        #         # mode="topdown"
        #     )
        # if tm or tc:
        #     print(
        #         "Time elapse: {:.4f}. Average FPS: {:.4f}, AVG_Reset_time: {:.4f}".format(
        #             time.time() - start, s / (time.time() - start - reset_used_time), reset_used_time / reset_num
        #         )
        #     )
        #     print("seed:{}, success".format(env.engine.global_random_seed))
        #     print(list(env.engine.curriculum_manager.recent_success.dict.values()))
        #     break
