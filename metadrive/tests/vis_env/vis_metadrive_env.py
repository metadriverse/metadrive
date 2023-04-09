import numpy as np
from metadrive.utils.math_utils import clip, norm
from metadrive.component.vehicle_module.mini_map import MiniMap
from metadrive.component.vehicle_module.rgb_camera import RGBCamera
from metadrive.component.vehicle_module.vehicle_panel import VehiclePanel

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger

if __name__ == "__main__":
    setup_logger(True)
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "start_seed": 2,
            # "_disable_detector_mask":True,
            # "debug_physics_world": True,
            "debug": False,
            # "global_light": False,
            # "debug_static_world": True,
            "cull_scene": False,
            # "image_observation": True,
            # "controller": "joystick",
            # "show_coordinates": True,
            "manual_control": True,
            "use_render": True,
            "accident_prob": 1,
            "decision_repeat": 5,
            "interface_panel": [MiniMap, VehiclePanel, RGBCamera],
            "need_inverse_traffic": False,
            "rgb_clip": True,
            "map": "SC",
            # "agent_policy": IDMPolicy,
            "random_traffic": False,
            "random_lane_width": True,
            # "random_agent_model": True,
            "driving_reward": 1.0,
            "force_destroy": False,
            # "window_size": (500, 800),
            # "camera_dist": -1,
            # "camera_pitch": 30,
            # "camera_height": 1,
            # "camera_smooth": False,
            # "camera_height": -1,
            "show_coordinates": True,
            "vehicle_config": {
                "enable_reverse": False,
                # "vehicle_model": "xl",
                # "rgb_camera": (1024, 1024),
                # "spawn_velocity": [8.728615581032535, -0.24411703918728195],
                "spawn_velocity_car_frame": True,
                # "image_source": "depth_camera",
                # "random_color": True
                # "show_lidar": True,
                "spawn_lane_index": None,
                # "destination":"2R1_3_",
                # "show_side_detector": True,
                # "show_lane_line_detector": True,
                # "side_detector": dict(num_lasers=2, distance=50),
                # "lane_line_detector": dict(num_lasers=2, distance=50),
                "show_line_to_navi_mark": True,
                "show_navi_mark": True,
                "show_dest_mark": True
            },
        }
    )
    import time

    init_state = {
        'position': (40.82264362985734, -509.3641208712943),
        'heading': -89.41878393159747,
        'velocity': [8.728615581032535, -0.24411703918728195],
        'valid': True
    }

    start = time.time()
    from metadrive.component.vehicle_module.rgb_camera import RGBCamera
    o = env.reset()
    # env.vehicle.get_camera("rgb_camera").save_image(env.vehicle)
    # for line in env.engine.coordinate_line:
    #     line.reparentTo(env.vehicle.origin)
    # env.vehicle.set_velocity([5, 0], in_local_frame=True)
    for s in range(1, 10000):
        # env.vehicle.set_velocity([8.728615581032535, -2.24411703918728195], in_local_frame=True)
        o, r, d, info = env.step(env.action_space.sample())

        # else:
        # if s % 100 == 0:
        #     env.close()
        #     env.reset()
        # info["fuel"] = env.vehicle.energy_consumption
        vehicle = env.vehicle
        heading_dir_last = vehicle.last_heading_dir
        heading_dir_now = vehicle.heading
        cos_beta = heading_dir_now.dot(heading_dir_last) / (norm(*heading_dir_now) * norm(*heading_dir_last))
        beta_diff = np.arccos(clip(cos_beta, 0.0, 1.0))
        yaw_rate = beta_diff / 0.1
        env.render(
            text={
                "heading_diff": env.vehicle.heading_diff(env.vehicle.lane),
                "left_side, right_side": (env.vehicle.dist_to_left_side, env.vehicle.dist_to_right_side),
                "position": env.vehicle.position,
                "yaw_rate": yaw_rate,
                "long, lateral": env.vehicle.lane.local_coordinates(env.vehicle.position),
                # "current_seed": env.current_seed
            }
        )
        # if d:
        #     env.reset()
        # # assert env.observation_space.contains(o)
        # if (s + 1) % 100 == 0:
        #     # print(
        #         "Finish {}/10000 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}".format(
        #             s + 1,f
        #             time.time() - start, (s + 1) / (time.time() - start)
        #         )
        #     )
        # if d:
        # #     # env.close()
        # #     # print(len(env.engine._spawned_objects))
        # env.reset()
