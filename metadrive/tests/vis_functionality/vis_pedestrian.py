from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import setup_logger
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.traffic_participants.cyclist import Cyclist

if __name__ == "__main__":
    # setup_logger(True)
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "start_seed": 22,
            "accident_prob": 1.0,
            # "debug_physics_world": True,
            "debug": True,
            # "debug_static_world": True,
            # "image_observation": True,
            "manual_control": True,
            "use_render": True,
            "decision_repeat": 5,
            "need_inverse_traffic": False,
            "norm_pixel": True,
            "map": "XSS",
            # "agent_policy": IDMPolicy,
            "random_traffic": False,
            "random_lane_width": True,
            "driving_reward": 1.0,
            "show_interface": True,
            "force_destroy": False,
            # "camera_dist": -1,
            # "camera_pitch": 30,
            # "camera_height": 1,
            # "camera_smooth": False,
            # "camera_height": -1,
            # "window_size": (2400, 1600),
            "show_coordinates": True,
            "vehicle_config": {
                "enable_reverse": True,
                # "show_lidar": True
                # "image_source": "depth_camera",
                # "random_color": True
                # "show_lidar": True,
                # "spawn_lane_index":("1r1_0_", "1r1_1_", 0),
                # "destination":"2R1_3_",
                # "show_side_detector": True,
                # "show_lane_line_detector": True,
                # "side_detector": dict(num_lasers=2, distance=50),
                # "lane_line_detector": dict(num_lasers=2, distance=50),
                # # "show_line_to_dest": True,
                # "show_dest_mark": True
            },
        }
    )
    import time

    start = time.time()
    o, _ = env.reset()
    obj_1 = env.engine.spawn_object(Pedestrian, position=[30, 0], heading_theta=0, random_seed=1)
    obj_2 = env.engine.spawn_object(Pedestrian, position=[30, 6], heading_theta=0, random_seed=1)
    c_1 = env.engine.spawn_object(Cyclist, position=[30, 8], heading_theta=0, random_seed=1)
    obj_1.set_velocity([1, 0], 1, in_local_frame=True)
    # obj_1.show_coordinates()
    obj_2.set_velocity([1, 0], 2, in_local_frame=True)
    c_1.set_velocity([3, 0], 2, in_local_frame=True)
    # obj_2.show_coordinates()
    env.agent.set_velocity([10, 0], in_local_frame=False)
    for s in range(1, 10000):
        # print(c_1.heading_theta)
        o, r, tm, tc, info = env.step(env.action_space.sample())
        # obj_1.set_velocity([1, 0], 2, in_local_frame=True)
        # obj_2.set_velocity([1, 0], 0.8, in_local_frame=True)
        if s == 300:
            obj_1.set_velocity([1, 0], 0, in_local_frame=True)
            # obj_2.set_velocity([1, 0], 0, in_local_frame=True)
        elif s == 500:
            obj_1.set_velocity([1, 0], 1, in_local_frame=True)
        # else:
        #     obj_1.set_velocity([1, 0], 1, in_local_frame=True)

        if 100 < s < 300:
            obj_2.set_velocity([1, 0], 2, in_local_frame=True)
        elif 500 > s > 300:
            # print("here stop")
            obj_2.set_velocity([1, 0], 0, in_local_frame=True)
        elif s >= 500:
            obj_2.set_velocity([1, 0], 2, in_local_frame=True)

        # else:
        # if s % 100 == 0:
        #     env.close()
        #     env.reset()
        # info["fuel"] = env.agent.energy_consumption
        env.render(
            text={
                "heading_diff": env.agent.heading_diff(env.agent.lane),
                "lane_width": env.agent.lane.width,
                "lateral": env.agent.lane.local_coordinates(env.agent.position),
                "current_seed": env.current_seed,
                "step": s,
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
