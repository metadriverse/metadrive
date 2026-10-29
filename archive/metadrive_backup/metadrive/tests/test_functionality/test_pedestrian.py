from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.envs.metadrive_env import MetaDriveEnv


def test_pedestrian(render=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "start_seed": 22,
            "debug": False,
            "manual_control": False,
            "use_render": render,
            "decision_repeat": 5,
            "need_inverse_traffic": False,
            "norm_pixel": True,
            "map": "X",
            # "agent_policy": IDMPolicy,
            "random_traffic": False,
            "random_lane_width": True,
            # "random_agent_model": True,
            "driving_reward": 1.0,
            "force_destroy": False,
            # "camera_dist": -1,
            # "camera_pitch": 30,
            # "camera_height": 1,
            # "camera_smooth": False,
            # "camera_height": -1,
            "window_size": (2400, 1600),
            "vehicle_config": {
                "enable_reverse": False,
            },
        }
    )
    env.reset()
    try:
        obj_1 = env.engine.spawn_object(Pedestrian, position=[30, 0], heading_theta=0, random_seed=1)
        obj_2 = env.engine.spawn_object(Pedestrian, position=[30, 6], heading_theta=0, random_seed=1)
        obj_1.set_velocity([1, 0], 1, in_local_frame=True)
        obj_2.set_velocity([1, 0], 0, in_local_frame=True)
        env.agent.set_velocity([5, 0], in_local_frame=False)
        for s in range(1, 1000):
            o, r, tm, tc, info = env.step([0, 0])
            # obj_1.set_velocity([1, 0], 2, in_local_frame=True)
            # obj_2.set_velocity([1, 0], 0.8, in_local_frame=True)
            if s == 300:
                obj_1.set_velocity([1, 0], 0, in_local_frame=True)
                # obj_2.set_velocity([1, 0], 0, in_local_frame=True)
            elif s == 500:
                obj_1.set_velocity([1, 0], 2, in_local_frame=True)

            # else:
            #     obj_1.set_velocity([1, 0], 1, in_local_frame=True)

            # if 100 < s < 300:
            #     obj_2.set_velocity([1, 0], 1, in_local_frame=True)
            # elif 500 > s > 300:
            #     obj_2.set_velocity([1, 0], 0, in_local_frame=True)
            # elif s >= 500:
            #     obj_2.set_velocity([1, 0], 2, in_local_frame=True)
        assert abs(obj_1.position[0] - 160) < 1, "Pedestrian movement error!"
    finally:
        env.close()


if __name__ == "__main__":
    test_pedestrian(True)
