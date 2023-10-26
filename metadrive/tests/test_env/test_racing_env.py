from metadrive.policy.idm_policy import IDMPolicy
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.racing_env import RacingEnv


def test_racing_env_idm(render=False):

    racing_config =  dict(
        # controller="joystick",
        # num_agents=2,
        use_render=False,
        manual_control=True,
        traffic_density=0,
        num_scenarios=10000,
        random_agent_model=False,
        top_down_camera_initial_x=95,
        top_down_camera_initial_y=15,
        top_down_camera_initial_z=120,
        # random_lane_width=True,
        # random_lane_num=True,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(show_lidar=False, show_navi_mark=False),
    )

    env = RacingEnv(racing_config)
    pass_test = False
    try:
        o, _ = env.reset()
        env.vehicle.set_velocity([1, 0.1], 10)
        # print(env.vehicle.speed)
        for s in range(1, 10000):
            o, r, tm, tc, info = env.step(env.action_space.sample())
            _, lat = env.vehicle.lane.local_coordinates(env.vehicle.position)
            if (tm or tc) and info["arrive_dest"]:
                break
        pass_test = True
        env.close()
    finally:
        env.close()
        assert pass_test



if __name__ == "__main__":
    test_racing_env_idm()
