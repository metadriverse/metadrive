from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv


def test_safe_env(vis=False):
    config = {"environment_num": 100, "start_seed": 75}
    if vis:
        config["vehicle_config"] = {"show_line_to_navi_mark": True}
        config["use_render"] = True
        config["manual_control"] = True
        config["controller"] = "joystick"

    env = SafeMetaDriveEnv(config)
    try:
        o = env.reset()
        total_cost = 0
        for ep in range(5):
            for i in range(1, 100):
                o, r, d, info = env.step([0, 1])
                total_cost += info["cost"]
                assert env.observation_space.contains(o)
                if d:
                    total_cost = 0
                    print("Reset")
                    env.reset()
        env.close()
    finally:
        env.close()


if __name__ == '__main__':
    test_safe_env(vis=True)
