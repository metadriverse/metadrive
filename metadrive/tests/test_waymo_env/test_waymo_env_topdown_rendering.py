from metadrive.envs.real_data_envs.waymo_env import WaymoEnv


def test_waymo_env_topdown_rendering(stop_at_end=False):
    for env in [WaymoEnv({"case_num": 3})]:
        try:
            o = env.reset()
            for i in range(1, 1000000 if stop_at_end else 1000):
                o, r, d, info = env.step([1.0, 0.])
                env.render("topdown")
                if d and not stop_at_end:
                    break
        finally:
            env.close()


if __name__ == "__main__":
    test_waymo_env_topdown_rendering(stop_at_end=True)
