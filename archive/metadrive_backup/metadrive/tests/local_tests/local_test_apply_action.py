from metadrive.envs.metadrive_env import MetaDriveEnv


def local_test_apply_action():
    try:
        env = MetaDriveEnv({"map": "SSS", "use_render": True})
        o, _ = env.reset()
        for act in [-1, 1]:
            for _ in range(300):
                assert env.observation_space.contains(o)
                o, r, tm, tc, i = env.step([act, 1])
                if tm or tc:
                    o, _ = env.reset()
                    break
        env.close()
    finally:
        if "env" in locals():
            env = locals()["env"]
            env.close()


if __name__ == '__main__':
    local_test_apply_action()
