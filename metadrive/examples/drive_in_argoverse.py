from envs.argoverse_env import ArgoverseEnv

if __name__ == "__main__":
    env = ArgoverseEnv({"manual_control": True, "use_render": True})

    o = env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step([1.0, 0.])
    env.close()
