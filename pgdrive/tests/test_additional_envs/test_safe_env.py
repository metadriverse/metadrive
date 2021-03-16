from pgdrive.envs.generation_envs.safe_pgdrive_env import SafePGDriveEnv


def test_safe_env():
    env = SafePGDriveEnv({"environment_num": 100, "start_seed": 75})
    o = env.reset()
    total_cost = 0
    for i in range(1, 100):
        o, r, d, info = env.step([0, 1])
        total_cost += info["cost"]
        if d:
            total_cost = 0
            print("Reset")
            env.reset()
    env.close()
