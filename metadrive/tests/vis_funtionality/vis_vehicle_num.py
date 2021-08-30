from metadrive.envs.metadrive_env import MetaDriveEnv

if __name__ == "__main__":
    env = MetaDriveEnv({
        "environment_num": 100,
        "start_seed": 5000,
        "traffic_density": 0.08,
    })
    env.reset()
    count = []
    for i in range(1, 101):
        o, r, d, info = env.step([0, 1])
        env.reset()
        print(
            "Current map {}, vehicle number {}.".format(env.current_seed, env.engine.traffic_manager.get_vehicle_num())
        )
        count.append(env.engine.traffic_manager.get_vehicle_num())
    print(min(count), sum(count) / len(count), max(count))
    env.close()
