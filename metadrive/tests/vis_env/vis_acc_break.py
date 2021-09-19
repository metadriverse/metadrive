from metadrive.envs.metadrive_env import MetaDriveEnv

if __name__ == "__main__":
    config = {
        "environment_num": 10,
        "traffic_density": .0,
        # "use_render":True,
        "map": "SSSSS",
        # "manual_control":True,
        "controller": "joystick",
        "random_agent_model": False,
        "vehicle_config": {
            "vehicle_model": "default",
            # "vehicle_model":"s",
            # "vehicle_model":"m",
            # "vehicle_model":"l",
            # "vehicle_model":"xl",
        }
    }
    env = MetaDriveEnv(config)
    import time

    start = time.time()
    o = env.reset()
    a = [.0, 1.]
    for s in range(1, 100000):
        o, r, d, info = env.step(a)
        if env.vehicle.speed > 100:
            a = [0, -1]
            print("0-100 km/h acc use time:{}".format(s * 0.1))
            pre_pos = env.vehicle.position[0]
        if a == [0, -1] and env.vehicle.speed < 1:
            print("0-100 brake use dist:{}".format(env.vehicle.position[0] - pre_pos))
            break
    env.close()
