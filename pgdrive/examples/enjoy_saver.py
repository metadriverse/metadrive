from pgdrive.envs.pgdrive_env import PGDriveEnvV2

if __name__ == "__main__":
    env = PGDriveEnvV2(
        {
            "environment_num": 1,
            "traffic_density": 0.3,
            "start_seed": 5,
            # "controller": "joystick",
            "manual_control": True,
            "use_render": True,
            "vehicle_config": {
                "use_saver": True
            },
            "map": 16
        }
    )

    o = env.reset()
    env.pg_world.force_fps.toggle()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        text = {"save": info["takeover_start"]}
        env.render(text=text)
        if info["arrive_dest"]:
            env.reset()
    env.close()
