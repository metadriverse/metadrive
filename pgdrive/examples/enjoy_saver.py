from pgdrive.envs.pgdrive_env import PGDriveEnv

if __name__ == "__main__":
    env = PGDriveEnv(
        {
            "environment_num": 1,
            "traffic_density": 0.3,
            "traffic_mode": "reborn",
            "start_seed": 5,
            # "controller": "joystick",
            "manual_control": True,
            "use_render": True,
            "use_saver": True,
            "map": 16
        }
    )

    o = env.reset()
    env.pg_world.force_fps.toggle()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        text = {"save": env.takeover_start}
        env.render(text=text)
        if info["arrive_dest"]:
            env.reset()
    env.close()
