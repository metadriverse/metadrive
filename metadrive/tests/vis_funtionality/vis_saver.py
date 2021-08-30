from metadrive.envs.metadrive_env import MetaDriveEnv

# setup_logger(True)

if __name__ == "__main__":
    env = MetaDriveEnv(
        {
            "environment_num": 5,
            "traffic_density": 0.1,
            "traffic_mode": "respawn",
            "start_seed": 5,
            # "controller": "joystick",
            "manual_control": True,
            "use_render": True,
            "use_AI_protector": True,
            "map": 30
        }
    )

    o = env.reset()
    # env.engine.force_fps.toggle()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        text = {"save": env.save_mode}
        env.render(text=text)
        # if d:
        #     print("Reset")
        #     env.reset()
    env.close()
