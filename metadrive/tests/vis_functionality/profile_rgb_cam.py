import time

from metadrive.envs.metadrive_env import MetaDriveEnv

if __name__ == "__main__":
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.1,
            "start_seed": 4,
            "vehicle_config": {
                # "stack_size": 5,
                "rgb_camera": (64, 64)
            },
            "manual_control": True,
            "use_render": False,
            "image_observation": True,  # it is a switch telling metadrive to use rgb as observation
            "norm_pixel": True,  # clip rgb to range(0,1) instead of (0, 255)
            # "pstats": True,
        }
    )
    env.reset()
    # # print m to capture rgb observation
    env.engine.accept("m", env.agent.get_camera(env.agent.config["image_source"]).save_image, extraArgs=[env.agent])
    start = time.time()
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([0, 0])
        assert env.observation_space.contains(o)
        # if i % 1000 == 0:
        # print("FPS: {}".format(i / (time.time() - start)))
        if tm or tc:
            # print("Reset")
            env.reset()
    env.close()
