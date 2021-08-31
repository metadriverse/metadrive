from metadrive.envs.metadrive_env import MetaDriveEnv

if __name__ == "__main__":
    env = MetaDriveEnv(
        {
            "environment_num": 1,
            "traffic_density": 0.1,
            "start_seed": 4,
            "manual_control": True,
            "use_render": True,
            "offscreen_render": True,  # it is a switch telling metadrive to use rgb as observation
            "rgb_clip": True,  # clip rgb to range(0,1) instead of (0, 255)
            "pstats": True,
        }
    )
    env.reset()
    # print m to capture rgb observation
    env.engine.accept(
        "m", env.vehicle.image_sensors[env.vehicle.config["image_source"]].save_image, extraArgs=[env.vehicle]
    )

    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        assert env.observation_space.contains(o)
        # if env.config["use_render"]:
        # for i in range(ImageObservation.STACK_SIZE):
        #      ObservationType.show_gray_scale_array(o["image"][:, :, i])
        # image = env.render(mode="any str except human", text={"can you see me": i})
        # ObservationType.show_gray_scale_array(image)
        if d:
            print("Reset")
            env.reset()
    env.close()
