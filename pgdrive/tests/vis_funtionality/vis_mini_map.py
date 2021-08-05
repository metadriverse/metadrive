from pgdrive.envs.pgdrive_env import PGDriveEnv


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "start_seed": 4,
                "image_source": "mini_map",
                "manual_control": True,
                "use_render": True,
                "offscreen_render": True,
                "rgb_clip": True,
                "headless_machine_render": False
            }
        )


if __name__ == "__main__":
    env = TestEnv()
    env.reset()
    env.engine.accept("m", env.vehicle.image_sensors[env.config["image_source"]].save_image)

    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        assert env.observation_space.contains(o)
        if env.config["use_render"]:
            # from pgdrive.envs.observation_type import ObservationType, ImageObservation
            # for i in range(ImageObservation.STACK_SIZE):
            #     ObservationType.show_gray_scale_array(o["image"][:, :, i])
            env.render(text={"can you see me": i})
        if d:
            print("Reset")
            env.reset()
    env.close()
