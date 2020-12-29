from pgdrive.envs.pgdrive_env import PGDriveEnv


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "start_seed": 4,
                "image_source": "rgb_cam",
                "manual_control": True,
                "use_render": True,
                "use_image": True,
                "rgb_clip": True,
                # "vehicle_config": dict(rgb_cam=(200, 88)),
                "pg_world_config": {
                    "headless_image": False
                }
            }
        )


if __name__ == "__main__":
    env = TestEnv()
    env.reset()
    env.pg_world.accept("m", env.vehicle.image_sensors[env.config["image_source"]].save_image)

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
