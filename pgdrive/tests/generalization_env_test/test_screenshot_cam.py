from pgdrive.envs.pgdrive_env import PGDriveEnv


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.0,
                "start_seed": 3,
                "pg_world_config": {
                    "onscreen_message": True,
                    "screenshot_cam": True,
                },
                "image_source": "mini_map",
                "manual_control": True,
                "use_render": True,
            }
        )


if __name__ == "__main__":
    env = TestEnv()
    o = env.reset()
    env.pg_world.accept("1", env.pg_world.screenshot_cam.save_image)
    for i in range(1, 100000):
        env.step([0, 1])
        env.render("Test: {}".format(i))
    env.close()
