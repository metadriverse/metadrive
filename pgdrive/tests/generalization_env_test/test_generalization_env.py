from pgdrive.envs.pgdrive_env import PGDriveEnv


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "start_seed": 3,
                "pg_world_config": {
                    "onscreen_message": True,
                },
                "image_source": "mini_map",
                "manual_control": True,
                "use_render": True,
                "use_image": False,
                "steering_penalty": 0.0,
                "decision_repeat": 5,
                "rgb_clip": True,
                # "map_config": {
                #     Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_NUM,
                #     Map.GENERATE_PARA: 12,
                #     Map.LANE_WIDTH: 3.5,
                #     Map.LANE_NUM: 3,
                # }
            }
        )


if __name__ == "__main__":
    env = TestEnv()

    o = env.reset()
    for i in range(1, 100000):
        # start = time.time()
        # print("Step: ", i)
        env.step([0, 1])
        # print(r)
        # print(o)
        # print(time.time() - start)
        # print(len(o), "Vs.", env.observation_space.shape[0])
        # print(info)w
        env.render("Test: {}".format(i))
        # if d:
        #     print("Reset")
        #     env.reset()
    env.close()
