from pg_drive.envs.generalization_racing import GeneralizationRacing
from pg_drive.scene_creator.map import Map, MapGenerateMethod


class TestEnv(GeneralizationRacing):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "start_seed": 4,
                "image_source": "depth_cam",
                "manual_control": True,
                "use_render": True,
                "use_image": True,
                "rgb_clip": True,
                "vehicle_config": dict(depth_cam=(200, 88)),
                "pg_world_config": {
                    "headless_rgb": False
                },
                "map_config": {
                    Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_NUM,
                    Map.GENERATE_PARA: 12,
                    Map.LANE_WIDTH: 3.5,
                    Map.LANE_NUM: 3,
                }
            }
        )


if __name__ == "__main__":
    env = TestEnv()
    env.reset()
    env.pg_world.accept("m", env.vehicle.image_sensors[env.config["image_source"]].save_image)
    from pg_drive.envs.observation_type import ObservationType, ImageObservation

    for i in range(1, 100000):
        # start = time.time()
        # print("Step: ", i)
        o, r, d, info = env.step([0, 1])
        assert env.observation_space.contains(o)
        # env.vehicle.rgb_cam.save_image()
        # # print(r)
        # # print(o)
        # # print(time.time() - start)
        # # print(len(o), "Vs.", env.observation_space.shape[0])
        # # print(info)
        if env.config["use_render"]:
            # for i in range(ImageObservation.STACK_SIZE):
            #     ObservationType.show_gray_scale_array(o["image"][:, :, i])
            env.render(text={"can you see me": i})
        # if d:
        #     print("Reset")
        #     env.reset()
    env.close()
