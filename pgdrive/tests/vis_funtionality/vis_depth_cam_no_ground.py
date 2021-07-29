from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.component.map.base_map import BaseMap, MapGenerateMethod


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "start_seed": 4,
                "manual_control": True,
                "use_render": True,
                "use_image": True,
                "rgb_clip": True,
                "vehicle_config": dict(depth_cam=(200, 88, False), image_source="depth_cam"),
                "headless_image": False,
                "map_config": {
                    BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
                    BaseMap.GENERATE_CONFIG: 12,
                    BaseMap.LANE_WIDTH: 3.5,
                    BaseMap.LANE_NUM: 3,
                }
            }
        )


if __name__ == "__main__":
    env = TestEnv()
    env.reset()
    env.engine.accept("m", env.vehicle.image_sensors[env.vehicle.config["image_source"]].save_image)

    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        assert env.observation_space.contains(o)
        if env.config["use_render"]:
            # for i in range(ImageObservation.STACK_SIZE):
            #     ObservationType.show_gray_scale_array(o["image"][:, :, i])
            env.render(text={"can you see me": i})
        if d:
            print("Reset")
            env.reset()
    env.close()
