from pg_drive.envs.generalization_racing import GeneralizationRacing
from pg_drive.scene_creator.algorithm.BIG import BigGenerateMethod
import logging
from pg_drive.scene_manager.traffic_manager import TrafficMode
from pg_drive.utils import setup_logger

# setup_logger(debug=True)


class ResetEnv(GeneralizationRacing):
    def __init__(self):
        super(ResetEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.0,
                "start_seed": 4,
                "bt_world_config": {
                    "debug": False,
                    "rgb_headless": True
                },
                "vehicle_config": {
                    "mini_map": (512, 512, 120),
                    "front_cam": (256, 256)
                },
                "image_buffer_name": "mini_map",
                "manual_control": True,
                "use_render": True,
                "use_rgb": True,
                "steering_penalty": 0.0,
                "decision_repeat": 5,
                "rgb_clip": True,
                "map_config": {
                    "type": BigGenerateMethod.BLOCK_SEQUENCE,
                    "config": "CCCCCC"
                }
            }
        )
        # self.reset()
        # self.bullet_world.accept("r", self.reset)
        # self.bullet_world.cam.setPos(0, 0, 1500)
        # self.bullet_world.cam.lookAt(0, 0, 0)


if __name__ == "__main__":
    env = ResetEnv()
    import time

    env.reset()
    env.bullet_world.accept("m", env.vehicle.mini_map.save_image)
    env.bullet_world.accept("c", env.vehicle.front_cam.save_image)
    for i in range(1, 100000):
        # start = time.time()
        # print("Step: ", i)
        o, r, d, info = env.step([0, 1])
        # print(r)
        # print(o)
        # print(time.time() - start)
        # print(len(o), "Vs.", env.observation_space.shape[0])
        # print(info)
        env.render(text={"can you see me": i})
        if d:
            print("Reset")
            env.reset()
    env.close()
