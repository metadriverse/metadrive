from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod
from pgdrive.utils import setup_logger

setup_logger(debug=True)


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "start_seed": 3,
                "pg_world_config": {
                    "debug": False,
                },
                "image_source": "mini_map",
                "manual_control": False,
                "use_render": False,
                "use_image": False,
                "steering_penalty": 0.0,
                "decision_repeat": 5,
                "rgb_clip": True,
                "map_config": {
                    Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                    Map.GENERATE_PARA: "OCrRCTXRCCCCrOr",
                    Map.LANE_WIDTH: 3.5,
                    Map.LANE_NUM: 3,
                }
            }
        )


if __name__ == "__main__":
    env = TestEnv()
    env.reset()
    env.current_map.save_map_image(simple_draw=False)
    # print(env.current_map.get_map_image_array())

    import numpy as np
    import cv2

    # env = TestEnv()
    # env.reset()
    # env.current_map.save_map_image(False)
    # print(env.current_map.get_map_image_array())

    # def remove_noise(gray, num):
    #     Y, X = gray.shape
    #     nearest_neigbours = [[
    #         np.argmax(
    #             np.bincount(
    #                 gray[max(i - num, 0):min(i + num, Y), max(j - num, 0):min(j + num, X)].ravel()))
    #         for j in range(X)] for i in range(Y)]
    #     result = np.array(nearest_neigbours, dtype=np.uint8)
    #     cv2.imwrite('result2.jpg', result)
    #     return result
    #
    #
    # img = cv2.imread('map_3.png')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # remove_noise(gray, 10)
