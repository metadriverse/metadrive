from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger

setup_logger(debug=True)

if __name__ == "__main__":
    env = MetaDriveEnv(
        {
            "environment_num": 4,
            "traffic_density": 0.1,
            "start_seed": 3,
            "image_source": "mini_map",
            "manual_control": True,
            "use_render": True,
            "offscreen_render": False,
            "decision_repeat": 5,
            "rgb_clip": True,
            "map_config": {
                BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
                BaseMap.GENERATE_CONFIG: 12,
                BaseMap.LANE_WIDTH: 3.5,
                BaseMap.LANE_NUM: 3,
            }
        }
    )

    env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        env.render(text={"Frame": i, "Speed": env.vehicle.speed})
    env.close()
