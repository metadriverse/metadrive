from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod

if __name__ == "__main__":
    env = PGDriveEnv(
        {
            "environment_num": 10000,
            "traffic_density": 0.1,
            # "traffic_mode": 0,  # 0 for Reborn mode.
            "map_config": {
                Map.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
                Map.GENERATE_CONFIG: 7,
            }
        }
    )
    env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        env.reset()
        print("Current map {}, vehicle number {}.".format(env.current_seed, env.get_vehicle_num()))
    env.close()
