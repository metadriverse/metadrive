from pgdrive import PGDriveEnv
from pgdrive.scene_creator.map.map import Map, MapGenerateMethod
from pgdrive.utils import draw_top_down_map

if __name__ == '__main__':
    env = PGDriveEnv(
        dict(
            environment_num=1,
            map_config={
                Map.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                Map.GENERATE_CONFIG: "OCrRCTXRCCCCrOr",
                Map.LANE_WIDTH: 3.5,
                Map.LANE_NUM: 3,
            }
        )
    )
    for i in range(100):
        env.reset()
        map = draw_top_down_map(env.current_map)
        print("Finish {} maps!".format(i + 1))
