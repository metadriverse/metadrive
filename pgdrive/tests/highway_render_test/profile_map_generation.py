import matplotlib.pyplot as plt

from pgdrive import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod

env = PGDriveEnv(
    dict(
        environment_num=1,
        map_config={
            Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            Map.GENERATE_PARA: "OCrRCTXRCCCCrOr",
            Map.LANE_WIDTH: 3.5,
            Map.LANE_NUM: 3,
        }
    )
)

for i in range(100):
    env.reset()
    map = env.get_map()
    print("Finish {} maps!".format(i))
    plt.figure()
    plt.imshow(map)
    plt.show()
