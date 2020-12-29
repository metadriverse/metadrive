from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod
from pgdrive.utils import setup_logger
from pgdrive.scene_manager import TrafficMode
import json

setup_logger(True)


class TestEnv(PGDriveEnv):
    def __init__(self, save_episode=True):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "start_seed": 5,
                "manual_control": True,
                "use_render": True,
                "traffic_mode": TrafficMode.Reborn,
                "record_episode": save_episode,
                "map_config": {
                    Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                    Map.GENERATE_PARA: "XTXTXTXTXT",
                    Map.LANE_WIDTH: 3.5,
                    Map.LANE_NUM: 3,
                }
            }
        )


if __name__ == "__main__":
    test_dump = False

    env = TestEnv()
    o = env.reset()
    epi_info = None
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        env.render()
        if d:
            epi_info = env.scene_manager.dump_episode()

            # test dump json
            if test_dump:
                with open("test.json", "w") as f:
                    json.dump(epi_info, f)
            break
    env.close()

    del env
    env = TestEnv(False)
    o = env.reset(epi_info)
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        env.render()
        if d:
            break
    env.close()
