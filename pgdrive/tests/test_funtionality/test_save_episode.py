import json

from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod
from pgdrive.scene_manager.traffic_manager import TrafficMode
from pgdrive.utils import setup_logger


class TestEnv(PGDriveEnv):
    def __init__(self, save_episode=True, vis=True):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "start_seed": 5,
                "manual_control": vis,
                "use_render": vis,
                "traffic_mode": TrafficMode.Hybrid,
                "record_episode": save_episode,
                "map_config": {
                    Map.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                    Map.GENERATE_CONFIG: "XTXTXTXTXT",
                    Map.LANE_WIDTH: 3.5,
                    Map.LANE_NUM: 3,
                }
            }
        )


def test_save_episode(vis=False):
    setup_logger(True)

    test_dump = False

    env = TestEnv(vis=vis)
    try:
        o = env.reset()
        epi_info = None
        for i in range(1, 100000 if vis else 2000):
            o, r, d, info = env.step([0, 1])
            if vis:
                env.render()
            if d:
                epi_info = env.scene_manager.dump_episode()

                # test dump json
                if test_dump:
                    with open("test.json", "w") as f:
                        json.dump(epi_info, f)
                break

        o = env.reset(epi_info)
        for i in range(1, 100000 if vis else 2000):
            o, r, d, info = env.step([0, 1])
            if vis:
                env.render()
            if d:
                break
    finally:
        env.close()


if __name__ == "__main__":
    test_save_episode(vis=False)
