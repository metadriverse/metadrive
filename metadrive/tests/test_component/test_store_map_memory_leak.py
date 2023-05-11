import os
from metadrive.engine.engine_utils import initialize_engine
from metadrive.manager.pg_map_manager import PGMapManager
import tqdm
import psutil
import logging
logging.basicConfig(level=logging.DEBUG)
from metadrive.component.map.pg_map import PGMap
from metadrive import MetaDriveEnv


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 # mb


def test_pgdrive_env_memory_leak():
    total_num = 200
    num = 1
    out_loop_num = int(total_num/num)
    env = MetaDriveEnv(dict(store_map=False,
                            num_scenarios=num,
                            traffic_density=0.))
    start_memory = process_memory()
    try:
        for i in range(out_loop_num):
            for i in range(num):
                obs = env.reset(force_seed=i)
            end_memory = process_memory()
        print("Start: {}, End: {}".format(start_memory, end_memory))
    finally:
        env.close()


def test_map_destroy():
    default_config = MetaDriveEnv.default_config()
    default_config["num_scenarios"] = 1
    engine = initialize_engine(default_config)

    engine.data_manager = ScenarioDataManager()
    map = PG(map_index=0)
    map.attach_to_world()
    engine.enableMouse()
    map.road_network.show_bounding_box(engine)

    # argoverse data set is as the same coordinates as panda3d
    pos = map.get_center_point()
    engine.main_camera.set_bird_view_pos(pos)
    while True:
        map.engine.step()

if __name__ == '__main__':
    test_pgdrive_env_memory_leak()
