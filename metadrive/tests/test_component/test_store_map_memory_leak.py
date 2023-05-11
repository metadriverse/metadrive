import logging
import os

import psutil

from metadrive.engine.engine_utils import initialize_engine, close_engine
from metadrive.manager.pg_map_manager import PGMapManager

logging.basicConfig(level=logging.DEBUG)
from metadrive import MetaDriveEnv


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # mb


def test_pgdrive_env_memory_leak():
    total_num = 200
    num = 1
    out_loop_num = int(total_num / num)
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

    total_num = 200
    num = 1
    out_loop_num = int(total_num / num)
    default_config["num_scenarios"] = num
    default_config["store_map"] = False
    default_config["map_config"] = {'type': 'block_num', 'config': 3, 'lane_width': 3.5, 'lane_num': 3,
                                    'exit_length': 50, 'seed': 0}
    no_map_memory = process_memory()
    engine = initialize_engine(default_config)
    engine.map_manager = PGMapManager()
    try:
        for j in range(out_loop_num):
            for i in range(num):
                engine.seed(i)
                engine.map_manager.before_reset()
                engine.map_manager.reset()
                if j == 0 and i == 0:
                    start_memory = process_memory()
        engine.current_map.destroy()
        end_memory = process_memory()
        print("Start: {}, End: {}, No Map: {}".format(start_memory, end_memory, no_map_memory))
    finally:
        close_engine()

    # map = PGMap()
    # map.attach_to_world()


if __name__ == '__main__':
    test_map_destroy()
    # test_pgdrive_env_memory_leak()
