import logging
import os

import psutil
import tqdm

from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import initialize_engine, close_engine
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.scenario_data_manager import ScenarioDataManager
from metadrive.manager.scenario_map_manager import ScenarioMapManager

logging.basicConfig(level=logging.DEBUG)
from metadrive import MetaDriveEnv


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # mb


def test_pgdrive_env_memory_leak():
    total_num = 400
    num = 20
    out_loop_num = int(total_num / num)
    env = MetaDriveEnv(dict(
        store_map=False,
        num_scenarios=num,
        # traffic_density=0.
    ))
    try:
        for j in tqdm.tqdm(range(out_loop_num)):
            for i in range(num):
                obs, _ = env.reset(seed=i)
                if j == 0 and i == 0:
                    start_memory = process_memory()
        end_memory = process_memory()
        print("Start: {}, End: {}".format(start_memory, end_memory))
        assert abs(start_memory - end_memory) < 15
    finally:
        env.close()


def test_pg_map_destroy():
    default_config = MetaDriveEnv.default_config()

    total_num = 200
    num = 20
    out_loop_num = int(total_num / num)
    default_config["num_scenarios"] = num
    default_config["store_map"] = False
    default_config["map_config"] = {
        'type': 'block_num',
        'config': 3,
        'lane_width': 3.5,
        'lane_num': 3,
        'exit_length': 50,
        'seed': 0
    }
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
        assert abs(start_memory - end_memory) < 10
    finally:
        close_engine()


def test_scenario_map_destroy():
    total_num = 90
    num = 3
    out_loop_num = int(total_num / num)

    default_config = ScenarioEnv.default_config()
    default_config["data_directory"] = AssetLoader.file_path("waymo", unix_style=False)
    default_config["num_scenarios"] = num
    default_config["store_map"] = False

    no_map_memory = process_memory()
    engine = initialize_engine(default_config)
    engine.data_manager = ScenarioDataManager()
    engine.map_manager = ScenarioMapManager()
    try:
        for j in tqdm.tqdm(range(out_loop_num)):
            for i in range(num):
                engine.seed(i)
                engine.map_manager.before_reset()
                engine.map_manager.reset()
                if j == 0 and i == 0:
                    start_memory = process_memory()
        engine.current_map.destroy()
        end_memory = process_memory()
        print("Start: {}, End: {}, No Map: {}".format(start_memory, end_memory, no_map_memory))
        assert abs(start_memory - end_memory) < 20
    finally:
        close_engine()


if __name__ == '__main__':
    # test_scenario_map_destroy()
    # test_pg_map_destroy()
    test_pgdrive_env_memory_leak()
