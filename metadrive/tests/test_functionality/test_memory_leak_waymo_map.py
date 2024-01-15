import logging
import time
from metadrive.engine.logger import set_log_level
from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import initialize_engine, close_engine
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.manager.scenario_data_manager import ScenarioDataManager
from metadrive.tests.test_functionality.test_memory_leak_engine import process_memory


def test_waymo_env_memory_leak(num_reset=100):
    env = ScenarioEnv(
        dict(
            num_scenarios=2,
            sequential_seed=True,
            store_map=True,
            data_directory=AssetLoader.file_path("waymo", unix_style=False),
        )
    )

    try:
        ct = time.time()
        cm = process_memory()
        for t in range(num_reset):
            lt = time.time()
            env.reset()
            nlt = time.time()
            lm = process_memory()
            # print(
            #     "After {} Iters, Time {:.3f} Total Time {:.3f}, Memory Usage {:,}".format(
            #         t + 1, nlt - lt, nlt - ct, lm - cm
            #     )
            # )
        assert lm - cm < 1024 * 1024 * 120, "We expect will cause ~120MB memory leak."

    finally:
        env.close()


def test_waymo_map_memory_leak():
    set_log_level(logging.DEBUG)
    default_config = ScenarioEnv.default_config()
    default_config["data_directory"] = AssetLoader.file_path("waymo", unix_style=False)
    default_config["num_scenarios"] = 1

    try:
        close_engine()
        engine = initialize_engine(default_config)

        ct = time.time()
        cm = process_memory()
        last_mem = 0.0

        lt = time.time()

        engine.data_manager = ScenarioDataManager()
        engine.seed(0)

        lm = process_memory()
        nlt = time.time()
        # print("After {} Iters, Time {:.3f} Total Time {:.3f}, Memory Usage {:,}".format(0, nlt - lt, nlt - ct, lm - cm))

        for t in range(10):
            lt = time.time()
            m_data = engine.data_manager.get_scenario(0, should_copy=False)["map_features"]
            map = ScenarioMap(map_index=0, map_data=m_data)
            map.attach_to_world(engine.render, engine.physics_world)
            map.destroy()

            # map.play()

            nlt = time.time()
            lm = process_memory()
            # print(
            #     "After {} Iters, Time {:.3f} Total Time {:.3f}, Memory Usage {:,}".format(
            #         t + 1, nlt - lt, nlt - ct, lm - cm
            #     )
            # )
            # if t > 5:
            #     assert abs((lm - cm) - last_mem) < 1024  # Memory should not have change > 1KB
            last_mem = lm - cm
        cost = (lm - cm) / 1024 / 1024
        print("Process takes {} MB".format(cost))
        assert cost < 40, "We expect will cause ~33MB memory leak."

    finally:
        close_engine()


if __name__ == "__main__":
    # https://code.activestate.com/recipes/65333/

    # import gc

    # def dump_garbage():
    #     """
    #     show us what's the garbage about
    #     """
    #
    #     # force collection
    #     # print("\nGARBAGE:")
    #     gc.collect()
    #
    #     # print("\nGARBAGE OBJECTS:")
    #     res = []
    #     for x in gc.garbage:
    #         s = str(x)
    #         if len(s) > 80:
    #             s = s[:80]
    #         # print(type(x), "\n  ", s)
    #         res.append([type(x), s, x])
    #     return res

    # gc.enable()
    # gc.set_debug(gc.DEBUG_LEAK)

    # test_waymo_env_memory_leak(num_reset=300)

    test_waymo_map_memory_leak()

    # show the dirt ;-)
    # ret = dump_garbage()

    # # print(ret)
