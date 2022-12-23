import time

from metadrive.component.map.waymo_map import WaymoMap
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import initialize_engine, close_engine
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.manager.waymo_data_manager import WaymoDataManager
from metadrive.tests.test_functionality.test_memory_leak_engine import process_memory


def test_waymo_env_memory_leak():
    env = WaymoEnv(dict(case_num=2, sequential_seed=True, store_map=True, store_map_buffer_size=1))

    try:
        ct = time.time()
        cm = process_memory()
        for t in range(20):
            lt = time.time()
            env.reset()
            nlt = time.time()
            lm = process_memory()
            print(
                "After {} Iters, Time {:.3f} Total Time {:.3f}, Memory Usage {:,}".format(
                    t + 1, nlt - lt, nlt - ct, lm - cm
                )
            )
        assert lm - cm < 1024 * 1024 * 80, "We expect will cause ~70MB memory leak."

    finally:
        env.close()


def test_waymo_map_memory_leak():
    default_config = WaymoEnv.default_config()
    default_config["waymo_data_directory"] = AssetLoader.file_path("waymo", return_raw_style=False)
    default_config["case_num"] = 1

    try:
        close_engine()
        engine = initialize_engine(default_config)

        ct = time.time()
        cm = process_memory()
        last_mem = 0.0

        lt = time.time()

        engine.data_manager = WaymoDataManager()

        lm = process_memory()
        nlt = time.time()
        print("After {} Iters, Time {:.3f} Total Time {:.3f}, Memory Usage {:,}".format(0, nlt - lt, nlt - ct, lm - cm))

        for t in range(10):
            lt = time.time()

            map = WaymoMap(map_index=0)
            del map

            # map.play()

            nlt = time.time()
            lm = process_memory()
            print(
                "After {} Iters, Time {:.3f} Total Time {:.3f}, Memory Usage {:,}".format(
                    t + 1, nlt - lt, nlt - ct, lm - cm
                )
            )
            # if t > 5:
            #     assert abs((lm - cm) - last_mem) < 1024  # Memory should not have change > 1KB
            last_mem = lm - cm

        assert lm - cm < 1024 * 1024 * 40, "We expect will cause ~33MB memory leak."

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
    #     print("\nGARBAGE:")
    #     gc.collect()
    #
    #     print("\nGARBAGE OBJECTS:")
    #     res = []
    #     for x in gc.garbage:
    #         s = str(x)
    #         if len(s) > 80:
    #             s = s[:80]
    #         print(type(x), "\n  ", s)
    #         res.append([type(x), s, x])
    #     return res

    # gc.enable()
    # gc.set_debug(gc.DEBUG_LEAK)

    test_waymo_env_memory_leak()

    test_waymo_map_memory_leak()

    # show the dirt ;-)
    # ret = dump_garbage()

    # print(ret)
