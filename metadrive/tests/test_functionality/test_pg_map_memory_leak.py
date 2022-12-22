import time

from metadrive.component.map.pg_map import PGMap
from metadrive.engine.engine_utils import initialize_engine
from metadrive.envs import MetaDriveEnv


# inner psutil function
def process_memory():
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def test_pg_map_memory_leak():
    default_config = MetaDriveEnv.default_config()
    default_config["map_config"]["config"] = 3
    engine = initialize_engine(default_config)

    ct = time.time()
    cm = process_memory()
    last_mem = 0.0
    for t in range(20):
        lt = time.time()

        map = PGMap(default_config["map_config"])
        del map

        # map = {"aaa": 222}
        # del map

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


if __name__ == "__main__":

    # https://code.activestate.com/recipes/65333/

    import gc


    def dump_garbage():
        """
        show us what's the garbage about
        """

        # force collection
        print("\nGARBAGE:")
        gc.collect()

        print("\nGARBAGE OBJECTS:")
        res = []
        for x in gc.garbage:
            s = str(x)
            if len(s) > 80:
                s = s[:80]
            # print(type(x), "\n  ", s)
            res.append([type(x), s, x])
        return res


    # gc.enable()
    # gc.set_debug(gc.DEBUG_LEAK)


    test_pg_map_memory_leak()

    # show the dirt ;-)
    # ret = dump_garbage()
    #
    # print(ret)
