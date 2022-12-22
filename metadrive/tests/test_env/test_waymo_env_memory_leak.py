import os
import time

from metadrive.envs.real_data_envs.waymo_env import WaymoEnv


# inner psutil function
def process_memory():
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def test_waymo_env_memory_leak():
    env = WaymoEnv(dict(case_num=2, sequential_seed=True, store_map=True, store_map_buffer_size=1))
    ct = time.time()
    cm = process_memory()
    last_mem = 0.0
    for t in range(50):
        lt = time.time()
        env.reset()
        nlt = time.time()
        lm = process_memory()
        print(
            "After {} Iters, Time {:.3f} Total Time {:.3f}, Memory Usage {:,}".format(
                t + 1, nlt - lt, nlt - ct, lm - cm
            )
        )
        if t > 20:
            assert abs((lm - cm) - last_mem) < 512 * 1024  # Memory should not have change > 512KB
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
            print(type(x), "\n  ", s)
            res.append([type(x), s, x])
        return res


    gc.enable()
    gc.set_debug(gc.DEBUG_LEAK)

    test_waymo_env_memory_leak()

    # show the dirt ;-)
    # ret = dump_garbage()

    # print(ret)