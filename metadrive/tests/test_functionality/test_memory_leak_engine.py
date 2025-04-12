import time
from collections import deque
from itertools import chain
from sys import getsizeof, stderr

from metadrive.engine.engine_utils import initialize_engine, get_engine, close_engine
from metadrive.envs import MetaDriveEnv

try:
    from reprlib import repr
except ImportError:
    pass


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory foot# print an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):

        if hasattr(o, "__dict__"):
            o = o.__dict__

        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        # if verbose:
        # print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


# inner psutil function
def process_memory(to_mb=False):
    """
    Return the memory usage of current process. The unit is byte by default.
    """
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    if to_mb:
        return mem_info.rss / (1024**2)
    else:
        return mem_info.rss


def test_engine_memory_leak():
    try:

        default_config = MetaDriveEnv.default_config()
        default_config["map_config"]["config"] = 3

        close_engine()

        engine = initialize_engine(default_config)

        ct = time.time()
        last_lm = cm = process_memory()
        last_mem = 0.0
        for t in range(300):
            lt = time.time()

            engine.seed(0)

            engine = get_engine()

            nlt = time.time()
            lm = process_memory()
            # # print(
            #     "After {} Iters, Time {:.3f} Total Time {:.3f}, Memory Usage {:,} Memory Change {:,}".format(
            #         t + 1, nlt - lt, nlt - ct, lm - cm, lm - last_lm
            #     )
            # )
            last_lm = lm
            if t > 100:
                time.sleep(0.1)
                assert abs((lm - cm) - last_mem) < 10  # Memory should not have change > 1KB
            last_mem = lm - cm
    finally:
        close_engine()


def test_config_memory_leak():

    ct = time.time()
    last_lm = cm = process_memory()
    last_mem = 0.0
    for t in range(800):
        lt = time.time()

        default_config = MetaDriveEnv.default_config()
        default_config.update({"map": 3, "type": "block_sequence", "config": 3})
        del default_config

        nlt = time.time()
        lm = process_memory()
        # print(
        #     "After {} Iters, Time {:.3f} Total Time {:.3f}, Memory Usage {:,} Memory Change {:,}".format(
        #         t + 1, nlt - lt, nlt - ct, lm - cm, lm - last_lm
        #     )
        # )
        last_lm = lm
        if t > 500:
            time.sleep(0.1)
            assert abs((lm - cm) - last_mem) < 10  # Memory should not have change > 1KB
        last_mem = lm - cm


if __name__ == "__main__":
    test_engine_memory_leak()
    test_config_memory_leak()
