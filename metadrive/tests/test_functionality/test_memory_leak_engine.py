import gc
import tracemalloc
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


# A step may keep this many bytes alive on average before it counts as a leak
MAX_LEAK_PER_STEP = 64


def python_memory_growth(step, num_steps, num_warmup_steps):
    """
    Run step() num_warmup_steps + num_steps times and return how many bytes allocated by Python during the last
    num_steps calls are still alive afterwards.

    The process RSS is not used here: it changes in whole pages depending on allocator state and other threads, so it
    grows now and then in a long test session even when nothing leaks.
    """
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        for _ in range(num_warmup_steps):
            step()
        gc.collect()
        before = tracemalloc.get_traced_memory()[0]
        for _ in range(num_steps):
            step()
        gc.collect()
        return tracemalloc.get_traced_memory()[0] - before
    finally:
        if not was_tracing:
            tracemalloc.stop()


def test_engine_memory_leak():
    default_config = MetaDriveEnv.default_config()
    default_config["map_config"]["config"] = 3
    close_engine()
    initialize_engine(default_config)
    try:
        growth = python_memory_growth(lambda: get_engine().seed(0), num_steps=200, num_warmup_steps=100)
    finally:
        close_engine()
    assert growth < 200 * MAX_LEAK_PER_STEP, "engine.seed() keeps {:.0f} bytes alive per call".format(growth / 200)


def test_config_memory_leak():
    def step():
        default_config = MetaDriveEnv.default_config()
        default_config.update({"map": 3, "type": "block_sequence", "config": 3})

    growth = python_memory_growth(step, num_steps=300, num_warmup_steps=500)
    assert growth < 300 * MAX_LEAK_PER_STEP, "default_config() keeps {:.0f} bytes alive per call".format(growth / 300)


if __name__ == "__main__":
    test_engine_memory_leak()
    test_config_memory_leak()
