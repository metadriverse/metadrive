import os.path as osp

from tqdm import tqdm

from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
import time
import os
import psutil


# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def test_waymo_env_memory_leak():

    import tracemalloc

    tracemalloc.start()

    env = WaymoEnv(dict(
        random_seed=0,
        case_num=5,
        sequential_seed=True,

        save_memory=True,
        save_memory_max_len=1,

        store_map=False

    ))

    ct = time.time()
    cm = process_memory()
    a = []
    for t in range(20):
        lt = time.time()
        env.reset()
        nlt = time.time()
        lm = process_memory()
        print("After {} Iters, Time {:.3f} Total Time {:.3f}, Memory Usage {:,}".format(t + 1, nlt - lt, nlt - ct,
                                                                                        lm - cm))
        if t > 10:
            print(1)

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)


if __name__ == "main":
    test_waymo_env_memory_leak()