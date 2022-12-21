import os
import time

from metadrive.envs.real_data_envs.waymo_env import WaymoEnv


# inner psutil function
def process_memory():
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def test_waymo_env_memory_leak():
    env = WaymoEnv(dict(random_seed=0, case_num=3, sequential_seed=True, store_map=True, store_map_buffer_size=1))
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
    test_waymo_env_memory_leak()
