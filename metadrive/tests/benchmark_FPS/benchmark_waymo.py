import time
from metadrive.envs.scenario_env import ScenarioEnv, AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


def process_memory():
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # return mb
    return mem_info.rss / 1024 / 1024


def benchmark_fps():
    env = ScenarioEnv(
        {
            "use_render": False,
            "data_directory": AssetLoader.file_path("waymo", unix_style=False),
            "agent_policy": ReplayEgoCarPolicy,
            "no_traffic": False,
            "window_size": (1200, 800),
            "horizon": 1000,
            "vehicle_config": dict(
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50),
            ),
        }
    )
    env.reset()
    total_time = 0
    total_steps = 0
    for seed in range(300, 400):
        env.reset(seed=seed % env.config["num_scenarios"])
        start = time.time()
        for i in range(100):
            o, r, tm, tc, info = env.step([0, 0])
            total_steps += 1
            if tm or tc:
                break
        total_time += time.time() - start
        if (seed + 300) % 20 == 0:
            print("Seed: {}, FPS: {}".format(seed, total_steps / total_time))
    # print("FPS: {}".format(total_steps / total_time))


if __name__ == "__main__":
    # benchmark_reset_1000(load_city_map=False)
    # benchmark_reset_1000(load_city_map=False)
    # benchmark_reset_5_map_1000_times(True)
    benchmark_fps()
