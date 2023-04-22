from metadrive.envs.real_data_envs.nuplan_env import NuPlanEnv
import tqdm
import time
from metadrive.policy.replay_policy import NuPlanReplayEgoCarPolicy


def process_memory():
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # return mb
    return mem_info.rss / 1024 / 1024


def benchmark_fps():
    env = NuPlanEnv(
        {
            "use_render": False,
            "agent_policy": NuPlanReplayEgoCarPolicy,
            "no_traffic": False,
            "no_pedestrian": True,
            "load_city_map": True,
            "window_size": (1200, 800),
            "start_scenario_index": 300,
            "scenario_radius": 200,
            "pstats": True,
            "num_scenarios": 2000,
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
        env.reset(force_seed=seed)
        start = time.time()
        for i in range(env.engine.data_manager.current_scenario_length * 10):
            o, r, d, info = env.step([0, 0])
            total_steps += 1
            if d:
                break
        total_time += time.time() - start
        if (seed + 300) % 20 == 0:
            print("Seed: {}, FPS: {}".format(seed, total_steps / total_time))
    # print("FPS: {}".format(total_steps / total_time))


def benchmark_reset_5_map_1000_times(load_city_map=True):
    env = NuPlanEnv(
        {
            "use_render": False,
            "agent_policy": NuPlanReplayEgoCarPolicy,
            "no_traffic": True,
            "no_pedestrian": True,
            "load_city_map": load_city_map,
            "start_scenario_index": 300,
            "num_scenarios": 5,
            "show_coordinates": False,
            "horizon": 1000,
            "vehicle_config": dict(
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50),
            ),
        }
    )
    start_time = time.time()
    # print("Before first reset process Memory: {}".format(process_memory()))
    env.reset()
    load_time = time.time() - start_time
    # print("After first reset process Memory: {}".format(process_memory()))
    for seed in tqdm.tqdm(range(300, 1300)):
        env.reset(force_seed=(seed % 5) + 300)
        # if seed % 500 == 0:
        # print("reset: {}, Time: {}, Process Memory: {}".format(seed, time.time() - start_time, process_memory()))
    # print(
    #     "Total Time: {}, Load time: {}, Total process Memory: {}".format(
    #         time.time() - start_time, load_time, process_memory()
    #     )
    # )


def benchmark_reset_1000(load_city_map=True):
    env = NuPlanEnv(
        {
            "use_render": False,
            "agent_policy": NuPlanReplayEgoCarPolicy,
            "no_traffic": True,
            "no_pedestrian": True,
            "load_city_map": load_city_map,
            "start_scenario_index": 300,
            "num_scenarios": 1000,
            "show_coordinates": False,
            "horizon": 1000,
            "vehicle_config": dict(
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50),
            ),
        }
    )
    start_time = time.time()
    # print("Before first reset process Memory: {}".format(process_memory()))
    env.reset()
    load_time = time.time() - start_time
    # print("After first reset process Memory: {}".format(process_memory()))
    for seed in tqdm.tqdm(range(300, 1300)):
        thisscenario = time.time()
        env.reset(force_seed=seed)
        # print("Seed: {}, Time: {}".format(seed, time.time() - thisscenario))
    #     if seed % 500 == 0:
    #         print("reset: {}, Time: {}, Process Memory: {}".format(seed, time.time() - start_time, process_memory()))
    # print(
    #     "Total Time: {}, Load time: {}, Total process Memory: {}".format(
    #         time.time() - start_time, load_time, process_memory()
    #     )
    # )


if __name__ == "__main__":
    benchmark_reset_1000(load_city_map=False)
    # benchmark_reset_1000(load_city_map=False)
    # benchmark_reset_5_map_1000_times(True)
    # benchmark_fps()
