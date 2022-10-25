import argparse
import time

import numpy as np

from metadrive import MetaDriveEnv
from metadrive.utils import setup_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", "-n", default=10_000, type=int, help="Total steps of profiling.")
    args = parser.parse_args()

    print("Start to profile the efficiency of MetaDrive with 1000 maps and ~8 vehicles!")
    setup_logger(debug=False)
    env = MetaDriveEnv(dict(
        environment_num=1000,
        start_seed=1010,
    ))
    obs = env.reset()
    start = time.time()
    action = [0.0, 1.]
    total_steps = args.num_steps
    vehicle_num = [len(env.engine.traffic_manager.vehicles)]
    for s in range(total_steps):
        o, r, d, i = env.step(action)
        if d:
            env.reset()
            vehicle_num.append(len(env.engine.traffic_manager.vehicles))
        if (s + 1) % 100 == 0:
            print(
                "Finish {}/{} simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}, Average number of "
                "vehicles: {:.4f}".format(
                    s + 1, total_steps,
                    time.time() - start, (s + 1) / (time.time() - start), np.mean(vehicle_num)
                )
            )
    print(
        "Total Time Elapse: {:.3f}, average FPS: {:.3f}, average number of vehicles: {:.3f}.".format(
            time.time() - start, total_steps / (time.time() - start), np.mean(vehicle_num)
        )
    )
