#!/usr/bin/env python
import argparse
import logging
import time

import numpy as np

from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv
from metadrive.utils import setup_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", "-n", default=10_000, type=int, help="Total steps of profiling.")
    args = parser.parse_args()

    print("Start to profile the efficiency of MetaDrive Multi-agent Roundabout environment!")
    setup_logger(debug=False)
    env = MultiAgentRoundaboutEnv(dict(start_seed=1010))
    obs, _ = env.reset()
    start = time.time()
    reset_used_time = 0
    action = [0.0, 1.]
    total_steps = args.num_steps
    vehicle_num = [len(env.agents)]
    for s in range(total_steps):
        o, r, tm, tc, i = env.step({k: action for k in env.agents})
        if tm["__all__"]:
            start_reset = time.time()
            env.reset()
            vehicle_num.append(len(env.agents))
            reset_used_time += time.time() - start_reset
        if (s + 1) % 100 == 0:
            print(
                "Finish {}/{} simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}, Average number of "
                "vehicles: {:.4f}".format(
                    s + 1, total_steps,
                    time.time() - start, (s + 1) / (time.time() - start - reset_used_time), np.mean(vehicle_num)
                )
            )
    print(
        "Total Time Elapse: {:.3f}, average FPS: {:.3f}, average number of vehicles: {:.3f}.".format(
            time.time() - start, total_steps / (time.time() - start - reset_used_time), np.mean(vehicle_num)
        )
    )
