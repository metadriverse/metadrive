import time

import numpy as np

from metadrive import MetaDriveEnv
from metadrive.utils import setup_logger

if __name__ == '__main__':
    print("Start to profile the efficiency of MetaDrive with 1000 maps and ~8 vehicles!")
    setup_logger(debug=False)
    env = MetaDriveEnv(dict(
        environment_num=1000,
        start_seed=1010,
    ))
    obs = env.reset()
    start = time.time()
    action = [0.0, 1.]
    total_steps = 10000
    vehicle_num = [len(env.engine.traffic_manager.vehicles)]
    for s in range(total_steps):
        o, r, d, i = env.step(action)
        if d:
            env.reset()
            vehicle_num.append(len(env.engine.traffic_manager.vehicles))
        if (s + 1) % 100 == 0:
            print(
                "Finish {}/10000 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}, Average number of "
                "vehicles: {:.4f}".format(
                    s + 1,
                    time.time() - start, (s + 1) / (time.time() - start), np.mean(vehicle_num)
                )
            )
    print(
        "Total Time Elapse: {:.3f}, average FPS: {:.3f}, average number of vehicles: {:.3f}.".format(
            time.time() - start, total_steps / (time.time() - start), np.mean(vehicle_num)
        )
    )
