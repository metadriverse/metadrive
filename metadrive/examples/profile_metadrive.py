import time

import numpy as np

from metadrive import MetaDriveEnv
from metadrive.utils import setup_logger

if __name__ == '__main__':
    setup_logger(debug=False)
    env = MetaDriveEnv(dict(
        environment_num=1000,
        start_seed=1010,
    ))
    obs = env.reset()
    start = time.time()
    action = [0.0, 1.]
    vehicle_num = [len(env.engine.traffic_manager.vehicles)]
    for s in range(10000000):
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
    print(f"Total Time Elapse: {time.time() - start}")
