import time

import numpy as np

from metadrive import MetaDriveEnv
from metadrive.utils import setup_logger


def _test_fps():
    print("Start to profile the efficiency of MetaDrive with 1000 maps and ~8 vehicles!")
    try:
        setup_logger(debug=False)
        env = MetaDriveEnv(dict(
            environment_num=1000,
            start_seed=1010,
        ))
        obs = env.reset()
        start = time.time()
        action = [0.0, 1.]
        total_steps = 5000
        vehicle_num = [len(env.engine.traffic_manager.vehicles)]
        for s in range(total_steps):
            o, r, d, i = env.step(action)
            if d:
                env.reset()
                vehicle_num.append(len(env.engine.traffic_manager.vehicles))
        assert total_steps / (time.time() - start) > 200
    finally:
        env.close()


if __name__ == '__main__':
    _test_fps()
