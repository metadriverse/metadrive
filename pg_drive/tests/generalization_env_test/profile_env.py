import time

import numpy as np

from pg_drive.envs.generalization_racing import GeneralizationRacing

if __name__ == '__main__':
    env = GeneralizationRacing(dict(render_mode="none"))
    obs = env.reset()
    start = time.time()
    for s in range(10000):
        action = np.array([0.0, 1.0])
        o, r, d, i = env.step(action)
        if d:
            env.reset()
        if (s + 1) % 100 == 0:
            print(f"{s + 1}/{1000} Time Elapse: {time.time() - start}")
    print(f"Total Time Elapse: {time.time() - start}")
