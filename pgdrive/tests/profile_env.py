import time

import numpy as np

from pgdrive import GeneralizationRacing

if __name__ == '__main__':
    env = GeneralizationRacing(dict(environment_num=10))
    obs = env.reset()
    start = time.time()
    total = 10000
    for s in range(total):
        if s < 30:
            action = np.array([0.0, 1.0])
        elif s < 200:
            action = np.array([0.0, -1.0])
        else:
            action = np.array([0.0, 1.0])
        o, r, d, i = env.step(action)
        if d:
            env.reset()
        if (s + 1) % 100 == 0:
            print(f"{s + 1}/{total} Time Elapse: {time.time() - start}")
    print(f"Total Time Elapse: {time.time() - start}")
