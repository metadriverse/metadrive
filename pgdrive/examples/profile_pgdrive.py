import time

from pgdrive import PGDriveEnv
from pgdrive.utils import setup_logger

if __name__ == '__main__':
    setup_logger(debug=False)
    env = PGDriveEnv(dict(
        environment_num=1000,
        # use_render=True, fast=True,
        start_seed=1010,
        pstats=True
    ))
    obs = env.reset()
    start = time.time()
    action = [0.0, 1.]
    for s in range(10000000):
        o, r, d, i = env.step(action)
        if d:
            env.reset()
        if (s + 1) % 100 == 0:
            print(
                "Finish {}/10000 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}".format(
                    s + 1,
                    time.time() - start, (s + 1) / (time.time() - start)
                )
            )
    print(f"Total Time Elapse: {time.time() - start}")
