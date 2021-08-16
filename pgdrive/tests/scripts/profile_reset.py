import time

from pgdrive import PGDriveEnv

if __name__ == '__main__':
    env = PGDriveEnv(dict(environment_num=20000, load_map_from_json=True))
    obs = env.reset()
    start = time.time()
    for s in range(1000):
        env.reset(force_seed=s)
        if (s + 1) % 1 == 0:
            print(f"{s + 1} | Time Elapse: {time.time() - start}")
