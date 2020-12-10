import time

from pg_drive import GeneralizationRacing

if __name__ == '__main__':
    env = GeneralizationRacing(dict(environment_num=100, load_map_from_json=False))
    obs = env.reset()
    start = time.time()
    for s in range(1000):
        env.reset()
        if (s + 1) % 1 == 0:
            print(f"{s + 1} | Time Elapse: {time.time() - start}")
