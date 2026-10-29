import time

from metadrive.envs.top_down_env import TopDownSingleFrameMetaDriveEnv

if __name__ == '__main__':
    env = TopDownSingleFrameMetaDriveEnv(dict(num_scenarios=10))
    o, _ = env.reset()
    start = time.time()
    action = [0.0, 0.1]
    # print(o.shape)
    for s in range(10000):
        o, r, tm, tc, i = env.step(action)
        if tm or tc:
            env.reset()
        if (s + 1) % 100 == 0:
            print(
                "(TopDownEnv) Finish {}/10000 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}".format(
                    s + 1,
                    time.time() - start, (s + 1) / (time.time() - start)
                )
            )
    # print(f"(TopDownEnv) Total Time Elapse: {time.time() - start}")
