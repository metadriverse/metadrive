"""
Note: please install memory profiler with: pip install memory_profiler

Usage:

cd this repo
mprof run python vis_memory_leak.py
mprof plot *.dat
"""

from memory_profiler import profile

from pgdrive.envs.pgdrive_env import PGDriveEnv


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__({"manual_control": True, "traffic_density": 0.0, "use_render": False})

    @profile(precision=4, stream=open('memory_leak_test.log', 'w+'))
    def step(self, action):
        return super(TestEnv, self).step(action)


if __name__ == "__main__":
    env = TestEnv()
    env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        # env.render("Test: {}".format(i))
        if d:
            env.reset()
    env.close()
