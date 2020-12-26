"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import random

from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_manager.traffic_manager import TrafficMode

if __name__ == "__main__":
    env = PGDriveEnv(
        dict(
            use_render=True,
            manual_control=True,
            traffic_density=0.2,
            traffic_mode=TrafficMode.Reborn,
            environment_num=100,
            map=7,
            start_seed=random.randint(0, 1000)
        )
    )
    env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 0])
        env.render()
        if d:
            env.reset()
    env.close()
