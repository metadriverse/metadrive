"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import random
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from pgdrive import PGDriveEnvV2

if __name__ == "__main__":
    env.reset()
    env = PGDriveEnvV2(
        dict(
            use_render=True,
            #use_saver=True,
            controller="joystick",
            manual_control=True,
            traffic_density=0.2,
            environment_num=100,
            map=7,
            start_seed=random.randint(0, 1000)
        )
    )
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 0])
        env.render()
        if d and info["arrive_dest"]:
            env.reset()
    env.close()
