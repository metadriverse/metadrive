"""
Please feel free to run this script to enjoy a journey carrying out by a professional driver!
Our expert can drive in 10000 maps with almost 90% likelihood to achieve the destination.

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
from pgdrive import PGDriveEnv
from pgdrive.examples import expert

env = PGDriveEnv(dict(use_render=True, environment_num=10000))
obs = env.reset()
try:
    while True:
        action = expert(obs)
        obs, reward, done, info = env.step(action)
        frame = env.render("rgb_array")  # Return numpy array as well as showing the window.
        if done:
            obs = env.reset()
finally:
    print("Closing the environment!")
    env.close()
