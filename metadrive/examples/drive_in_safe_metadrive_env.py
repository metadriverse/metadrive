"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.

Auto-Drive mode may fail to solve some scenarios due to distribution mismatch
"""
import logging

from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

if __name__ == "__main__":
    env = SafeMetaDriveEnv(dict(
        use_render=True,
        manual_control=True,
    ))
    env.reset()
    for i in range(1, 1000000000):
        previous_takeover = env.current_track_vehicle.expert_takeover
        o, r, d, info = env.step([0, 0])
        env.render(
            text={
                "Auto-Drive (Press T)": env.current_track_vehicle.expert_takeover,
                "Total episode cost": env.episode_cost
            }
        )
        if not previous_takeover and env.current_track_vehicle.expert_takeover:
            logging.warning("Auto-Drive mode may fail to solve some scenarios due to distribution mismatch")
        if d and info["arrive_dest"]:
            env.reset()
    env.close()
