#!/usr/bin/env python
"""
This script demonstrates how to setup the Safe RL environments.

Please feel free to run this script to enjoy a journey by keyboard! Remember to press H to see help message!

Auto-Drive mode may fail to solve some scenarios due to distribution mismatch.
"""
import logging

from metadrive.constants import HELP_MESSAGE
from metadrive.tests.test_functionality.test_object_collision_detection import ComplexEnv

if __name__ == "__main__":
    env = ComplexEnv(dict(use_render=True, manual_control=True, vehicle_config={"show_navi_mark": False}))
    try:
        env.reset()
        print(HELP_MESSAGE)
        env.agent.expert_takeover = True
        for i in range(1, 1000000000):
            previous_takeover = env.current_track_agent.expert_takeover
            o, r, tm, tc, info = env.step([0, 0])
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_agent.expert_takeover else "off",
                    "Total episode cost": env.episode_cost,
                    "Keyboard Control": "W,A,S,D",
                }
            )
            if not previous_takeover and env.current_track_agent.expert_takeover:
                logging.warning("Auto-Drive mode may fail to solve some scenarios due to distribution mismatch")
            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_agent.expert_takeover = True
    finally:
        env.close()
