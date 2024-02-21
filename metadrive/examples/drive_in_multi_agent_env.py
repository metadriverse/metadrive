#!/usr/bin/env python
"""
This script demonstrates how to setup the Multi-agent RL environments.


Usage: python -m metadrive.examples.drive_in_multi_agent_env --env pgma

Options for --env argument:
    (1) roundabout
    (2) intersection
    (3) tollgate
    (4) bottleneck
    (5) parkinglot
    (6) pgma

"""
import argparse
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)
from metadrive.constants import HELP_MESSAGE
from metadrive.policy.idm_policy import ManualControllableIDMPolicy

if __name__ == "__main__":
    envs = dict(
        roundabout=MultiAgentRoundaboutEnv,
        intersection=MultiAgentIntersectionEnv,
        tollgate=MultiAgentTollgateEnv,
        bottleneck=MultiAgentBottleneckEnv,
        parkinglot=MultiAgentParkingLotEnv,
        pgma=MultiAgentMetaDrive
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="roundabout", choices=list(envs.keys()))
    parser.add_argument("--top_down", "--topdown", action="store_true")
    args = parser.parse_args()
    env_cls_name = args.env
    extra_args = dict(film_size=(800, 800)) if args.top_down else {}
    assert env_cls_name in envs.keys(), "No environment named {}, argument accepted: \n" \
                                        "(1) roundabout\n" \
                                        "(2) intersection\n" \
                                        "(3) tollgate\n" \
                                        "(4) bottleneck\n" \
                                        "(5) parkinglot\n" \
                                        "(6) pgma" \
        .format(env_cls_name)
    env = envs[env_cls_name](
        {
            "use_render": True if not args.top_down else False,
            "crash_done": False,
            "sensors": dict(rgb_camera=(RGBCamera, 400, 300)),
            "interface_panel": ["rgb_camera", "dashboard"],
            "agent_policy": ManualControllableIDMPolicy
        }
    )
    try:
        env.reset()
        # if env.current_track_agent:
        #     env.current_track_agent.expert_takeover = True
        print(HELP_MESSAGE)
        env.switch_to_third_person_view()  # Default is in Top-down view, we switch to Third-person view.
        for i in range(1, 10000000000):
            o, r, tm, tc, info = env.step({agent_id: [0, 0] for agent_id in env.agents.keys()})
            env.render(
                **extra_args,
                mode="top_down" if args.top_down else None,
                text={
                    "Quit": "ESC",
                    "Number of existing vehicles": len(env.agents),
                    "Tracked agent (Press Q)": env.engine.agent_manager.object_to_agent(env.current_track_agent.id),
                    "Keyboard Control": "W,A,S,D",
                    # "Auto-Drive (Switch mode: T)": "on" if env.current_track_agent.expert_takeover else "off",
                } if not args.top_down else {}
            )
            if tm["__all__"]:
                env.reset()
                # if env.current_track_agent:
                #     env.current_track_agent.expert_takeover = True
    finally:
        env.close()
