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
    parser.add_argument("--pygame_render", action="store_true")
    args = parser.parse_args()
    env_cls_name = args.env
    extra_args = dict(mode="top_down", film_size=(800, 800)) if args.pygame_render else {}
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
            "use_render": True if not args.pygame_render else False,
            "manual_control": True,
            "crash_done": False,
            "agent_policy": ManualControllableIDMPolicy
        }
    )
    try:
        env.reset()
        if env.current_track_vehicle:
            env.current_track_vehicle.expert_takeover = True
        print(HELP_MESSAGE)
        env.switch_to_third_person_view()  # Default is in Top-down view, we switch to Third-person view.
        for i in range(1, 10000000000):
            o, r, d, info = env.step({agent_id: [0, 0] for agent_id in env.vehicles.keys()})
            env.render(
                **extra_args,
                text={
                    "Number of existing vehicles": len(env.vehicles),
                    "Tracked agent (Press Q)": env.engine.agent_manager.object_to_agent(env.current_track_vehicle.id),
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                } if not args.pygame_render else {}
            )
            if d["__all__"]:
                env.reset()
                if env.current_track_vehicle:
                    env.current_track_vehicle.expert_takeover = True
    except:
        pass
    finally:
        env.close()
