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
    args = parser.parse_args()
    env_cls_name = args.env
    assert env_cls_name in envs.keys(), "No environment named {}, argument accepted: \n" \
                                        "(1) roundabout\n" \
                                        "(2) intersection\n" \
                                        "(3) tollgate\n" \
                                        "(4) bottleneck\n" \
                                        "(5) parkinglot\n" \
                                        "(6) pgma" \
        .format(env_cls_name)
    env = envs[env_cls_name]({"use_render": True, "manual_control": True, "crash_done": False})
    env.reset()
    env.switch_to_third_person_view()  # Default is in Top-down view, we switch to Third-person view.
    for i in range(1, 10000000000):
        o, r, d, info = env.step({agent_id: [0, 0] for agent_id in env.vehicles.keys()})
        env.render(
            text={
                "Number of existing vehicles": len(env.vehicles),
                "Tracked agent (Press Q)": env.engine.agent_manager.object_to_agent(env.current_track_vehicle.id)
            }
        )
        if d["__all__"]:
            env.reset()
    env.close()
