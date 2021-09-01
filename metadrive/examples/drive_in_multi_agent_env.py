"""
Manually drive in marl-env
Args:
    (1) roundabout
    (2) intersection
    (3) tollgate
    (4) bottleneck
    (5) parkinglot
    (6) pgmap

"""
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.envs.marl_envs.marl_tollgate import MultiAgentTollgateEnv
from metadrive.envs.marl_envs.marl_bottleneck import MultiAgentBottleneckEnv
from metadrive.envs.marl_envs.marl_intersection import MultiAgentIntersectionEnv
from metadrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv
from metadrive.envs.marl_envs.marl_parking_lot import MultiAgentParkingLotEnv
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="roundabout")
    envs = dict(roundabout=MultiAgentRoundaboutEnv,
                intersection=MultiAgentIntersectionEnv,
                tollgate=MultiAgentTollgateEnv,
                bottleneck=MultiAgentBottleneckEnv,
                parkinglot=MultiAgentParkingLotEnv,
                pgmap=MultiAgentMetaDrive)
    args = parser.parse_args()
    env_cls_name = args.env
    assert env_cls_name in envs.keys(), "No environment named {}, argument accepted: \n" \
                                        "(1) roundabout\n" \
                                        "(2) intersection\n" \
                                        "(3) tollgate\n" \
                                        "(4) bottleneck\n" \
                                        "(5) parkinglot\n" \
                                        "(6) pgmap" \
        .format(env_cls_name)
    env = envs[env_cls_name]({"use_render": True, "manual_control": True, "crash_done": False})
    env.reset()
    for i in range(1, 10000000000):
        o, r, d, info = env.step({agent_id: [0, 0] for agent_id in env.vehicles.keys()})
        env.render(text={"Number of existing vehicles": len(env.vehicles),
                         "Tracked agent (Press Q)": env.engine.agent_manager.object_to_agent(
                             env.current_track_vehicle.id)})
        if d["__all__"]:
            env.reset()
    env.close()
