#!/usr/bin/env python

import numpy as np

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import WayPointPolicy


def test_waypoint_policy():
    """
    Test the waypoint policy by running a scenario in the ScenarioEnv.
    """
    asset_path = AssetLoader.asset_path
    print(HELP_MESSAGE)
    cfg = {
        "agent_policy": WayPointPolicy,
        "map_region_size": 1024,  # use a large number if your map is toooooo big
        "sequential_seed": True,
        "reactive_traffic": False,
        "use_render": False,  # True if not args.top_down else False,
        "data_directory": AssetLoader.file_path(asset_path, "nuscenes", unix_style=False),
        "num_scenarios": 10,
    }
    env = ScenarioEnv(cfg)
    o, _ = env.reset()
    seen_seeds = set()
    seen_seeds.add(env.engine.data_manager.current_scenario["id"])
    # Load the information
    scenario = env.engine.data_manager.current_scenario
    ego_id = scenario["metadata"]["sdc_id"]
    ego_track = scenario["tracks"][ego_id]
    ego_traj = ego_track["state"]["position"][..., :2]
    ego_traj_ego = np.array([env.agent.convert_to_local_coordinates(point, env.agent.position) for point in ego_traj])
    ADEs, FDEs, flag = [], [], True
    for i in range(1, 100000):

        # FIXME: This part looks wierd as we don't set the action except for the first step.
        #  Then this won't test our waypoint policy in a closed-loop manner.
        if flag:
            action = dict(position=ego_traj_ego)
            flag = False
        else:
            action = None

        o, r, tm, tc, info = env.step(actions=action)
        if tm or tc:
            policy = env.engine.get_policy(env.agent.id)
            original_info, final_info = policy.traj_info, policy.online_traj_info
            original_waypoints = np.array([record["position"] for record in original_info])
            final_waypoints = np.array([record["position"] for record in final_info])
            ade = np.mean(np.linalg.norm(original_waypoints - final_waypoints, axis=-1))
            fde = np.linalg.norm(original_waypoints[-1] - final_waypoints[-1], axis=-1)
            ADEs.append(ade)
            FDEs.append(fde)
            print(f"ADE: {ade}, FDE: {fde}")
            env.reset()
            if env.engine.data_manager.current_scenario["id"] in seen_seeds:
                break
            else:
                seen_seeds.add(env.engine.data_manager.current_scenario["id"])
            trajectory, flag = [], True
            scenario = env.engine.data_manager.current_scenario
            ego_id = scenario["metadata"]["sdc_id"]
            ego_track = scenario["tracks"][ego_id]
            ego_traj = ego_track["state"]["position"][..., :2]
            ego_traj_ego = np.array(
                [env.agent.convert_to_local_coordinates(point, env.agent.position) for point in ego_traj]
            )
    print(f"Mean ADE: {np.mean(ADEs)}, Mean FDE: {np.mean(FDEs)}")

    mean_ade = np.mean(ADEs)
    mean_fde = np.mean(FDEs)
    assert mean_ade < 1 and mean_fde < 2

    env.close()


if __name__ == '__main__':
    test_waypoint_policy()
