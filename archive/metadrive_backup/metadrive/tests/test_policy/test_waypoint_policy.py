#!/usr/bin/env python

import numpy as np

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioWaypointEnv


def test_waypoint_policy(render=False):
    """
    Test the waypoint policy by running a scenario in the ScenarioEnv.
    """
    asset_path = AssetLoader.asset_path
    env = ScenarioWaypointEnv(
        {
            "sequential_seed": True,
            "use_render": False,  # True if not args.top_down else False,
            "data_directory": AssetLoader.file_path(asset_path, "nuscenes", unix_style=False),
            "num_scenarios": 10,
        }
    )
    o, _ = env.reset()
    seen_seeds = set()
    seen_seeds.add(env.engine.data_manager.current_scenario["id"])
    # Load the information
    scenario = env.engine.data_manager.current_scenario
    ego_id = scenario["metadata"]["sdc_id"]
    ego_track = scenario["tracks"][ego_id]
    ego_traj = ego_track["state"]["position"][..., :2]
    # ego_traj_ego = np.array([env.agent.convert_to_local_coordinates(point, env.agent.position) for point in ego_traj])

    waypoint_horizon = env.engine.global_config["waypoint_horizon"]

    ADEs, FDEs, flag = [], [], True

    episode_step = 0
    replay_traj = []
    for _ in range(1, 100000):

        if episode_step % waypoint_horizon == 0:
            # prepare action:
            # X-coordinate is the forward direction of the vehicle, Y-coordinate is the left of the vehicle
            # Since we start with step 0, the first waypoint should be at step 1. Thus we start from step 1.
            local_traj = np.array(
                [
                    env.agent.convert_to_local_coordinates(point, env.agent.position)
                    for point in ego_traj[episode_step + 1:episode_step + waypoint_horizon + 1]
                ]
            )
            action = dict(position=local_traj)
        else:
            action = None
        replay_traj.append(env.agent.position)  # Store the pre-step position
        o, r, tm, tc, info = env.step(actions=action)
        episode_step += 1
        if render:
            env.render(mode="top_down")
        if tm or tc:
            if episode_step > 100:
                replay_traj = np.array(replay_traj)

                # Align their shape
                if replay_traj.shape[0] > ego_traj.shape[0]:
                    replay_traj = replay_traj[:ego_traj.shape[0]]
                elif replay_traj.shape[0] < ego_traj.shape[0]:
                    ego_traj = ego_traj[:replay_traj.shape[0]]

                ade = np.mean(np.linalg.norm(replay_traj - ego_traj, axis=-1))
                fde = np.linalg.norm(replay_traj[-1] - ego_traj[-1], axis=-1)
                ADEs.append(ade)
                FDEs.append(fde)
                print(
                    "For seed {}, horizon: {}, ADE: {}, FDE: {}".format(
                        env.engine.data_manager.current_scenario["id"], episode_step, ade, fde
                    )
                )
            else:
                # An early terminated episode. Skip.
                print(
                    "Early terminated episode {}, horizon: {}".format(
                        env.engine.data_manager.current_scenario["id"], episode_step
                    )
                )
                pass

            episode_step = 0
            env.reset()
            replay_traj = []
            if env.engine.data_manager.current_scenario["id"] in seen_seeds:
                break
            else:
                seen_seeds.add(env.engine.data_manager.current_scenario["id"])
            scenario = env.engine.data_manager.current_scenario
            ego_id = scenario["metadata"]["sdc_id"]
            ego_track = scenario["tracks"][ego_id]
            ego_traj = ego_track["state"]["position"][..., :2]

    print(f"Mean ADE: {np.mean(ADEs)}, Mean FDE: {np.mean(FDEs)}")

    mean_ade = np.mean(ADEs)
    mean_fde = np.mean(FDEs)
    assert mean_ade < 1e-4 and mean_fde < 1e-4

    env.close()


if __name__ == '__main__':
    test_waypoint_policy(render=True)
