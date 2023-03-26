from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.replay_policy import WaymoReplayEgoCarPolicy
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv


def test_export_metadrive_scenario():
    env = MetaDriveEnv(dict(start_seed=0, environment_num=10))
    policy = lambda x: [0, 1]
    try:
        scenarios = env.export_scenarios(policy, scenario_index=[i for i in range(10)])
    finally:
        env.close()


def test_export_waymo_scenario():
    env = WaymoEnv(dict(agent_policy=WaymoReplayEgoCarPolicy))
    policy = lambda x: [0, 1]
    try:
        scenarios = env.export_scenarios(policy, scenario_index=[i for i in range(3)])
    finally:
        env.close()
