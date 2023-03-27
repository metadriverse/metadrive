from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.replay_policy import WaymoReplayEgoCarPolicy
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
import pickle
import os
import shutil


def test_export_metadrive_scenario(render_export_env=False, render_load_env=False):
    scenario_num = 3
    env = MetaDriveEnv(
        dict(start_seed=0, use_render=render_export_env, environment_num=scenario_num, agent_policy=IDMPolicy)
    )
    policy = lambda x: [0, 1]
    dir = None
    try:
        scenarios = env.export_scenarios(policy, scenario_index=[i for i in range(scenario_num)])
        dir = os.path.join(os.path.dirname(__file__), "test_export")
        os.makedirs(dir, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
        env.close()

        env = WaymoEnv(
            dict(
                agent_policy=WaymoReplayEgoCarPolicy,
                waymo_data_directory=dir,
                use_render=render_load_env,
                case_num=scenario_num
            )
        )
        for index in range(scenario_num):
            env.reset(force_seed=index)
            done = False
            while not done:
                o, r, done, i = env.step([0, 0])
    finally:
        env.close()
        if dir is not None:
            shutil.rmtree(dir)


def test_export_waymo_scenario(render_export_env=False, render_load_env=False):
    scenario_num = 3
    env = WaymoEnv(
        dict(
            agent_policy=WaymoReplayEgoCarPolicy,
            use_render=render_export_env,
            start_case_index=0,
            case_num=scenario_num
        )
    )
    policy = lambda x: [0, 1]
    dir = None
    try:
        scenarios = env.export_scenarios(policy, scenario_index=[i for i in range(scenario_num)])
        dir = os.path.join(os.path.dirname(__file__), "test_export")
        os.makedirs(dir, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
        env.close()

        env = WaymoEnv(
            dict(
                agent_policy=WaymoReplayEgoCarPolicy,
                waymo_data_directory=dir,
                use_render=render_load_env,
                case_num=scenario_num
            )
        )
        for index in range(scenario_num):
            env.reset(force_seed=index)
            done = False
            while not done:
                o, r, done, i = env.step([0, 0])
    finally:
        env.close()
        if dir is not None:
            shutil.rmtree(dir)


if __name__ == "__main__":
    # test_export_metadrive_scenario(render_export_env=True, render_load_env=True)
    test_export_waymo_scenario(render_export_env=True, render_load_env=True)
