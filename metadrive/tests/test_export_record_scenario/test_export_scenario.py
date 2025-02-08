import os
import pathlib
import shutil

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv, AssetLoader
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import save_dataset


def test_export_metadrive_scenario(render_export_env=False, render_load_env=False):
    num_scenarios = 3
    env = MetaDriveEnv(
        dict(start_seed=0, use_render=render_export_env, num_scenarios=num_scenarios, agent_policy=IDMPolicy)
    )
    policy = lambda x: [0, 1]
    dataset_dir = None
    try:
        scenarios, done_info = env.export_scenarios(policy, scenario_index=[i for i in range(num_scenarios)])

        dataset_dir = pathlib.Path(os.path.dirname(__file__)) / "../test_component/test_export"
        save_dataset(
            scenario_list=list(scenarios.values()),
            dataset_name="reconstructed_waymo",
            dataset_version="v0",
            dataset_dir=dataset_dir
        )
        env.close()

        env = ScenarioEnv(
            dict(
                agent_policy=ReplayEgoCarPolicy,
                data_directory=dataset_dir,
                use_render=render_load_env,
                num_scenarios=num_scenarios
            )
        )
        for index in range(num_scenarios):
            env.reset(seed=index)
            done = False
            while not done:
                o, r, tm, tc, i = env.step([0, 0])
                done = tm or tc
    finally:
        env.close()
        if dataset_dir is not None:
            shutil.rmtree(dataset_dir)


def test_export_waymo_scenario(num_scenarios=3, render_export_env=False, render_load_env=False):
    env = ScenarioEnv(
        dict(
            agent_policy=ReplayEgoCarPolicy,
            use_render=render_export_env,
            data_directory=AssetLoader.file_path("waymo", unix_style=False),
            start_scenario_index=0,
            num_scenarios=num_scenarios
        )
    )
    policy = lambda x: [0, 1]
    dataset_dir = None
    try:
        scenarios, done_info = env.export_scenarios(
            policy, scenario_index=[i for i in range(num_scenarios)], verbose=True
        )
        dataset_dir = pathlib.Path(os.path.dirname(__file__)) / "../test_component/test_export"
        save_dataset(
            scenario_list=list(scenarios.values()),
            dataset_name="reconstructed_waymo",
            dataset_version="v0",
            dataset_dir=dataset_dir
        )
        env.close()

        print("===== Start restoring =====")
        env = ScenarioEnv(
            dict(
                agent_policy=ReplayEgoCarPolicy,
                data_directory=dataset_dir,
                use_render=render_load_env,
                num_scenarios=num_scenarios
            )
        )
        for index in range(num_scenarios):
            print("Start replaying scenario {}".format(index))
            env.reset(seed=index)
            done = False
            count = 0
            while not done:
                o, r, tm, tc, i = env.step([0, 0])
                count += 1
                done = tm or tc
            print("Finish replaying scenario {} with step {}".format(index, count))
    finally:
        env.close()
        if dataset_dir is not None:
            shutil.rmtree(dataset_dir)


if __name__ == "__main__":
    # test_export_metadrive_scenario(render_export_env=False, render_load_env=False)
    test_export_waymo_scenario(num_scenarios=1, render_export_env=False, render_load_env=False)
