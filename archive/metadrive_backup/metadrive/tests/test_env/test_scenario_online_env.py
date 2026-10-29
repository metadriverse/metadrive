import pathlib

import pytest

from metadrive.constants import get_color_palette
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import read_dataset_summary, read_scenario_data


@pytest.mark.parametrize("data_directory", ["waymo", "nuscenes"])
def test_scenario_online_env(data_directory, render=False):
    path = pathlib.Path(AssetLoader.file_path(AssetLoader.asset_path, data_directory, unix_style=False))
    summary, scenario_ids, mapping = read_dataset_summary(path)
    try:
        env = ScenarioOnlineEnv(config=dict(
            use_render=render,
            agent_policy=ReplayEgoCarPolicy,
        ))
        for file_name, file_path in mapping.items():
            full_path = path / file_path / file_name
            assert full_path.exists(), f"{full_path} does not exist"
            scenario_description = read_scenario_data(full_path)
            print("Running scenario: ", scenario_description["id"])
            env.set_scenario(scenario_description)
            env.reset()
            for i in range(1000):
                o, r, tm, tc, info = env.step([1.0, 0.])
                assert env.observation_space.contains(o)
                if tm or tc:
                    assert info["arrive_dest"], "Can not arrive dest"
                    print("{} track_length: ".format(env.engine.global_seed), info["track_length"])
                    # assert info["arrive_dest"], "Can not arrive dest"
                    break

                if i == 999:
                    raise ValueError("Can not arrive dest")
            assert env.agent.panda_color == get_color_palette()[2]
    finally:
        env.close()


if __name__ == "__main__":
    test_scenario_online_env("nuscenes", render=True)
