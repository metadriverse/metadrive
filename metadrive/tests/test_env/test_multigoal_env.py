import pathlib

# import pytest

from metadrive.constants import get_color_palette
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import read_dataset_summary, read_scenario_data
from metadrive.envs.multigoal_intersection import MultiGoalIntersectionEnv
import numpy as np


# @pytest.mark.parametrize("data_directory", ["waymo", "nuscenes"])
def test_multigoal_env(render=False):
    # path = pathlib.Path(AssetLoader.file_path(AssetLoader.asset_path, data_directory, unix_style=False))
    # summary, scenario_ids, mapping = read_dataset_summary(path)
    env = MultiGoalIntersectionEnv(
        config=dict(
            use_render=render,
            use_multigoal_intersection=True,
            # agent_policy=ReplayEgoCarPolicy,
        )
    )
    try:
        env.reset()
        for ep in range(10):
            print("Current seed: ", env.engine.global_seed)
            while True:
                o, r, tm, tc, info = env.step([0.0, 1.0])
                assert env.observation_space.contains(o)

                # print(
                #     f"Current seed: {env.engine.global_seed}, "
                #     f"Current position: {env.vehicle.origin.get_pos()}, "
                #     f"Terrain position: {env.engine.terrain.height}, "
                #     f"Terrain pos: {env.engine.terrain.origin.get_pos()}, "
                #     f"plane_collision_terrain position: {env.engine.terrain.plane_collision_terrain.get_pos()}, "
                # )

                goal = str(info['current_goal'])

                assert np.all(o == info["obs/goals/{}".format(goal)])

                assert np.all(r == info["reward/goals/{}".format(goal)])
                assert np.all(r == info["reward/goals/{}".format("default")])

                rc = info["route_completion"]
                assert np.all(rc == info["route_completion/goals/{}".format(goal)])
                assert np.all(rc == info["route_completion/goals/{}".format("default")])

                if tm or tc:
                    # assert info["arrive_dest"], "Can not arrive dest"
                    # print("{} track_length: ".format(env.engine.global_seed), info["track_length"])
                    # assert info["arrive_dest"], "Can not arrive dest"
                    env.reset()
                    break

            # if i == 999:
            #     raise ValueError("Can not arrive dest")
        # assert env.agent.panda_color == get_color_palette()[2]
    finally:
        env.close()


if __name__ == "__main__":
    test_multigoal_env(render=False)
