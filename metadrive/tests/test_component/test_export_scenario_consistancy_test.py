import os
import pickle
import shutil

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.replay_policy import WaymoReplayEgoCarPolicy
import numpy as np
from metadrive.scenario import ScenarioDescription as SD


def test_export_metadrive_scenario(render_export_env=False, render_load_env=False):

    # ===== Save data =====
    scenario_num = 3
    env = MetaDriveEnv(
        dict(start_seed=0, use_render=render_export_env, environment_num=scenario_num, agent_policy=IDMPolicy)
    )
    policy = lambda x: [0, 1]
    dir1 = None
    try:
        scenarios = env.export_scenarios(policy, scenario_index=[i for i in range(scenario_num)])
        dir1 = os.path.join(os.path.dirname(__file__), "test_export")
        os.makedirs(dir1, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir1, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
    finally:
        env.close()

    # ===== Save data of the restoring environment =====
    env = WaymoEnv(
        dict(
            agent_policy=WaymoReplayEgoCarPolicy,
            waymo_data_directory=dir1,
            use_render=render_load_env,
            case_num=scenario_num
        )
    )
    try:
        scenarios_restored = env.export_scenarios(policy, scenario_index=[i for i in range(scenario_num)])
    finally:
        env.close()

    if dir1 is not None:
        shutil.rmtree(dir1)

    # ===== These two set of data should align =====
    assert set(scenarios.keys()) == set(scenarios_restored.keys())
    for k in scenarios.keys():
        old_scene = scenarios[k]
        new_scene = scenarios_restored[k]
        SD.sanity_check(old_scene)
        SD.sanity_check(new_scene)
        assert old_scene[SD.LENGTH] == new_scene[SD.LENGTH]
        assert set(old_scene[SD.TRACKS].keys()) == set(new_scene[SD.TRACKS].keys())
        assert set(old_scene[SD.MAP_FEATURES].keys()) == set(new_scene[SD.MAP_FEATURES].keys())
        assert set(old_scene[SD.DYNAMIC_MAP_STATES].keys()) == set(new_scene[SD.DYNAMIC_MAP_STATES].keys())

        for track_id, track in old_scene[SD.TRACKS].items():
            assert np.all(new_scene[SD.TRACKS][track_id][SD.STATE] == track[SD.STATE])
            assert new_scene[SD.TRACKS][track_id][SD.TYPE] == track[SD.TYPE]

        for map_id, map_feat in old_scene[SD.MAP_FEATURES].items():
            assert np.all(new_scene[SD.MAP_FEATURES][map_id]["polyline"] == map_feat["polyline"])
            assert new_scene[SD.MAP_FEATURES][map_id][SD.TYPE] == map_feat[SD.TYPE]

        for obj_id, obj_state in old_scene[SD.DYNAMIC_MAP_STATES].items():
            assert np.all(new_scene[SD.DYNAMIC_MAP_STATES][obj_id][SD.STATE] == obj_state[SD.STATE])
            assert new_scene[SD.DYNAMIC_MAP_STATES][obj_id][SD.TYPE] == obj_state[SD.TYPE]


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
        scenarios = env.export_scenarios(policy, scenario_index=[i for i in range(scenario_num)], verbose=True)
        dir = os.path.join(os.path.dirname(__file__), "test_export")
        os.makedirs(dir, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
        env.close()

        print("===== Start restoring =====")
        env = WaymoEnv(
            dict(
                agent_policy=WaymoReplayEgoCarPolicy,
                waymo_data_directory=dir,
                use_render=render_load_env,
                case_num=scenario_num
            )
        )
        for index in range(scenario_num):
            print("Start replaying scenario {}".format(index))
            env.reset(force_seed=index)
            done = False
            count = 0
            while not done:
                o, r, done, i = env.step([0, 0])
                count += 1
            print("Finish replaying scenario {} with step {}".format(index, count))
    finally:
        env.close()
        if dir is not None:
            shutil.rmtree(dir)


if __name__ == "__main__":
    test_export_metadrive_scenario(render_export_env=False, render_load_env=False)
    # test_export_waymo_scenario(render_export_env=True, render_load_env=True)
