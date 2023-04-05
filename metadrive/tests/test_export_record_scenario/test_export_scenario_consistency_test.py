import os
import pickle
import shutil

import numpy as np

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.replay_policy import WaymoReplayEgoCarPolicy
from metadrive.scenario import ScenarioDescription as SD
from metadrive.utils.coordinates_shift import waymo_to_metadrive_vector
from metadrive.utils.math_utils import wrap_to_pi

NP_ARRAY_DECIMAL = 4
VELOCITY_DECIMAL = 1  # velocity can not be set accurately

MIN_LENGTH_RATIO = 0.8


def assert_scenario_equal(scenarios1, scenarios2, only_compare_sdc=False, coordinate_transform=False):
    # ===== These two set of data should align =====
    assert set(scenarios1.keys()) == set(scenarios2.keys())
    for scenario_id in scenarios1.keys():
        SD.sanity_check(scenarios1[scenario_id], check_self_type=True)
        SD.sanity_check(scenarios2[scenario_id], check_self_type=True)
        old_scene = SD(scenarios1[scenario_id])
        new_scene = SD(scenarios2[scenario_id])
        SD.sanity_check(old_scene)
        SD.sanity_check(new_scene)
        assert old_scene[SD.LENGTH] >= new_scene[SD.LENGTH], (old_scene[SD.LENGTH], new_scene[SD.LENGTH])

        if only_compare_sdc:
            sdc1 = old_scene[SD.METADATA][SD.SDC_ID]
            sdc2 = new_scene[SD.METADATA][SD.SDC_ID]
            state_dict1 = old_scene[SD.TRACKS][sdc1]
            state_dict2 = new_scene[SD.TRACKS][sdc2]
            min_len = min(state_dict1[SD.STATE]["position"].shape[0], state_dict2[SD.STATE]["position"].shape[0])
            max_len = max(state_dict1[SD.STATE]["position"].shape[0], state_dict2[SD.STATE]["position"].shape[0])
            assert min_len / max_len > MIN_LENGTH_RATIO, "Replayed Scenario length ratio: {}".format(min_len / max_len)
            for k in state_dict1[SD.STATE].keys():
                if k in ["action", "throttle_brake", "steering"]:
                    continue
                elif k == "position":
                    np.testing.assert_almost_equal(
                        state_dict1[SD.STATE][k][:min_len][..., :2],
                        state_dict2[SD.STATE][k][:min_len][..., :2],
                        decimal=NP_ARRAY_DECIMAL
                    )
                elif k == "heading":
                    np.testing.assert_almost_equal(
                        wrap_to_pi(state_dict1[SD.STATE][k][:min_len] - state_dict2[SD.STATE][k][:min_len]),
                        np.zeros_like(state_dict2[SD.STATE][k][:min_len]),
                        decimal=NP_ARRAY_DECIMAL
                    )
                elif k == "velocity":
                    np.testing.assert_almost_equal(
                        state_dict1[SD.STATE][k][:min_len],
                        state_dict2[SD.STATE][k][:min_len],
                        decimal=VELOCITY_DECIMAL
                    )
            assert state_dict1[SD.TYPE] == state_dict2[SD.TYPE]

        else:
            assert set(old_scene[SD.TRACKS].keys()).issuperset(set(new_scene[SD.TRACKS].keys()) - {"default_agent"})
            for track_id, track in old_scene[SD.TRACKS].items():
                if track_id == "default_agent":
                    continue
                if track_id not in new_scene[SD.TRACKS]:
                    continue
                for state_k in new_scene[SD.TRACKS][track_id][SD.STATE]:
                    state_array_1 = new_scene[SD.TRACKS][track_id][SD.STATE][state_k]
                    state_array_2 = track[SD.STATE][state_k]
                    min_len = min(state_array_1.shape[0], state_array_2.shape[0])
                    max_len = max(state_array_1.shape[0], state_array_2.shape[0])
                    assert min_len / max_len > MIN_LENGTH_RATIO, "Replayed Scenario length ratio: {}".format(
                        min_len / max_len
                    )

                    if state_k == "velocity":
                        decimal = VELOCITY_DECIMAL
                    else:
                        decimal = NP_ARRAY_DECIMAL

                    if state_k == "heading":
                        # error < 5.7 degree is acceptable
                        broader_ratio = 1
                        ret = abs(wrap_to_pi(state_array_1[:min_len] - state_array_2[:min_len])) < 1e-1
                        ratio = np.sum(np.asarray(ret, dtype=np.int8)) / len(ret)
                        if ratio < broader_ratio:
                            raise ValueError("Match ration: {}, Target: {}".format(ratio, broader_ratio))

                        strict_ratio = 0.98
                        ret = abs(wrap_to_pi(state_array_1[:min_len] - state_array_2[:min_len])) < 1e-4
                        ratio = np.sum(np.asarray(ret, dtype=np.int8)) / len(ret)
                        if ratio < strict_ratio:
                            raise ValueError("Match ration: {}, Target: {}".format(ratio, strict_ratio))
                    else:
                        strict_ratio = 0.99
                        ret = abs(wrap_to_pi(state_array_1[:min_len] - state_array_2[:min_len])) < pow(10, -decimal)
                        ratio = np.sum(np.asarray(ret, dtype=np.int8)) / len(ret)
                        if ratio < strict_ratio:
                            raise ValueError("Match ration: {}, Target: {}".format(ratio, strict_ratio))

                assert new_scene[SD.TRACKS][track_id][SD.TYPE] == track[SD.TYPE]

            track_id = "default_agent"
            for k in new_scene.get_sdc_track()["state"]:
                state_array_1 = new_scene.get_sdc_track()["state"][k]
                state_array_2 = old_scene.get_sdc_track()["state"][k]
                min_len = min(state_array_1.shape[0], state_array_2.shape[0])
                max_len = max(state_array_1.shape[0], state_array_2.shape[0])
                assert min_len / max_len > MIN_LENGTH_RATIO, "Replayed Scenario length ratio: {}".format(
                    min_len / max_len
                )

                if k == "velocity":
                    decimal = VELOCITY_DECIMAL
                else:
                    decimal = NP_ARRAY_DECIMAL
                np.testing.assert_almost_equal(state_array_1[:min_len], state_array_2[:min_len], decimal=decimal)

            assert new_scene[SD.TRACKS][track_id][SD.TYPE] == track[SD.TYPE]

        assert set(old_scene[SD.MAP_FEATURES].keys()).issuperset(set(new_scene[SD.MAP_FEATURES].keys()))
        assert set(old_scene[SD.DYNAMIC_MAP_STATES].keys()) == set(new_scene[SD.DYNAMIC_MAP_STATES].keys())

        for map_id, map_feat in new_scene[SD.MAP_FEATURES].items():
            # It is possible that some line are not included in new scene but exist in old scene.
            old_scene_polyline = map_feat["polyline"]
            if coordinate_transform:
                old_scene_polyline = waymo_to_metadrive_vector(old_scene_polyline)
            np.testing.assert_almost_equal(
                new_scene[SD.MAP_FEATURES][map_id]["polyline"], map_feat["polyline"], decimal=NP_ARRAY_DECIMAL
            )
            assert new_scene[SD.MAP_FEATURES][map_id][SD.TYPE] == map_feat[SD.TYPE]

        for obj_id, obj_state in old_scene[SD.DYNAMIC_MAP_STATES].items():
            new_state_dict = new_scene[SD.DYNAMIC_MAP_STATES][obj_id][SD.STATE]
            old_state_dict = obj_state[SD.STATE]
            assert set(new_state_dict.keys()) == set(old_state_dict.keys())
            for k in new_state_dict.keys():
                min_len = min(new_state_dict[k].shape[0], old_state_dict[k].shape[0])
                max_len = max(new_state_dict[k].shape[0], old_state_dict[k].shape[0])
                assert min_len / max_len > MIN_LENGTH_RATIO, "Replayed Scenario length ratio: {}".format(
                    min_len / max_len
                )
                np.testing.assert_almost_equal(
                    new_state_dict[k][:min_len], old_state_dict[k][:min_len], decimal=NP_ARRAY_DECIMAL
                )

            assert new_scene[SD.DYNAMIC_MAP_STATES][obj_id][SD.TYPE] == obj_state[SD.TYPE]


def test_export_metadrive_scenario_reproduction(num_scenarios=3, render_export_env=False, render_load_env=False):
    env = MetaDriveEnv(
        dict(start_seed=0, use_render=render_export_env, num_scenarios=num_scenarios, agent_policy=IDMPolicy)
    )
    policy = lambda x: [0, 1]
    dir1 = None
    try:
        scenarios = env.export_scenarios(policy, scenario_index=[i for i in range(num_scenarios)])
        dir1 = os.path.join(os.path.dirname(__file__), "../test_component/test_export")
        os.makedirs(dir1, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir1, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
    finally:
        env.close()

    # Same environment, same config
    env = MetaDriveEnv(
        dict(start_seed=0, use_render=render_load_env, num_scenarios=num_scenarios, agent_policy=IDMPolicy)
    )
    policy = lambda x: [0, 1]
    try:
        scenarios2 = env.export_scenarios(policy, scenario_index=[i for i in range(num_scenarios)])
    finally:
        env.close()

    if dir1 is not None:
        shutil.rmtree(dir1)

    # We can't make sure traffic vehicles has same name, so just test SDC here.
    assert_scenario_equal(scenarios, scenarios2, only_compare_sdc=True)


def test_export_metadrive_scenario_easy(num_scenarios=5, render_export_env=False, render_load_env=False):
    # ===== Save data =====
    env = MetaDriveEnv(
        dict(start_seed=0, map="S", use_render=render_export_env, num_scenarios=num_scenarios, agent_policy=IDMPolicy)
    )
    policy = lambda x: [0, 1]
    dir1 = None
    try:
        scenarios = env.export_scenarios(policy, scenario_index=[i for i in range(num_scenarios)])
        dir1 = os.path.join(os.path.dirname(__file__), "test_export_scenario_consistancy")
        os.makedirs(dir1, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir1, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
    finally:
        env.close()
        # pass

    # ===== Save data of the restoring environment =====
    env = WaymoEnv(
        dict(
            agent_policy=WaymoReplayEgoCarPolicy,
            data_directory=dir1,
            use_render=render_load_env,
            num_scenarios=num_scenarios,
            force_reuse_object_name=True,
            vehicle_config=dict(no_wheel_friction=True)
        )
    )
    try:
        scenarios_restored, done_info = env.export_scenarios(
            policy,
            scenario_index=[i for i in range(num_scenarios)],
            render_topdown=render_load_env,
            return_done_info=True
        )
        for seed, info in done_info.items():
            if not info["arrive_dest"]:
                raise ValueError("Seed: {} Can not arrive dest!".format(seed))
    finally:
        env.close()

    if dir1 is not None:
        shutil.rmtree(dir1)

    assert_scenario_equal(scenarios, scenarios_restored, only_compare_sdc=False)


def test_export_metadrive_scenario_hard(start_seed=0, num_scenarios=3, render_export_env=False, render_load_env=False):
    # ===== Save data =====
    env = MetaDriveEnv(
        dict(
            start_seed=start_seed,
            map=7,
            use_render=render_export_env,
            num_scenarios=num_scenarios,
            agent_policy=IDMPolicy
        )
    )
    policy = lambda x: [0, 1]
    dir1 = None
    try:
        scenarios = env.export_scenarios(
            policy, scenario_index=[i for i in range(start_seed, start_seed + num_scenarios)]
        )
        dir1 = os.path.join(os.path.dirname(__file__), "test_export_metadrive_scenario_hard")
        os.makedirs(dir1, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir1, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
    finally:
        env.close()
        # pass

    # ===== Save data of the restoring environment =====
    env = WaymoEnv(
        dict(
            agent_policy=WaymoReplayEgoCarPolicy,
            data_directory=dir1,
            use_render=render_load_env,
            num_scenarios=num_scenarios,
            start_scenario_index=start_seed,
            debug=True,
            force_reuse_object_name=True,
            vehicle_config=dict(no_wheel_friction=True)
            # debug_physics_world=True,
            # debug_static_world=True
        )
    )
    try:
        scenarios_restored, done_info = env.export_scenarios(
            policy,
            scenario_index=[i for i in range(num_scenarios)],
            render_topdown=render_load_env,
            return_done_info=True
        )
        for seed, info in done_info.items():
            if not info["arrive_dest"]:
                raise ValueError("Seed: {} Can not arrive dest!".format(seed))
    finally:
        env.close()

    if dir1 is not None:
        shutil.rmtree(dir1)

    assert_scenario_equal(scenarios, scenarios_restored, only_compare_sdc=False)


def test_export_waymo_scenario(num_scenarios=3, render_export_env=False, render_load_env=False):
    env = WaymoEnv(
        dict(
            agent_policy=WaymoReplayEgoCarPolicy,
            use_render=render_export_env,
            start_scenario_index=0,
            num_scenarios=num_scenarios
        )
    )
    policy = lambda x: [0, 1]
    dir = None
    try:
        scenarios = env.export_scenarios(policy, scenario_index=[i for i in range(num_scenarios)], verbose=True)
        dir = os.path.join(os.path.dirname(__file__), "../test_component/test_export")
        os.makedirs(dir, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
    finally:
        env.close()

    try:
        print("===== Start restoring =====")
        env = WaymoEnv(
            dict(
                agent_policy=WaymoReplayEgoCarPolicy,
                data_directory=dir,
                use_render=render_load_env,
                num_scenarios=num_scenarios,
                force_reuse_object_name=True,
                vehicle_config=dict(no_wheel_friction=True)
            )
        )
        scenarios_restored, done_info = env.export_scenarios(
            policy,
            scenario_index=[i for i in range(num_scenarios)],
            render_topdown=render_load_env,
            return_done_info=True
        )
        for seed, info in done_info.items():
            if not info["arrive_dest"]:
                raise ValueError("Seed: {} Can not arrive dest!".format(seed))

    finally:
        env.close()
        # if dir is not None:
        #     shutil.rmtree(dir)

    assert_scenario_equal(scenarios, scenarios_restored, only_compare_sdc=False, coordinate_transform=True)


if __name__ == "__main__":
    # test_export_metadrive_scenario_reproduction(num_scenarios=10)
    test_export_metadrive_scenario_easy(num_scenarios=1, render_export_env=False, render_load_env=False)
    # test_export_metadrive_scenario_hard(num_scenarios=3, render_export_env=True, render_load_env=True)
    # test_export_waymo_scenario(num_scenarios=3, render_export_env=False, render_load_env=False)
