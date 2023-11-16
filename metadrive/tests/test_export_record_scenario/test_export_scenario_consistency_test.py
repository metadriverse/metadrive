import os
import pickle
import shutil

import numpy as np

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import NP_ARRAY_DECIMAL
from metadrive.scenario.utils import assert_scenario_equal
from metadrive.scenario.utils import read_dataset_summary
from metadrive.type import MetaDriveType
from metadrive.utils.math import wrap_to_pi


def test_export_metadrive_scenario_reproduction(num_scenarios=3, render_export_env=False, render_load_env=False):
    env = MetaDriveEnv(
        dict(start_seed=0, use_render=render_export_env, num_scenarios=num_scenarios, agent_policy=IDMPolicy)
    )
    policy = lambda x: [0, 1]
    dir1 = None
    try:
        scenarios, done_info = env.export_scenarios(policy, scenario_index=[i for i in range(num_scenarios)])
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
        scenarios2, done_info = env.export_scenarios(policy, scenario_index=[i for i in range(num_scenarios)])
    finally:
        env.close()

    if dir1 is not None:
        shutil.rmtree(dir1)

    # We can't make sure traffic vehicles has same name, so just test SDC here.
    assert_scenario_equal(scenarios, scenarios2, only_compare_sdc=True)


def test_export_metadrive_scenario_easy(num_scenarios=5, render_export_env=False, render_load_env=False):
    # ===== Save data =====
    env = MetaDriveEnv(
        dict(
            start_seed=0, map="SCS", use_render=render_export_env, num_scenarios=num_scenarios, agent_policy=IDMPolicy
        )
    )
    policy = lambda x: [0, 1]
    dir1 = None
    try:
        scenarios, done_info = env.export_scenarios(policy, scenario_index=[i for i in range(num_scenarios)])
        dir1 = os.path.join(os.path.dirname(__file__), "test_export_scenario_consistancy")
        os.makedirs(dir1, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir1, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
    finally:
        env.close()
        # pass

    # ===== Save data of the restoring environment =====
    env = ScenarioEnv(
        dict(
            agent_policy=ReplayEgoCarPolicy,
            data_directory=dir1,
            use_render=render_load_env,
            num_scenarios=num_scenarios,
            force_reuse_object_name=True,
            horizon=1000,
            # debug=True,
            # debug_static_world=True,
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
        scenarios, done_info = env.export_scenarios(
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
    env = ScenarioEnv(
        dict(
            agent_policy=ReplayEgoCarPolicy,
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
    env = ScenarioEnv(
        dict(
            agent_policy=ReplayEgoCarPolicy,
            use_render=render_export_env,
            start_scenario_index=0,
            data_directory=AssetLoader.file_path("waymo", unix_style=False),
            num_scenarios=num_scenarios
        )
    )
    policy = lambda x: [0, 1]
    dir = None
    try:
        scenarios, done_info = env.export_scenarios(
            policy, scenario_index=[i for i in range(num_scenarios)], verbose=True
        )
        dir = os.path.join(os.path.dirname(__file__), "../test_component/test_export")
        os.makedirs(dir, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
    finally:
        env.close()

    try:
        print("===== Start restoring =====")
        env = ScenarioEnv(
            dict(
                agent_policy=ReplayEgoCarPolicy,
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


def test_export_nuscenes_scenario(num_scenarios=2, render_export_env=False, render_load_env=False):
    env = ScenarioEnv(
        dict(
            data_directory=AssetLoader.file_path("nuscenes", unix_style=False),
            agent_policy=ReplayEgoCarPolicy,
            use_render=render_export_env,
            start_scenario_index=0,
            num_scenarios=num_scenarios
        )
    )
    policy = lambda x: [0, 1]
    try:
        scenarios, done_info = env.export_scenarios(
            policy, scenario_index=[i for i in range(num_scenarios)], verbose=True
        )
        dir = os.path.join(os.path.dirname(__file__), "../test_component/test_export")
        os.makedirs(dir, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
    finally:
        env.close()

    try:
        print("===== Start restoring =====")
        env = ScenarioEnv(
            dict(
                agent_policy=ReplayEgoCarPolicy,
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
        if os.path.exists(dir):
            shutil.rmtree(dir)
    finally:
        env.close()
        print("Finish!")
        # if dir is not None:
        #     shutil.rmtree(dir)


def compare_exported_scenario_with_origin(scenarios, data_manager, data_dir="waymo", compare_map=False):
    _, _, mapping = read_dataset_summary(AssetLoader.file_path(data_dir, unix_style=False))
    for index, scenario in scenarios.items():
        file_name = data_manager.summary_lookup[index]
        file_path = AssetLoader.file_path(data_dir, mapping[file_name], file_name, unix_style=False)
        with open(file_path, "rb+") as file:
            origin_data = pickle.load(file)
        export_data = scenario
        new_tracks = export_data["tracks"]

        for data in export_data["map_features"].values():
            assert "type" in data and ("polygon" in data or "polyline" in data)

        if compare_map:
            assert len(origin_data["map_features"]) == len(export_data["map_features"])

        original_ids = [new_tracks[obj_name]["metadata"]["original_id"] for obj_name in new_tracks.keys()]
        # assert len(set(original_ids)) == len(origin_data["tracks"]), "Object Num mismatch!"
        for obj_id, track in new_tracks.items():
            # if obj_id != scenario["metadata"]["sdc_id"]:
            new_pos = track["state"]["position"]
            new_heading = track["state"]["heading"]
            new_valid = track["state"]["valid"]
            old_id = track["metadata"]["original_id"]
            old_track = origin_data["tracks"][old_id]
            old_pos = old_track["state"]["position"]
            old_heading = old_track["state"]["heading"]
            old_valid = old_track["state"]["valid"]

            if track["type"] in [MetaDriveType.TRAFFIC_BARRIER, MetaDriveType.TRAFFIC_CONE]:
                index_to_compare = np.where(old_valid[:len(new_valid)])[0]
                assert new_valid[index_to_compare].all(), "Frame mismatch!"
                decimal = 0
            else:
                index_to_compare = np.where(new_valid)[0]
                decimal = NP_ARRAY_DECIMAL
            assert old_valid[index_to_compare].all(), "Frame mismatch!"
            old_pos = old_pos[index_to_compare][..., :2]
            new_pos = new_pos[index_to_compare][..., :2]
            np.testing.assert_almost_equal(old_pos, new_pos, decimal=decimal)

            old_heading = wrap_to_pi(old_heading[index_to_compare].reshape(-1))
            new_heading = wrap_to_pi(new_heading[index_to_compare].reshape(-1))
            np.testing.assert_almost_equal(old_heading, new_heading, decimal=decimal)

        for light_id, old_light in origin_data["dynamic_map_states"].items():
            new_light = export_data["dynamic_map_states"][light_id]

            if "stop_point" in old_light["state"]:
                old_pos = old_light["state"]["stop_point"][..., :2]
                old_pos = old_pos[np.where(old_pos > 0)[0][0]]

            else:
                old_pos = old_light["stop_point"]

            if "stop_point" in new_light["state"]:
                new_pos = new_light["state"]["stop_point"]
                ind = np.where(new_pos > 0)[0]
                if ind.size > 0:
                    ind = ind[0]
                else:
                    ind = 0
                new_pos = new_pos[ind]
            else:
                new_pos = new_light["stop_point"]

            length = min(len(old_pos), len(new_pos))
            np.testing.assert_almost_equal(old_pos[:2], new_pos[:2], decimal=NP_ARRAY_DECIMAL)

            if "lane" in old_light["state"]:
                old_light_lane = str(max(old_light["state"]["lane"]))
            else:
                old_light_lane = old_light["lane"]

            if "lane" in new_light["state"]:
                new_light_lane = max(new_light["state"]["lane"])
            else:
                new_light_lane = new_light["lane"]

            assert str(old_light_lane) == str(new_light_lane)

            for k, light_status in enumerate(old_light["state"]["object_state"][:length]):
                assert MetaDriveType.parse_light_status(light_status, simplifying=True) == \
                       new_light["state"]["object_state"][k]

        print("Finish Seed: {}".format(index))


def test_waymo_export_and_original_consistency(num_scenarios=3, render_export_env=False):
    env = ScenarioEnv(
        dict(
            agent_policy=ReplayEgoCarPolicy,
            use_render=render_export_env,
            start_scenario_index=0,
            num_scenarios=num_scenarios,
            data_directory=AssetLoader.file_path("waymo", unix_style=False),
            # force_reuse_object_name=True, # Don't allow discontinuous trajectory in our system
        )
    )
    policy = lambda x: [0, 1]
    dir = None
    try:
        scenarios, done_info = env.export_scenarios(
            policy, scenario_index=[i for i in range(num_scenarios)], verbose=True
        )
        compare_exported_scenario_with_origin(scenarios, env.engine.data_manager)
    finally:
        env.close()


def test_nuscenes_export_and_original_consistency(num_scenarios=7, render_export_env=False):
    assert num_scenarios <= 7
    env = ScenarioEnv(
        dict(
            data_directory=AssetLoader.file_path("nuscenes", unix_style=False),
            agent_policy=ReplayEgoCarPolicy,
            use_render=render_export_env,
            start_scenario_index=3,
            num_scenarios=num_scenarios
        )
    )
    policy = lambda x: [0, 1]
    dir = None
    try:
        scenarios, done_info = env.export_scenarios(
            policy, scenario_index=[i for i in range(3, 3 + num_scenarios)], verbose=True
        )
        compare_exported_scenario_with_origin(scenarios, env.engine.data_manager, data_dir="nuscenes", compare_map=True)
    finally:
        env.close()


if __name__ == "__main__":
    # test_export_metadrive_scenario_reproduction(num_scenarios=10)
    # test_export_metadrive_scenario_easy(render_export_env=False, render_load_env=False)
    # test_export_metadrive_scenario_hard(num_scenarios=3, render_export_env=True, render_load_env=True)
    # test_export_waymo_scenario(num_scenarios=3, render_export_env=False, render_load_env=False)
    # test_waymo_export_and_original_consistency(num_scenarios=3, render_export_env=False)
    # test_export_nuscenes_scenario(num_scenarios=2, render_export_env=False, render_load_env=False)
    test_nuscenes_export_and_original_consistency()
