from tqdm import tqdm

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


def _test_level(level=1, render=False):
    env = ScenarioEnv(
        {
            "use_render": render,
            "agent_policy": ReplayEgoCarPolicy,
            "sequential_seed": True,
            "reactive_traffic": False,
            "window_size": (1600, 900),
            "num_scenarios": 10,
            "horizon": 1000,
            "curriculum_level": level,
            "no_static_vehicles": True,
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
        }
    )
    try:
        scenario_id = set()
        for i in tqdm(range(10), desc=str(level)):
            env.reset(seed=i)
            for i in range(10):
                o, r, d, _, _ = env.step([0, 0])
                if d:
                    break

            scenario_id.add(env.engine.data_manager.current_scenario_summary["id"])
        assert len(scenario_id) == (env.engine.current_level + 1) * int(10 / level)
    finally:
        env.close()


def test_curriculum_seed():
    _test_level(level=5)
    _test_level(level=1)
    _test_level(level=2)
    # _test_level(level=3)


def test_curriculum_up_1_level(render=False, level=5):
    env = ScenarioEnv(
        {
            "use_render": render,
            "agent_policy": ReplayEgoCarPolicy,
            "sequential_seed": True,
            "reactive_traffic": False,
            "window_size": (1600, 900),
            "num_scenarios": 10,
            "episodes_to_evaluate_curriculum": 2,
            "horizon": 1000,
            "curriculum_level": level,
            "no_static_vehicles": True,
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
        }
    )
    try:
        scenario_id = []
        for i in tqdm(range(10), desc=str(level)):
            env.reset(seed=i)
            for i in range(10):
                o, r, d, _, _ = env.step([0, 0])
            scenario_id.append(env.engine.data_manager.current_scenario_summary["id"])
        assert len(set(scenario_id)) == 4
        ids = [env.engine.data_manager.summary_dict[f]["id"] for f in env.engine.data_manager.summary_lookup]
        assert set(scenario_id) == set(ids[:4])
    finally:
        env.close()


def test_curriculum_level_up(render=False):
    env = ScenarioEnv(
        {
            "use_render": render,
            "agent_policy": ReplayEgoCarPolicy,
            "sequential_seed": True,
            "reactive_traffic": False,
            "window_size": (1600, 900),
            "num_scenarios": 10,
            "episodes_to_evaluate_curriculum": 5,
            "target_success_rate": 1.,
            "horizon": 1000,
            "curriculum_level": 2,
            "no_static_vehicles": True,
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
        }
    )
    try:
        scenario_id = []
        for i in tqdm(range(20), desc=str(2)):
            env.reset()
            for i in range(250):
                o, r, d, _, _ = env.step([0, 0])
            scenario_id.append(env.engine.data_manager.current_scenario_summary["id"])
        assert len(set(scenario_id)) == 10
        ids = [env.engine.data_manager.summary_dict[f]["id"] for f in env.engine.data_manager.summary_lookup]
        assert scenario_id[:10] == ids
        assert scenario_id[-5:] == ids[-5:] == scenario_id[-10:-5]
    finally:
        env.close()


def _worker_env(render, worker_index, level_up=False):
    assert worker_index in [0, 1]
    level = 2
    env = ScenarioEnv(
        {
            "use_render": render,
            "agent_policy": ReplayEgoCarPolicy,
            "sequential_seed": True,
            "reactive_traffic": False,
            "window_size": (1600, 900),
            "num_scenarios": 8,
            "episodes_to_evaluate_curriculum": 4,
            "curriculum_level": level,
            "no_static_vehicles": True,
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
            "worker_index": worker_index,
            "num_workers": 2,
        }
    )
    try:
        env.reset()
        if level_up:
            env.engine.curriculum_manager._level_up()
            env.reset()
        scenario_id = []
        for i in range(20):
            env.reset()
            for i in range(10):
                o, r, d, _, _ = env.step([0, 0])
            scenario_id.append(env.engine.data_manager.current_scenario_summary["id"])
            print(env.current_seed)
        all_scenario = [
            env.engine.data_manager.summary_dict[f]["id"] for f in env.engine.data_manager.summary_lookup[:8]
        ]
        assert len(set(scenario_id)) == 2
        assert env.engine.data_manager.data_coverage == 0.75 if level_up else 0.5
    finally:
        env.close()
    return scenario_id, all_scenario


def test_curriculum_multi_worker(render=False):
    # 1
    all_scenario_id = []
    all_scenario_id.extend(_worker_env(render, 1, level_up=True)[0])
    set_1, all_scenario = _worker_env(render, 0, level_up=True)
    all_scenario_id.extend(set_1)

    ids = all_scenario
    assert set(all_scenario_id) == set(ids[-4:])
    # 2
    all_scenario_id = []
    set_1, all_scenario = _worker_env(render, 0)
    all_scenario_id.extend(set_1)
    all_scenario_id.extend(_worker_env(render, 1)[0])

    ids = all_scenario
    assert set(all_scenario_id) == set(ids[:4])


def level_up_worker(render, worker_index):
    env = ScenarioEnv(
        {
            "use_render": render,
            "agent_policy": ReplayEgoCarPolicy,
            "sequential_seed": True,
            "reactive_traffic": False,
            "window_size": (1600, 900),
            "num_scenarios": 8,
            "episodes_to_evaluate_curriculum": 2,
            "horizon": 1000,
            "curriculum_level": 4,
            "no_static_vehicles": True,
            "num_workers": 2,
            "worker_index": worker_index,
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
        }
    )
    try:
        scenario_id = []
        for i in tqdm(range(20), desc=str(2)):
            env.reset()
            for i in range(250):
                o, r, d, _, _ = env.step([0, 0])
            scenario_id.append(env.engine.data_manager.current_scenario_summary["id"])
        assert len(set(scenario_id)) == 4
        ids = [env.engine.data_manager.summary_dict[f]["id"] for f in env.engine.data_manager.summary_lookup[:8]]
        assert scenario_id[:4] == ids[worker_index::2]
        assert scenario_id[-5:] == [ids[-2] if worker_index == 0 else ids[-1]] * 5 == scenario_id[-10:-5]
    finally:
        env.close()
    return scenario_id, ids


def test_curriculum_worker_level_up(render=False):
    scenario_id_1, all_id = level_up_worker(render, 0)
    scenario_id_2, all_id = level_up_worker(render, 1)
    assert scenario_id_1[:4] + scenario_id_2[:4] == all_id[::2] + all_id[1::2]


def test_start_seed_not_0(render=False, worker_index=0):
    env = ScenarioEnv(
        {
            "use_render": render,
            "agent_policy": ReplayEgoCarPolicy,
            "sequential_seed": True,
            "reactive_traffic": False,
            "window_size": (1600, 900),
            "num_scenarios": 8,
            "start_scenario_index": 2,
            "episodes_to_evaluate_curriculum": 2,
            "horizon": 1000,
            "curriculum_level": 4,
            "no_static_vehicles": True,
            "num_workers": 2,
            "worker_index": worker_index,
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
        }
    )
    try:
        scenario_id = []
        for i in tqdm(range(20), desc=str(2)):
            env.reset()
            for i in range(250):
                o, r, d, _, _ = env.step([0, 0])
            scenario_id.append(env.engine.data_manager.current_scenario_summary["id"])
        all_scenarios = sorted(list(env.engine.data_manager.summary_dict.keys()))[2:]
        summary_lookup = env.engine.data_manager.summary_lookup[2:]
        assert set(all_scenarios) == set(summary_lookup)
        assert len(set(scenario_id)) == 4
        ids = [env.engine.data_manager.summary_dict[f]["id"] for f in summary_lookup]
        assert scenario_id[:4] == ids[worker_index::2]
        assert scenario_id[-5:] == [ids[-2] if worker_index == 0 else ids[-1]] * 5 == scenario_id[-10:-5]
    finally:
        env.close()
    return scenario_id, ids


def test_start_seed_1_9(render=False, worker_index=0):
    env = ScenarioEnv(
        {
            "use_render": render,
            "agent_policy": ReplayEgoCarPolicy,
            "sequential_seed": True,
            "reactive_traffic": False,
            "window_size": (1600, 900),
            "num_scenarios": 8,
            "start_scenario_index": 1,
            "episodes_to_evaluate_curriculum": 2,
            "horizon": 1000,
            "curriculum_level": 4,
            "no_static_vehicles": True,
            "num_workers": 2,
            "worker_index": worker_index,
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
        }
    )
    try:
        scenario_id = []
        for i in tqdm(range(20), desc=str(2)):
            env.reset()
            for i in range(250):
                o, r, d, _, _ = env.step([0, 0])
            scenario_id.append(env.engine.data_manager.current_scenario_summary["id"])
        all_scenarios = sorted(list(env.engine.data_manager.summary_dict.keys()))[1:9]
        summary_lookup = env.engine.data_manager.summary_lookup[1:9]
        assert set(all_scenarios) == set(summary_lookup)
        assert len(set(scenario_id)) == 4
        ids = [env.engine.data_manager.summary_dict[f]["id"] for f in summary_lookup]
        assert scenario_id[:4] == ids[worker_index::2]
        assert scenario_id[-5:] == [ids[-2] if worker_index == 0 else ids[-1]] * 5 == scenario_id[-10:-5]
    finally:
        env.close()
    return scenario_id, ids


if __name__ == '__main__':
    test_curriculum_multi_worker()
    # test_curriculum_seed()
    # test_curriculum_level_up()
    # test_curriculum_worker_level_up()
    # test_start_seed_not_0()
    # test_start_seed_1_9()
