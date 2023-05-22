from tqdm import tqdm

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.real_data_envs.nuscenes_env import NuScenesEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


def _test_level(level=1, render=False):
    env = NuScenesEnv(
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
            "data_directory": AssetLoader.file_path("nuscenes", return_raw_style=False),
        }
    )
    try:
        scenario_id = set()
        for i in tqdm(range(10), desc=str(level)):
            env.reset(force_seed=i)
            for i in range(10):
                o, r, d, _ = env.step([0, 0])
                if d:
                    break

            scenario_id.add(env.engine.data_manager.current_scenario_summary["id"])
        assert len(scenario_id) == int(10 / level)
    finally:
        env.close()


def test_curriculum_seed():
    _test_level(level=5)
    _test_level(level=1)
    _test_level(level=2)
    _test_level(level=3)


def test_curriculum_up_1_level(render=False, level=5):
    env = NuScenesEnv(
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
            "data_directory": AssetLoader.file_path("nuscenes", return_raw_style=False),
        }
    )
    try:
        scenario_id = []
        for i in tqdm(range(10), desc=str(level)):
            env.reset(force_seed=i)
            for i in range(10):
                o, r, d, _ = env.step([0, 0])
            scenario_id.append(env.engine.data_manager.current_scenario_summary["id"])
        assert len(set(scenario_id)) == 4
        ids = [env.engine.data_manager.summary_dict[f]["id"] for f in env.engine.data_manager.summary_lookup]
        assert set(scenario_id) == set(ids[:4])
    finally:
        env.close()


def test_curriculum_level_up(render=False, level=5):
    env = NuScenesEnv(
        {
            "use_render": render,
            "agent_policy": ReplayEgoCarPolicy,
            "sequential_seed": True,
            "reactive_traffic": False,
            "window_size": (1600, 900),
            "num_scenarios": 10,
            "episodes_to_evaluate_curriculum": 2,
            "horizon": 1000,
            "curriculum_level": int(10 / level),
            "no_static_vehicles": True,
            "data_directory": AssetLoader.file_path("nuscenes", return_raw_style=False),
        }
    )
    try:
        scenario_id = []
        for i in tqdm(range(20), desc=str(level)):
            env.reset()
            for i in range(250):
                o, r, d, _ = env.step([0, 0])
            scenario_id.append(env.engine.data_manager.current_scenario_summary["id"])
        assert len(set(scenario_id)) == 10
        ids = [env.engine.data_manager.summary_dict[f]["id"] for f in env.engine.data_manager.summary_lookup]
        assert scenario_id[:10] == ids
        assert scenario_id[-2:] == ids[-2:] == scenario_id[-4:-2]
    finally:
        env.close()


if __name__ == '__main__':
    # test_curriculum_seed()
    test_curriculum_level_up()
