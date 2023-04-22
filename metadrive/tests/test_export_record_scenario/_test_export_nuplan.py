import os
import pickle
import shutil

from metadrive.envs.real_data_envs.nuplan_env import NuPlanEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import NuPlanReplayEgoCarPolicy, ReplayEgoCarPolicy
from metadrive.scenario.utils import assert_scenario_equal


def _test_export_nuplan_scenario_hard(start_seed=0, num_scenarios=5, render_export_env=False, render_load_env=False):
    # ===== Save data =====
    env = NuPlanEnv(
        {
            "use_render": render_export_env,
            "agent_policy": NuPlanReplayEgoCarPolicy,
            "start_scenario_index": start_seed,
            "load_city_map": True,
            "num_scenarios": num_scenarios,
            "vehicle_config": dict(
                lidar=dict(num_lasers=120, distance=50, num_others=0),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50),
                # show_lidar=True
                show_navi_mark=False,
                show_dest_mark=False,
                no_wheel_friction=True,
            ),
        }
    )
    policy = lambda x: [0, 1]
    dir1 = None
    try:
        scenarios, done_info = env.export_scenarios(
            policy, scenario_index=[i for i in range(start_seed, start_seed + num_scenarios)]
        )
        dir1 = os.path.join(os.path.dirname(__file__), "test_export_nuplan_scenario_hard")
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
            force_reuse_object_name=True,
            # debug=True,
            # debug_static_world=True,
            vehicle_config=dict(no_wheel_friction=True)
        )
    )
    try:
        scenarios_restored, done_info = env.export_scenarios(
            policy, scenario_index=[i for i in range(num_scenarios)], render_topdown=False, return_done_info=True
        )
        for seed, info in done_info.items():
            if not info["arrive_dest"]:
                raise ValueError("Seed: {} Can not arrive dest!".format(seed))
    finally:
        env.close()

    if dir1 is not None:
        shutil.rmtree(dir1)

    assert_scenario_equal(scenarios, scenarios_restored, only_compare_sdc=False)


if __name__ == "__main__":
    _test_export_nuplan_scenario_hard(num_scenarios=5, render_export_env=False, render_load_env=True)
