import copy
from metadrive.utils import recursive_equal

import numpy as np

from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.envs.scenario_env import ScenarioEnv, AssetLoader
from metadrive.policy.idm_policy import TrajectoryIDMPolicy


def test_trajectory_idm(render=False):
    env = ScenarioEnv(
        {
            "use_render": render,
            "agent_policy": TrajectoryIDMPolicy,
            "data_directory": AssetLoader.file_path("waymo", unix_style=False),
            "start_scenario_index": 0,
            # "show_coordinates": True,
            # "start_scenario_index": 1000,
            # "show_coordinates": True,
            "num_scenarios": 3,
            # "show_policy_mark": True,
            # "no_static_vehicles": True,
            "no_traffic": True,
            "horizon": 1000,
            "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
        }
    )
    try:
        for seed in [0, 1, 2]:
            o, _ = env.reset(seed=seed)
            sdc_route = env.engine.map_manager.current_sdc_route
            v_config = copy.deepcopy(env.engine.global_config["vehicle_config"])
            overwrite_config = dict(
                navigation_module=None,
                show_navi_mark=True if seed == 1 else False,
                show_dest_mark=False,
                enable_reverse=True if seed == 0 else False,
                lidar=dict(
                    num_lasers=240 if seed == 2 else 120,
                    distance=50,
                    num_others=0,
                    gaussian_noise=0.0,
                    dropout_prob=0.0,
                    add_others_navi=False
                ),
                show_lidar=False,
                show_lane_line_detector=False,
                show_side_detector=False,
            )
            v_config.update(overwrite_config)

            v_list = []
            list = [(25, 0, 0)] if seed != 0 else []
            list += [(45, -3, np.pi / 2), (70, 3, -np.pi / 4)]
            for long, lat, heading in list:
                position = sdc_route.position(long, lat)
                v_1 = env.engine.spawn_object(
                    SVehicle, vehicle_config=overwrite_config, position=position, heading=heading
                )
                v = env.engine.spawn_object(
                    SVehicle, vehicle_config=v_config, position=position, heading=heading, random_seed=v_1.random_seed
                )
                assert recursive_equal(v.config, v_1.config, need_assert=True)
                env.engine.clear_objects([v_1.id])
                v_list.append(v)

            for s in range(1000):
                o, r, tm, tc, info = env.step(env.action_space.sample())
                if s == 100:
                    v = v_list.pop(0)
                    env.engine.clear_objects([v.id])

                if s == 200:
                    v = v_list.pop(0)
                    env.engine.clear_objects([v.id])

                if s == 300:
                    v = v_list.pop(0)
                    env.engine.clear_objects([v.id])
                    assert len(v_list) == 0

                assert not info["crash"]

                if tm or tc:
                    assert info["arrive_dest"]
                    break
    finally:
        env.close()


if __name__ == "__main__":
    test_trajectory_idm(False)
    # test_check_multi_discrete_space()
