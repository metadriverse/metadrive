import copy

import numpy as np

from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.idm_policy import TrajectoryIDMPOlicy


def test_trajectory_idm(render=False):
    env = WaymoEnv(
        {
            "use_render": render,
            "agent_policy": TrajectoryIDMPOlicy,
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
            o = env.reset(force_seed=seed)
            sdc_route = env.engine.map_manager.current_sdc_route
            v_config = copy.deepcopy(env.engine.global_config["vehicle_config"])
            v_config.update(
                dict(
                    need_navigation=False,
                    show_navi_mark=False,
                    show_dest_mark=False,
                    enable_reverse=False,
                    show_lidar=False,
                    show_lane_line_detector=False,
                    show_side_detector=False,
                )
            )
            v_list = []
            list = [(25, 0, 0)] if seed != 0 else []
            list += [(45, -3, np.pi / 2), (70, 3, -np.pi / 4)]
            for long, lat, heading in list:
                position = sdc_route.position(long, lat)
                v = env.engine.spawn_object(SVehicle, vehicle_config=v_config, position=position, heading=heading)
                v_list.append(v)

            for s in range(1000):
                o, r, d, info = env.step(env.action_space.sample())
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

                if d:
                    assert info["arrive_dest"]
                    break
    finally:
        env.close()


if __name__ == "__main__":
    test_trajectory_idm(True)
    # test_check_multi_discrete_space()
