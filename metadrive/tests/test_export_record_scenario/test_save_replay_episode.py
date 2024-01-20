import pickle
from metadrive.utils.math import wrap_to_pi
import numpy as np

from metadrive import MultiAgentRoundaboutEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import setup_logger


def test_save_episode(vis=False):
    setup_logger(True)

    test_dump = True
    save_episode = True
    vis = vis
    env = SafeMetaDriveEnv(
        {
            "accident_prob": 0.8,
            "num_scenarios": 1,
            "traffic_density": 0.1,
            "start_seed": 1000,
            # "manual_control": vis,
            "use_render": False,
            "agent_policy": IDMPolicy,
            "traffic_mode": TrafficMode.Trigger,
            "record_episode": save_episode,
            "map_config": {
                BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                BaseMap.GENERATE_CONFIG: "CrCSC",
                BaseMap.LANE_WIDTH: 3.5,
                BaseMap.LANE_NUM: 3,
            }
        }
    )
    step_info = []
    try:
        o, _ = env.reset()
        for i in range(1, 100000 if vis else 2000):
            step_info.append(
                {
                    name: [obj.position, obj.heading_theta, obj.class_name]
                    for name, obj in env.engine._spawned_objects.items()
                }
            )
            o, r, tm, tc, info = env.step([0, 1])
            if vis:
                env.render(mode="top_down")
            if tm or tc:
                epi_info = env.engine.dump_episode("test_dump_single.pkl" if test_dump else None)
                break
        # with open("../test_export_record_scenario/test_dump_single.pkl", "rb") as f:
        env.config["replay_episode"] = epi_info
        env.config["record_episode"] = False
        o, _ = env.reset()
        for i in range(0, 100000 if vis else 2000):
            # if i % 5 ==0:
            for old_id, new_id in env.engine.replay_manager.record_name_to_current_name.items():
                obj = env.engine.replay_manager.spawned_objects[new_id]
                pos = obj.position
                heading = obj.heading_theta
                record_pos = env.engine.replay_manager.current_frame.step_info[old_id]["position"]
                record_heading = env.engine.replay_manager.current_frame.step_info[old_id]["heading_theta"]
                assert np.isclose(np.array([pos[0], pos[1], obj.get_z()]), np.array(record_pos)).all()
                assert abs(wrap_to_pi(heading - record_heading)) < 1e-2

                assert np.isclose(np.array([pos[0], pos[1]]), np.array(step_info[i][old_id][0]), 1e-2, 1e-2).all()
                assert abs(wrap_to_pi(heading - np.array(step_info[i][old_id][1]))) < 1e-2
            # assert abs(env.agent.get_z() - record_pos[-1]) < 1e-3
            o, r, tm, tc, info = env.step([0, 1])
            if vis:
                env.render()
            if info.get("replay_done", False):
                break
    finally:
        env.close()


def test_save_episode_marl(vis=False):
    """
    1. Set record_episode=True to record each episode
    2. dump_episode when terminated[__all__] == True
    3. You can keep recent episodes
    4. Input episode data to reset() function can replay the episode !
    """

    # setup_logger(True)

    test_dump = True
    dump_recent_episode = 5
    dump_count = 0
    env = MultiAgentRoundaboutEnv(
        dict(use_render=vis, manual_control=False, record_episode=True, horizon=100, force_seed_spawn_manager=True)
    )
    try:
        # Test Record
        o, _ = env.reset(seed=0)
        epi_info = None
        # for tt in range(10, 100):
        tt = 13
        # print("\nseed: {}\n".format(tt))
        env.engine.spawn_manager.seed(tt)
        o, _ = env.reset()
        for i in range(1, 100000 if vis else 600):
            o, r, tm, tc, info = env.step({agent_id: [0, .2] for agent_id in env.agents.keys()})
            if vis:
                env.render()
            if tm["__all__"]:
                epi_info = env.engine.dump_episode("test_dump.pkl")
                # test dump json
                # if test_dump:
                #     with open("test_dump_{}.json".format(dump_count), "w") as f:
                #         json.dump(epi_info, f)
                #     dump_count += 1
                #     dump_count = dump_count % dump_recent_episode
                break
                # env.reset()

        # with open("../test_export_record_scenario/test_dump.pkl", "rb") as epi_record:
        # input episode_info to restore
        env.config["replay_episode"] = epi_info

        env.config["record_episode"] = False
        o, _ = env.reset()
        for i in range(1, 100000 if vis else 2000):
            # if i % 5 ==0:
            for old_id, new_id in env.engine.replay_manager.record_name_to_current_name.items():
                obj = env.engine.replay_manager.spawned_objects[new_id]
                pos = obj.position
                heading = obj.heading_theta
                record_pos = env.engine.replay_manager.current_frame.step_info[old_id]["position"]
                record_heading = env.engine.replay_manager.current_frame.step_info[old_id]["heading_theta"]
                assert np.isclose(np.array([pos[0], pos[1], obj.get_z()]), np.array(record_pos)).all()
                assert abs(wrap_to_pi(heading - record_heading)) < 1e-2
            # print("Replay MARL step: {}".format(i))
            o, r, tm, tc, info = env.step({agent_id: [0, 0.1] for agent_id in env.agents.keys()})
            if vis:
                env.render()
            if tm["__all__"]:
                break
    finally:
        env.close()


if __name__ == "__main__":
    test_save_episode(vis=False)
    test_save_episode_marl(vis=False)
