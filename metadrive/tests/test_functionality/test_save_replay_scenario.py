import pickle

from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import setup_logger


def _test_save_scenario(vis=False):
    setup_logger(True)

    save_episode = True
    vis = vis
    env = SafeMetaDriveEnv(
        {
            "accident_prob": 0.8,
            "environment_num": 1,
            "traffic_density": 0.1,
            "start_seed": 1000,
            "manual_control": True,
            "use_render": vis,
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
    # try:
    o = env.reset()
    for i in range(1, 100000 if vis else 2000):
        o, r, d, info = env.step([0, 1])
        # if vis:
        #     env.render(mode="top_down", road_color=(35, 35, 35))
        if d:
            epi_info = env.engine.record_manager.get_episode_metadata()
            break
    env.close()
    env = SafeMetaDriveEnv(
        {
            "accident_prob": 0.8,
            "environment_num": 1,
            "traffic_density": 0.1,
            "start_seed": 1000,
            "manual_control": True,
            "debug": True,
            "use_render": vis,
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
    env.config["replay_episode"] = epi_info
    env.config["record_episode"] = False
    env.config["only_reset_when_replay"] = True
    o = env.reset()
    for i in range(1, 100000 if vis else 2000):
        o, r, d, info = env.step([0, 1])
        # if vis:
        #     env.render(mode="top_down", )
        if info.get("replay_done", False):
            break
    # finally:
    #     env.close()


if __name__ == "__main__":
    _test_save_scenario(vis=True)
