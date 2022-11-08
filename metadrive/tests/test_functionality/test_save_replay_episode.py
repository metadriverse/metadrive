import pickle

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
            "environment_num": 1,
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
    try:
        o = env.reset()
        for i in range(1, 100000 if vis else 2000):
            o, r, d, info = env.step([0, 1])
            if vis:
                env.render(mode="top_down", road_color=(35, 35, 35))
            if d:
                epi_info = env.engine.dump_episode("test_dump.pkl" if test_dump else None)
                break
        f = open("test_dump.pkl", "rb+")
        env.config["replay_episode"] = pickle.load(f)
        env.config["record_episode"] = False
        o = env.reset()
        for i in range(1, 100000 if vis else 2000):
            o, r, d, info = env.step([0, 1])
            if vis:
                env.render(mode="top_down", )
            if info.get("replay_done", False):
                break
    finally:
        env.close()


if __name__ == "__main__":
    test_save_episode(vis=True)
