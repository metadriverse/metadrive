import signal
import os
import sys

from metadrive.envs.waymo_env import WaymoEnv
from metadrive.policy.idm_policy import WaymoIDMPolicy
from tqdm import tqdm

try:
    from metadrive.utils.waymo_utils.process_scenario_20s import AgentType
    from metadrive.utils.waymo_utils.process_scenario_20s import RoadEdgeType
    from metadrive.utils.waymo_utils.process_scenario_20s import RoadLineType
finally:
    pass


def handler(signum, frame):
    raise Exception("end of time")


if __name__ == "__main__":
    case_data_path = sys.argv[1]
    processed_data_path = sys.argv[2]
    pre_fix = sys.argv[3]
    if not os.path.exists(case_data_path) or not os.path.exists(processed_data_path):
        raise ValueError("Path Not exist")
    case_num=len([name for name in os.listdir(case_data_path) if os.path.isfile(name)])
    max_step = 1000
    min_step = 100

    env = WaymoEnv(
        {
            "use_render": False,
            "agent_policy": WaymoIDMPolicy,
            "waymo_data_directory": AssetLoader.file_path(case_data_path, return_raw_style=False),
            "case_num": case_num,
            # "manual_control": True,
            # "debug":True,
            "horizon": 1000,
        }
    )
    success = []
    for i in tqdm(range(case_num)):
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(10)
            env.reset(force_seed=i)
            while True:
                o, r, d, info = env.step([0, 0])
                c_lane = env.vehicle.lane
                long, lat, = c_lane.local_coordinates(env.vehicle.position)
                if env.config["use_render"]:
                    env.render(
                        text={
                            "routing_lane_idx": env.engine._object_policies[env.vehicle.id].routing_target_lane.index,
                            "lane_index": env.vehicle.lane_index,
                            "current_ckpt_index": env.vehicle.navigation.current_checkpoint_lane_index,
                            "next_ckpt_index": env.vehicle.navigation.next_checkpoint_lane_index,
                            "ckpts": env.vehicle.navigation.checkpoints,
                            "lane_heading": c_lane.heading_theta_at(long),
                            "long": long,
                            "lat": lat,
                            "v_heading": env.vehicle.heading_theta
                        }
                    )

                if d or env.episode_steps > max_step:
                    if info["arrive_dest"] and env.episode_steps > min_step:
                        success.append(i)
                    break
        except:
            # print("\n No Route or Timeout, Fail, Seed: {}".format(i))
            pass
