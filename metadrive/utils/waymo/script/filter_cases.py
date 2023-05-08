import os
import signal
import sys

from tqdm import tqdm

from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.idm_policy import WaymoIDMPolicy

try:
    from metadrive.utils.waymo.waymo_type import WaymoAgentType
    from metadrive.utils.waymo.waymo_type import WaymoRoadLineType, WaymoRoadEdgeType
finally:
    pass

import logging

logger = logging.getLogger(__name__)

logger.warning(
    "This converters will de deprecated. "
    "Use tools in ScenarioNet instead: https://github.com/metadriverse/ScenarioNet."
)


def handler(signum, frame):
    raise Exception("end of time")


if __name__ == "__main__":
    scenario_data_path = sys.argv[1]
    start = int(sys.argv[2])
    processed_data_path = scenario_data_path + "_filtered"
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)
    if not os.path.exists(scenario_data_path) or not os.path.exists(processed_data_path):
        raise ValueError("Path Not exist")
    num_scenarios = len(os.listdir(scenario_data_path))
    max_step = 1500
    min_step = 50

    env = WaymoEnv(
        {
            "use_render": False,
            "agent_policy": WaymoIDMPolicy,
            "data_directory": scenario_data_path,
            "start_scenario_index": start * 1000,
            "num_scenarios": num_scenarios,
            "store_map": False,
            # "manual_control": True,
            # "debug":True,
            "no_traffic": True,
            "horizon": 1500,
        }
    )
    try:
        env.reset()
    except:
        pass
    finally:
        pass
    for i in tqdm(range(num_scenarios)):
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(10)
            env.reset(force_seed=i)
            while True:
                o, r, d, info = env.step([0, 0])
                if d or env.episode_step > max_step:
                    if info["arrive_dest"] and env.episode_step > min_step:
                        os.rename(
                            os.path.join(scenario_data_path, "{}.pkl".format(i + start * 1000)),
                            os.path.join(processed_data_path, "{}.pkl".format(i + start * 1000))
                        )
                    break
        except:
            # print("\n No Route or Timeout, Fail, Seed: {}".format(i))
            pass
