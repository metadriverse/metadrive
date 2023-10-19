import argparse
from metadrive.envs.racing_env import RacingEnv
import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE

if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        # num_agents=2,
        use_render=False,
        manual_control=True,
        traffic_density=0.1,
        num_scenarios=10000,
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(show_lidar=False, show_navi_mark=False),
        map_config={"config": "CCC", "type": "block_sequence"},
        start_seed=10,
    )
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    env = RacingEnv(config)
    try:
        o, _ = env.reset(seed=21)
        print(HELP_MESSAGE)
        env.vehicle.expert_takeover = True
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            env.render(mode="topdown")
            if (tm or tc) and info["arrive_dest"]:
                env.reset(env.current_seed + 1)
                env.current_track_vehicle.expert_takeover = True
    except Exception as e:
        raise e
    finally:
        env.close()
