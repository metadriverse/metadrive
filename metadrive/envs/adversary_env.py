from metadrive.envs.metadrive_env import MetaDriveEnv
import argparse
import cv2
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE
from metadrive.policy.idm_policy import IDMPolicy
from typing import Union
from metadrive.utils import Config
from metadrive.envs.intersection_env import IntersectionEnv

class AdversaryEnv(Inter):
    # TODO: Implement testing environment using the adversary manager
    def __init__(self, config: Union[dict, None] = None):
        super(AdversaryEnv, self).__init__(config)

    @staticmethod
    def default_config() -> Config:
        AdversaryConfig = {"num_adversary_vehicles": 0, "traffic_mode":"adversary",}
        assert isinstance(AdversaryConfig["num_adversary_vehicles"], int) and AdversaryConfig["num_adversary_vehicles"] >=0
        return MetaDriveEnv.default_config().update(AdversaryConfig, allow_add_new_key=True)

    def setup_engine(self):
        super(AdversaryEnv, self).setup_engine()
        from metadrive.manager.adversary_traffic_manager import PGAdversaryVehicleManager
        self.engine.update_manager("traffic_manager", PGAdversaryVehicleManager())






if __name__ == '__main__':
    # running the adversary environment directly in here; rightnow the policy is IDMPolicy
    config = dict(
        use_render=False,
        manual_control=True,
        # debug=True,
        # debug_static_world=True,
        agent_policy=IDMPolicy,
        num_adversary_vehicles=2,
        num_scenarios=20,
        traffic_mode="adversary",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(rgb_camera=(RGBCamera, 400, 300)),
                interface_panel=["rgb_camera", "dashboard"],
                num_adversary_vehicles=2,
            )
        )
    env = AdversaryEnv(config)
    try:
        o, _ = env.reset(seed=1)
        print(HELP_MESSAGE)
        env.agent.expert_takeover = True
        if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            print("The observation is an numpy array with shape: ", o.shape)
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            env.render(mode="top_down")
            # env.render(
            #     text={
            #         "Auto-Drive (Switch mode: T)": "on" if env.current_track_agent.expert_takeover else "off",
            #         "Current Observation": args.observation,
            #         "Keyboard Control": "W,A,S,D",
            #     }
            # )
            if args.observation == "rgb_camera":
                cv2.imshow('RGB Image in Observation', o["image"][..., -1])
                cv2.waitKey(1)
            if (tm or tc) and info["arrive_dest"]:
                env.reset(env.current_seed + 1)
                env.current_track_agent.expert_takeover = True
    finally:
        env.close()

