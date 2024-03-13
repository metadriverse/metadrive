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
# from metadrive.envs.intersection_env import IntersectionEnv # TODO: convert this to a single agent environment
# from metadrive.policy.adv_policy import AdvPolicy



class AdversaryEnv(MetaDriveEnv): # TODO: Implement single agent intersection environment using the adversary manager
    # TODO: Implement testing environment using the adversary manager
    def __init__(self, config: Union[dict, None] = None):
        super(AdversaryEnv, self).__init__(config)

    @staticmethod
    def default_config() -> Config:
        AdversaryConfig = {"num_adversary_vehicles": 0, "traffic_mode":"adversary", "need_inverse_traffic":True}
        assert isinstance(AdversaryConfig["num_adversary_vehicles"], int) and AdversaryConfig["num_adversary_vehicles"] >=0
        return MetaDriveEnv.default_config().update(AdversaryConfig, allow_add_new_key=True)

    def setup_engine(self):
        super(AdversaryEnv, self).setup_engine()
        from metadrive.manager.adversary_traffic_manager import PGAdversaryVehicleManager
        self.engine.update_manager("traffic_manager", PGAdversaryVehicleManager())






if __name__ == '__main__':
    # running the adversary environment directly in here; rightnow the policy is IDMPolicy
    default_config = AdversaryEnv.default_config()
    env_config = dict(crash_vehicle_penalty=-1.0,
                      success_reward=10.0,

                      # traffic_vehicle_config=dict(
                      #
                      # ),
                      num_adversary_vehicles=5,
                      agent_policy=IDMPolicy,
                      )

    default_config.update(env_config)



    env = AdversaryEnv(env_config)
    try:
        o, _ = env.reset()
        print(HELP_MESSAGE)
        env.agent.expert_takeover = True


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

            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_agent.expert_takeover = True
    finally:
        env.close()
    #
