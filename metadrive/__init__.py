import os

import metadrive.register
from metadrive.envs import MetaDriveEnv, TopDownMetaDrive, TopDownSingleFrameMetaDriveEnv, TopDownMetaDriveEnvV2, \
    SafeMetaDriveEnv, MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, MultiAgentParkingLotEnv, \
    MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentMetaDrive
from metadrive.utils.registry import get_metadrive_class

MetaDrive_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
# metadrive.register(
#      id='racing-v0',
#      entry_point='metadrive.envs.racing_env:RacingEnv')
# env = gym.make('racing-v0')
