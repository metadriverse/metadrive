import metadrive.register
from metadrive.envs import MetaDriveEnv, TopDownMetaDrive, TopDownSingleFrameMetaDriveEnv, TopDownMetaDriveEnvV2, \
    SafeMetaDriveEnv, MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, MultiAgentParkingLotEnv, \
    MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentMetaDrive
from metadrive.utils.registry import get_metadrive_class
import os

MetaDrive_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
