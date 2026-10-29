import os

from metadrive.envs import (
    MetaDriveEnv, SafeMetaDriveEnv, ScenarioEnv, BaseEnv, MultiAgentMetaDrive, MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv, MultiAgentIntersectionEnv, MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv,
    MultiAgentTinyInter, VaryingDynamicsEnv
)
from metadrive.utils.registry import get_metadrive_class

MetaDrive_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
