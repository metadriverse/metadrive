from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame  # it is important to import pygame after that

from metadrive.envs import MetaDriveEnv, TopDownMetaDrive, TopDownSingleFrameMetaDriveEnv, TopDownMetaDriveEnvV2, \
    SafeMetaDriveEnv, MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, MultiAgentParkingLotEnv, \
    MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentMetaDrive, ScenarioEnv
from metadrive.utils.registry import get_metadrive_class
import os

MetaDrive_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
