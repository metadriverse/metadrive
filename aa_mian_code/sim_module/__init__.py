"""
仿真模块包

该包包含了重构后的PPOCheckpointSimulator的所有子模块。
提供模块化、可维护的PPO检查点仿真功能。
"""

# 导出主要的类供外部使用
from .simulation_runner import SimulationRunner
from .checkpoint_loader import CheckpointLoader
from .lane_change_tracker import LaneChangeTracker
from .environment_factory import EnvironmentFactory
from .cognitive_module_manager import CognitiveModuleManager
from .action_processor import ActionProcessor
from .episode_statistics import EpisodeStatistics
from .visualization_manager import VisualizationManager

# 保持向后兼容性，仍然导出原有的类
from .checkpointsimulator import PPOCheckpointSimulator

__all__ = [
    'SimulationRunner',
    'CheckpointLoader', 
    'LaneChangeTracker',
    'EnvironmentFactory',
    'CognitiveModuleManager',
    'ActionProcessor',
    'EpisodeStatistics',
    'VisualizationManager',
    'PPOCheckpointSimulator'  # 向后兼容
]

__version__ = "2.0.0" 