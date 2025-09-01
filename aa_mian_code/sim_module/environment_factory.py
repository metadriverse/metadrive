"""
环境工厂模块

该模块专门负责MetaDrive环境的创建、配置和管理。
从原始PPOCheckpointSimulator中提取环境创建相关功能。
"""

import argparse
from typing import Dict, Optional
from pathlib import Path
import sys

# 添加metadrive到路径
current_dir = Path(__file__).parent.absolute()
metadrive_root = current_dir.parent.parent.parent
sys.path.insert(0, str(metadrive_root))

# 导入环境类
from metadrive.envs.metadrive_env import MetaDriveEnv

# 导入速度控制环境（如果存在）
try:
    from ppo_expert_reproduction_with_cog import SpeedControlMetaDriveEnv
    SPEED_CONTROL_AVAILABLE = True
    print("✅ SpeedControlMetaDriveEnv导入成功")
except ImportError:
    SPEED_CONTROL_AVAILABLE = False
    print("⚠️ SpeedControlMetaDriveEnv导入失败，将使用标准MetaDriveEnv")


class EnvironmentFactory:
    """MetaDrive环境工厂
    
    专门负责：
    1. 环境配置的管理和生成
    2. 标准MetaDriveEnv和SpeedControlMetaDriveEnv的创建
    3. 环境参数的动态配置
    4. 速度控制子模块的配置
    """
    
    def __init__(self, config: Optional[Dict] = None, args: Optional[argparse.Namespace] = None):
        """
        初始化环境工厂
        
        Args:
            config: 环境配置字典
            args: 命令行参数
        """
        self.config = config if config is not None else self._get_default_config()
        self.args = args
        
        print(f"🏭 环境工厂初始化完成")
        print(f"   基础地图: {self.config.get('map', 'SSSSSSS')}")
        print(f"   交通密度: {self.config.get('traffic_density', 0.08)}")
        print(f"   最大步数: {self.config.get('horizon', 1000)}")
    
    def create_environment(self, render: bool = True, scenario_seed: Optional[int] = None) -> MetaDriveEnv:
        """
        创建MetaDrive仿真环境
        
        Args:
            render: 是否渲染
            scenario_seed: 场景种子
            
        Returns:
            创建的MetaDriveEnv实例
        """
        env_config = self.config.copy()
        env_config["use_render"] = render
        
        # 设置种子以确保背景车初始状态可重复
        if scenario_seed is not None:
            env_config["start_seed"] = scenario_seed
            # 对于第一个场景，额外确保随机性控制
            if scenario_seed == 8888:
                env_config["random_traffic"] = False  # 确保第一个场景交通状态固定
                print(f"🔒 固定第一个场景种子: {scenario_seed} (背景车状态将保持一致)")
        
        # 设置观察配置 - 避免与认知感知模块的双重噪声
        env_config["vehicle_config"]["lidar"] = {
            "num_lasers": 240,
            "distance": 50,
            "num_others": 4,
            # 关键：必须设置为0以避免与认知感知模块的噪声叠加
            "gaussian_noise": 0.0,
            "dropout_prob": 0.0
        }
        
        # 根据配置选择环境类型
        use_speed_control = self._should_use_speed_control(env_config)
        
        if use_speed_control and SPEED_CONTROL_AVAILABLE:
            # 使用速度控制环境
            env = self._create_speed_control_env(env_config)
        else:
            # 使用标准MetaDrive环境
            env = self._create_standard_env(env_config, use_speed_control)
        
        return env
    
    def _should_use_speed_control(self, env_config: Dict) -> bool:
        """判断是否应该使用速度控制环境"""
        # 检查配置中的速度控制标志
        use_speed_control = env_config.get("use_speed_control_reward", False)
        
        # 检查命令行参数中的速度控制标志
        if self.args and getattr(self.args, 'use_speed_control_reward', False):
            use_speed_control = True
            print(f"🔧 从命令行参数启用速度控制奖励")
        
        return use_speed_control
    
    def _create_speed_control_env(self, env_config: Dict) -> 'SpeedControlMetaDriveEnv':
        """创建速度控制环境"""
        print("🚀 创建SpeedControlMetaDriveEnv环境")
        
        # 创建包含速度控制参数的配置（SpeedControlMetaDriveEnv会自动清理）
        speed_control_config = env_config.copy()
        
        # 确保use_speed_control_reward被设置
        speed_control_config["use_speed_control_reward"] = True
        
        # 添加速度控制参数到配置中
        if self.args:
            speed_control_config.update({
                "speed_control_k": getattr(self.args, 'speed_control_k', 1.0),
                "speed_control_kappa": getattr(self.args, 'speed_control_kappa', 0.5),
                "speed_control_mu": getattr(self.args, 'speed_control_mu', 0.3),
                "speed_control_nu": getattr(self.args, 'speed_control_nu', 0.2),
                "speed_control_v_tolerance": getattr(self.args, 'speed_control_v_tolerance', 1.0),
                "speed_control_v_ref": getattr(self.args, 'speed_control_v_ref', 15.0)
            })
        
        # 关闭原始速度奖励，避免重复
        speed_control_config["speed_reward"] = 0.0
        
        print(f"   🔧 配置更新:")
        print(f"      use_speed_control_reward: {speed_control_config['use_speed_control_reward']}")
        print(f"      speed_reward: {speed_control_config['speed_reward']}")
        print(f"      speed_control_k: {speed_control_config.get('speed_control_k', 'N/A')}")
        print(f"      speed_control_v_ref: {speed_control_config.get('speed_control_v_ref', 'N/A')}")
        
        # 创建SpeedControlMetaDriveEnv实例（它会自动清理速度控制参数）
        env = SpeedControlMetaDriveEnv(speed_control_config)
        
        # 设置速度控制子模块启用状态（在环境创建后设置）
        self._configure_speed_control_submodules(env)
        
        return env
    
    def _create_standard_env(self, env_config: Dict, use_speed_control: bool) -> MetaDriveEnv:
        """创建标准MetaDrive环境"""
        if use_speed_control and not SPEED_CONTROL_AVAILABLE:
            print("⚠️ 速度控制环境不可用，使用标准MetaDriveEnv")
        else:
            print("🔧 使用标准MetaDriveEnv环境")
        
        env = MetaDriveEnv(env_config)
        return env
    
    def _configure_speed_control_submodules(self, env):
        """配置速度控制子模块"""
        if not self.args:
            print("   ✅ 使用默认子模块配置")
            print(f"      enable_tracking: {getattr(env, 'enable_tracking', True)}")
            print(f"      enable_soft_wall: {getattr(env, 'enable_soft_wall', True)}")
            print(f"      enable_behavior_guidance: {getattr(env, 'enable_behavior_guidance', True)}")
            return
        
        tracking_enabled = getattr(self.args, 'speed_control_enable_tracking', False)
        soft_wall_enabled = getattr(self.args, 'speed_control_enable_soft_wall', False)
        behavior_enabled = getattr(self.args, 'speed_control_enable_behavior_guidance', False)
        
        # 直接设置环境属性
        env.enable_tracking = tracking_enabled
        env.enable_soft_wall = soft_wall_enabled
        env.enable_behavior_guidance = behavior_enabled
        
        print(f"   🔧 子模块开关设置:")
        print(f"      enable_tracking: {tracking_enabled}")
        print(f"      enable_soft_wall: {soft_wall_enabled}")
        print(f"      enable_behavior_guidance: {behavior_enabled}")
        
        # 验证设置是否成功
        print(f"   🔍 验证子模块开关设置:")
        print(f"      env.enable_tracking: {getattr(env, 'enable_tracking', 'N/A')}")
        print(f"      env.enable_soft_wall: {getattr(env, 'enable_soft_wall', 'N/A')}")
        print(f"      env.enable_behavior_guidance: {getattr(env, 'enable_behavior_guidance', 'N/A')}")
    
    def _get_default_config(self) -> Dict:
        """获取默认环境配置 - 与训练时保持一致"""
        return {
            # 基础环境配置 - 与训练时一致
            "num_scenarios": 1,
            "traffic_density": 0.08,
            "random_traffic": True,
            "random_agent_model": False,
            "horizon": 1000,
            "map": "SSSSSSS",  # 使用动态数量的直线段
            "start_seed": 8888,
            
            # 奖励配置 - 与训练时完全一致
            "success_reward": 100.0,
            "driving_reward": 0.4,
            "speed_reward": 0,
            "use_lateral_reward": True,
            
            # 惩罚配置 - 与训练时一致
            "out_of_road_penalty": 8.0,
            "crash_vehicle_penalty": 8.0,
            "crash_object_penalty": 8.0,
            "crash_sidewalk_penalty": 8.0,
            
            # 终止条件 - 与训练时一致
            "out_of_road_done": True,
            "crash_vehicle_done": True,
            "crash_object_done": True,
            "on_continuous_line_done": False,
            "on_broken_line_done": False,
            
            # 车辆配置 - 与训练时完全一致 (主车配置)
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 240,
                    "distance": 50,
                    "num_others": 4,
                    "gaussian_noise": 0.0,
                    "dropout_prob": 0.0
                },
                "side_detector": {
                    "num_lasers": 0,
                    "distance": 50,
                    "gaussian_noise": 0.0,
                    "dropout_prob": 0.0
                },
                "lane_line_detector": {
                    "num_lasers": 0,
                    "distance": 20,
                    "gaussian_noise": 0.0,
                    "dropout_prob": 0.0
                }
            },
            
            # 背景车专用配置
            "traffic_vehicle_config": {
                "show_navi_mark": False,
                "show_dest_mark": False,
                "enable_reverse": False,
                "show_lidar": False,
                "show_lane_line_detector": False,
                "show_side_detector": False,
            },
            
            # 渲染配置
            "use_render": True
        }
    
    def update_config(self, new_config: Dict):
        """更新环境配置"""
        self.config.update(new_config)
        print(f"🔧 环境配置已更新: {list(new_config.keys())}")
    
    def get_config(self) -> Dict:
        """获取当前环境配置"""
        return self.config.copy() 