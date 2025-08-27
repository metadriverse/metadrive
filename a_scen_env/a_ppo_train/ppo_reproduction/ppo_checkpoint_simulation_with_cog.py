#!/usr/bin/env python3
"""
PPO检查点仿真控制器
基于训练好的PPO检查点在MetaDrive仿真环境中控制主车行为
支持加载指定检查点、可视化仿真、性能评估等功能
集成认知模块：认知偏差、认知延迟、认知感知

python ppo_checkpoint_simulation_with_cog.py \
  --checkpoint /home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/runs/ppo_expert_reproduction_20250820_154108/checkpoints/checkpoint_2270.pt \
  --use_cognitive_modules \
  --use_cognitive_bias \
  --use_cognitive_delay \
  --use_cognitive_perception \
  --enable_cognitive_viz \
  --episodes 1 \
  --max_steps 500 \
  --no_render
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from collections import defaultdict, deque

# 添加metadrive到路径
current_dir = Path(__file__).parent.absolute()
metadrive_root = current_dir.parent.parent.parent
sys.path.insert(0, str(metadrive_root))

# 添加认知模块路径
cognitive_module_path = current_dir.parent.parent / "cognitive_module"
sys.path.insert(0, str(cognitive_module_path))

# 导入认知模块
from cognitive_module.cognitive_bias_module import CognitiveBiasModule
from cognitive_module.cognitive_delay_module import CognitiveDelayModule
from cognitive_module.cognitive_perception_module import CognitivePerceptionModule

# 导入环境类
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.state_obs import LidarStateObservation

# 导入速度控制环境（如果存在）
try:
    from ppo_expert_reproduction_with_cog import SpeedControlMetaDriveEnv
    SPEED_CONTROL_AVAILABLE = True
    print("✅ SpeedControlMetaDriveEnv导入成功")
except ImportError:
    SPEED_CONTROL_AVAILABLE = False
    print("⚠️ SpeedControlMetaDriveEnv导入失败，将使用标准MetaDriveEnv")


class PPONetwork(nn.Module):
    """PPO网络结构 - 支持动态观测维度"""
    
    def __init__(self, obs_dim: int = 275, action_dim: int = 2, hidden_dim: int = 256, use_cognitive_modules: bool = False):
        super(PPONetwork, self).__init__()
        
        # 根据认知模块启用状态确定观测维度
        if use_cognitive_modules:
            # 认知模块启用时，使用扩展观测维度
            self.obs_dim = obs_dim
            print(f"🧠 认知模块启用，使用观测维度: {self.obs_dim}")
        else:
            # 认知模块不启用时，使用标准275维度
            self.obs_dim = 275
            print(f"🔧 认知模块禁用，使用标准观测维度: {self.obs_dim}")
        
        # Actor网络
        self.actor_fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim * 2)  # mean + log_std
        
        # Critic网络
        self.critic_fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)
        
        # 激活函数
        self.tanh = nn.Tanh()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs):
        """前向传播"""
        # 检查观测维度匹配
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"观测维度不匹配: 期望{self.obs_dim}, 实际{obs.shape[-1]}")
        
        # Actor前向
        x_actor = self.tanh(self.actor_fc1(obs))
        x_actor = self.tanh(self.actor_fc2(x_actor))
        action_logits = self.actor_out(x_actor)
        
        # Critic前向
        x_critic = self.tanh(self.critic_fc1(obs))
        x_critic = self.tanh(self.critic_fc2(x_critic))
        value = self.critic_out(x_critic)
        
        return action_logits, value
    
    def get_action_and_value(self, obs, action=None, deterministic=False):
        """获取动作和价值"""
        action_logits, value = self.forward(obs)
        
        # 分离均值和标准差
        action_mean, action_log_std = torch.chunk(action_logits, 2, dim=-1)
        action_std = torch.exp(action_log_std)
        
        # 创建分布
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)


class PPOCheckpointSimulator:
    """PPO检查点仿真控制器"""
    
    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None, device: str = "auto", args: Optional[argparse.Namespace] = None):
        """
        初始化仿真控制器
        
        Args:
            checkpoint_path: 检查点文件路径
            config_path: 配置文件路径（可选）
            device: 计算设备
            args: 命令行参数（用于变道惩罚配置和认知模块）
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.args = args  # 🔧 新增：保存命令行参数引用
        
        # === 认知模块初始化 ===
        self.use_cognitive_modules = args and getattr(args, 'use_cognitive_modules', False)
        self.cognitive_bias_module = None
        self.cognitive_delay_module = None  
        self.cognitive_perception_module = None
        
        if self.use_cognitive_modules:
            print("🧠 初始化认知模块...")
            
            # 先初始化认知感知模块（因为认知偏差模块需要引用它）
            if args and getattr(args, 'use_cognitive_perception', False):
                perception_config = {
                    'sigma0': getattr(args, 'perception_noise_std', 0.01) * 10,  # 转换为米制噪声
                    'k': 0.02,
                    'p_miss0': 0.01,
                    'far_distance': 50.0,
                    'p_false': 0.0001,
                    'use_ar1': True,
                    'rho': 0.8,
                    'use_kf': True,
                    'kf_dt': 0.1
                }
                self.cognitive_perception_module = CognitivePerceptionModule(noise_config=perception_config)
                
                # 启用雷达束可视化（如果指定）
                if getattr(args, 'enable_radar_beam_viz', False):
                    self.cognitive_perception_module.enable_radar_visualization(True)
                
                print(f"   ✅ 认知感知模块已启用")
            
            # 初始化认知偏差模块（传入认知感知模块引用）
            if args and getattr(args, 'use_cognitive_bias', False):
                bias_config = {
                    'inverse_tta_coef': getattr(args, 'bias_inverse_tta_coef', 1.5),
                    'tta_threshold': getattr(args, 'bias_tta_threshold', 0.1),
                    'visual_detection_distance': getattr(args, 'bias_visual_distance', 50.0),
                    'verbose': True
                }
                # 🔧 新增：传入认知感知模块引用
                self.cognitive_bias_module = CognitiveBiasModule(
                    bias_config=bias_config,
                    cognitive_perception_module=self.cognitive_perception_module
                )
                print(f"   ✅ 认知偏差模块已启用")
                if self.cognitive_perception_module:
                    print(f"      🔗 已连接到认知感知模块")
            
            # 初始化认知延迟模块
            if args and getattr(args, 'use_cognitive_delay', False):
                self.cognitive_delay_module = CognitiveDelayModule(
                    delay_steps=int(getattr(args, 'delay_steps', 2)),  # 确保是整数类型
                    enable_smoothing=False,
                    smoothing_factor=0.3,
                    enable_visualization=True
                )
                print(f"   ✅ 认知延迟模块已启用 (延迟{getattr(args, 'delay_steps', 2)}步)")
        
        # 加载检查点
        self._load_checkpoint()
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # 使用默认配置
            self.config = self._get_default_config()
        
        # 🔧 新增：变道冷却时间跟踪
        self._last_lane_change_step = {}  # 存储最后变道的时间步
        self._last_lane_index = {}        # 存储每个agent的上一个车道索引
        # 动态计算冷却时间步数，基于环境实际频率
        self._lane_change_cooldown_steps = None  # 将在环境创建后动态设置
        
        print(f"✅ PPO检查点仿真控制器初始化完成")
        print(f"📁 检查点: {os.path.basename(checkpoint_path)}")
        print(f"🔧 设备: {self.device}")
        print(f"🎯 训练迭代: {self.checkpoint.get('iteration', 'Unknown')}")
        print(f"🚀 全局步数: {self.checkpoint.get('global_step', 'Unknown')}")
        print(f"🧠 认知模块: {'启用' if self.use_cognitive_modules else '禁用'}")
        
        # === 认知可视化数据收集 ===
        self.enable_cognitive_visualization = args and getattr(args, 'enable_cognitive_viz', False)
        if self.enable_cognitive_visualization and self.use_cognitive_modules:
            self.cognitive_viz_data = {
                'timestamps': [],
                'bias_strength': [],
                'bias_applied': [],
                'delay_steps': [],
                'delay_applied': [],
                'perception_noise': [],
                'perception_applied': [],
                'original_rewards': [],
                'modified_rewards': [],
                'original_actions': [],
                'delayed_actions': [],
                'original_observations': [],
                'noisy_observations': [],
                'step_count': []
            }
            print(f"🎨 认知可视化: 已启用")
        else:
            self.cognitive_viz_data = None
        
        # === 速度控制奖励数据收集 ===
        self.enable_speed_control_visualization = args and getattr(args, 'use_speed_control_reward', False)
        if self.enable_speed_control_visualization:
            self.speed_control_viz_data = {
                'step_count': [],
                'speed_control_total': [],
                'speed_control_tracking': [],
                'speed_control_soft_wall': [],
                'speed_control_behavior_guidance': [],
                'vehicle_speeds': [],
                'speed_references': [],
                'speed_deviations': []
            }
            print(f"🚀 速度控制可视化: 已启用")
        else:
            self.speed_control_viz_data = None
    
    def _setup_lane_change_cooldown(self, env):
        """🔧 新增：动态设置变道冷却时间步数"""
        try:
            if env and hasattr(env, 'config'):
                # 获取物理步长和决策重复次数
                physics_step_size = env.config.get('physics_world_step_size', 0.02)
                decision_repeat = env.config.get('decision_repeat', 5)
                
                # 计算实际有效频率
                effective_time_step = physics_step_size * decision_repeat
                effective_frequency = 1.0 / effective_time_step
                
                # 计算冷却时间步数
                lc_cooldown_s = getattr(self, 'args', None) and getattr(self.args, 'lc_cooldown_s', 4.0) or 4.0
                self._lane_change_cooldown_steps = int(lc_cooldown_s * effective_frequency)
                
                print(f"🔧 变道冷却时间设置:")
                print(f"   物理步长: {physics_step_size:.3f}s")
                print(f"   决策重复: {decision_repeat}")
                print(f"   有效频率: {effective_frequency:.1f}Hz")
                print(f"   冷却时间: {lc_cooldown_s}s → {self._lane_change_cooldown_steps}步")
            else:
                # 如果无法获取环境配置，使用默认值
                lc_cooldown_s = getattr(self, 'args', None) and getattr(self.args, 'lc_cooldown_s', 4.0) or 4.0
                self._lane_change_cooldown_steps = int(lc_cooldown_s * 10)
                print(f"⚠️  无法获取环境配置，使用默认10Hz假设")
                print(f"   冷却时间: {lc_cooldown_s}s → {self._lane_change_cooldown_steps}步")
                
        except Exception as e:
            # 异常处理，使用默认值
            lc_cooldown_s = getattr(self, 'args', None) and getattr(self.args, 'lc_cooldown_s', 4.0) or 4.0
            self._lane_change_cooldown_steps = int(lc_cooldown_s * 10)
            print(f"⚠️  设置冷却时间失败: {e}，使用默认10Hz假设")
            print(f"   冷却时间: {lc_cooldown_s}s → {self._lane_change_cooldown_steps}步")
    
    def _load_checkpoint(self):
        """加载检查点"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {self.checkpoint_path}")
        
        # 加载检查点数据
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # 检测检查点中的观测维度
        checkpoint_obs_dim = None
        if 'network_state_dict' in self.checkpoint:
            # 从第一个线性层的权重推断观测维度
            first_layer_key = None
            for key in self.checkpoint['network_state_dict'].keys():
                if 'actor_fc1.weight' in key:
                    first_layer_key = key
                    break
            
            if first_layer_key:
                checkpoint_obs_dim = self.checkpoint['network_state_dict'][first_layer_key].shape[1]
                print(f"🔍 检查点检测到观测维度: {checkpoint_obs_dim}")
        
        # 根据认知模块启用状态确定使用的观测维度
        if self.use_cognitive_modules:
            # 认知模块启用时，使用检查点的原始维度
            obs_dim = checkpoint_obs_dim if checkpoint_obs_dim else 275
            print(f"🧠 认知模块启用，使用观测维度: {obs_dim}")
        else:
            # 认知模块不启用时，强制使用275维度
            obs_dim = 275
            print(f"🔧 认知模块禁用，强制使用标准观测维度: {obs_dim}")
        
        # 创建并加载网络
        self.network = PPONetwork(obs_dim=obs_dim, use_cognitive_modules=self.use_cognitive_modules).to(self.device)
        
        # 如果观测维度不匹配，需要调整网络权重
        if checkpoint_obs_dim and checkpoint_obs_dim != obs_dim:
            print(f"⚠️ 观测维度不匹配，检查点: {checkpoint_obs_dim}, 目标: {obs_dim}")
            if obs_dim == 275 and checkpoint_obs_dim > 275:
                # 从扩展维度截取到275维度
                print("🔧 从扩展维度截取到275维度")
                self._adjust_network_weights(checkpoint_obs_dim, obs_dim)
            else:
                print("⚠️ 无法自动调整权重，将尝试直接加载")
        
        # 尝试加载权重
        try:
            self.network.load_state_dict(self.checkpoint['network_state_dict'])
            print("✅ 权重加载成功")
        except Exception as e:
            print(f"⚠️ 权重加载失败: {e}")
            print("🔧 将使用随机初始化的权重")
        
        self.network.eval()
        print(f"📥 检查点加载完成: {os.path.basename(self.checkpoint_path)}")
    
    def _adjust_network_weights(self, from_dim: int, to_dim: int):
        """调整网络权重以适应不同的观测维度"""
        if from_dim <= to_dim:
            return
        
        print(f"🔧 调整网络权重: {from_dim} → {to_dim}")
        
        # 获取检查点权重
        checkpoint_weights = self.checkpoint['network_state_dict']
        
        # 调整第一层权重（截取前to_dim个输入）
        for layer_name in ['actor_fc1.weight', 'critic_fc1.weight']:
            if layer_name in checkpoint_weights:
                old_weight = checkpoint_weights[layer_name]
                new_weight = old_weight[:, :to_dim]  # 截取前to_dim个输入维度
                checkpoint_weights[layer_name] = new_weight
                print(f"   {layer_name}: {old_weight.shape} → {new_weight.shape}")
        
        # 更新检查点
        self.checkpoint['network_state_dict'] = checkpoint_weights
    
    def _get_default_config(self):
        """获取默认环境配置 - 与训练时保持一致"""
        return {
            # 基础环境配置 - 与训练时一致
            "num_scenarios": 1,
            "traffic_density": 0.12,
            "random_traffic": True,
            "random_agent_model": False,
            "horizon": 1000,
            "map": "SSSSSSS",
            # "map": 301,
            "start_seed": 8888,
                 # 使用动态数量的直线段


            
            # 奖励配置 - 与训练时完全一致
            "success_reward": 20.0,
            "driving_reward": 1.0,
            "speed_reward": 0,
            "use_lateral_reward": True,
            
            # 速度控制奖励配置（不在环境配置中，由SpeedControlMetaDriveEnv内部处理）
            
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
            
            # 🚗 背景车专用配置
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
    
    def create_environment(self, render: bool = True, scenario_seed: Optional[int] = None) -> MetaDriveEnv:
        """创建MetaDrive仿真环境"""
        env_config = self.config.copy()
        env_config["use_render"] = render
        
        # 设置种子以确保背景车初始状态可重复
        if scenario_seed is not None:
            env_config["start_seed"] = scenario_seed
            # 对于第一个场景，额外确保随机性控制
            if scenario_seed == 8888:
                env_config["random_traffic"] = False  # 确保第一个场景交通状态固定
                print(f"🔒 固定第一个场景种子: {scenario_seed} (背景车状态将保持一致)")
        
        # 设置观察配置 - ⚠️ 重要：避免与认知感知模块的双重噪声
        env_config["vehicle_config"]["lidar"] = {
            "num_lasers": 240,
            "distance": 50,
            "num_others": 4,
            # 🚫 关键：必须设置为0以避免与认知感知模块的噪声叠加
            "gaussian_noise": 0.0,
            "dropout_prob": 0.0
        }
        
        # 根据配置选择环境类型
        use_speed_control = env_config.get("use_speed_control_reward", False)
        
        # 🔧 修复：检查命令行参数中的use_speed_control_reward
        if hasattr(self, 'args') and self.args and getattr(self.args, 'use_speed_control_reward', False):
            use_speed_control = True
            print(f"🔧 从命令行参数启用速度控制奖励")
        
        if use_speed_control and SPEED_CONTROL_AVAILABLE:
            # 使用速度控制环境
            print("🚀 创建SpeedControlMetaDriveEnv环境")
            
            # 创建包含速度控制参数的配置（SpeedControlMetaDriveEnv会自动清理）
            speed_control_config = env_config.copy()
            
            # 🔧 修复：确保use_speed_control_reward被设置
            speed_control_config["use_speed_control_reward"] = True
            
            # 添加速度控制参数到配置中
            if hasattr(self, 'args') and self.args:
                speed_control_config.update({
                    "speed_control_k": getattr(self.args, 'speed_control_k', 1.0),
                    "speed_control_kappa": getattr(self.args, 'speed_control_kappa', 0.5),
                    "speed_control_mu": getattr(self.args, 'speed_control_mu', 0.3),
                    "speed_control_nu": getattr(self.args, 'speed_control_nu', 0.2),
                    "speed_control_v_tolerance": getattr(self.args, 'speed_control_v_tolerance', 1.0),
                    "speed_control_v_ref": getattr(self.args, 'speed_control_v_ref', 15.0)
                })
            
            # 🔧 修复：关闭原始速度奖励，避免重复
            speed_control_config["speed_reward"] = 0.0
            
            print(f"   🔧 配置更新:")
            print(f"      use_speed_control_reward: {speed_control_config['use_speed_control_reward']}")
            print(f"      speed_reward: {speed_control_config['speed_reward']}")
            print(f"      speed_control_k: {speed_control_config.get('speed_control_k', 'N/A')}")
            print(f"      speed_control_v_ref: {speed_control_config.get('speed_control_v_ref', 'N/A')}")
            
            # 创建SpeedControlMetaDriveEnv实例（它会自动清理速度控制参数）
            env = SpeedControlMetaDriveEnv(speed_control_config)
            
            # 🔧 修复：设置速度控制子模块启用状态（在环境创建后设置）
            if hasattr(self, 'args') and self.args:
                tracking_enabled = getattr(self.args, 'speed_control_enable_tracking', False)
                soft_wall_enabled = getattr(self.args, 'speed_control_enable_soft_wall', False)
                behavior_enabled = getattr(self.args, 'speed_control_enable_behavior_guidance', False)
                
                # 🔧 修复：直接设置环境属性
                env.enable_tracking = tracking_enabled
                env.enable_soft_wall = soft_wall_enabled
                env.enable_behavior_guidance = behavior_enabled
                
                print(f"   🔧 子模块开关设置:")
                print(f"      enable_tracking: {tracking_enabled}")
                print(f"      enable_soft_wall: {soft_wall_enabled}")
                print(f"      enable_behavior_guidance: {behavior_enabled}")
                
                # 🔧 修复：验证设置是否成功
                print(f"   🔍 验证子模块开关设置:")
                print(f"      env.enable_tracking: {getattr(env, 'enable_tracking', 'N/A')}")
                print(f"      env.enable_soft_wall: {getattr(env, 'enable_soft_wall', 'N/A')}")
                print(f"      env.enable_behavior_guidance: {getattr(env, 'enable_behavior_guidance', 'N/A')}")
            else:
                print("   ✅ 使用默认子模块配置")
                print(f"      enable_tracking: {getattr(env, 'enable_tracking', True)}")
                print(f"      enable_soft_wall: {getattr(env, 'enable_soft_wall', True)}")
                print(f"      enable_behavior_guidance: {getattr(env, 'enable_behavior_guidance', True)}")
        
        else:
            # 使用标准MetaDrive环境
            if use_speed_control and not SPEED_CONTROL_AVAILABLE:
                print("⚠️ 速度控制环境不可用，使用标准MetaDriveEnv")
            else:
                print("🔧 使用标准MetaDriveEnv环境")
            
            env = MetaDriveEnv(env_config)
        
        # 🔧 新增：动态设置变道冷却时间
        if not hasattr(self, '_lane_change_cooldown_steps') or self._lane_change_cooldown_steps is None:
            self._setup_lane_change_cooldown(env)
        
        return env
    
    def get_action(self, observation: np.ndarray, deterministic: bool = True, env=None, step_count: int = 0) -> np.ndarray:
        """
        根据观察获取PPO策略动作
        
        Args:
            observation: 环境观察
            deterministic: 是否使用确定性策略
            env: 环境实例（用于认知模块处理）
            step_count: 当前步数（用于可视化）
            
        Returns:
            动作数组 [steering, acceleration]
        """
        original_obs = observation.copy() if self.enable_cognitive_visualization and self.cognitive_viz_data is not None else None
        
        # === 认知感知模块：噪声已在传感器层自动注入 ===
        # 注意：认知感知模块通过 attach_to_env() 已经在传感器层注入噪声
        # 这里的 observation 已经包含了噪声效果，无需额外处理
        processed_obs = observation
        perception_applied = bool(self.use_cognitive_modules and self.cognitive_perception_module)

        # 转换为tensor
        obs_tensor = torch.FloatTensor(processed_obs).unsqueeze(0).to(self.device)
        
        # 获取动作
        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(obs_tensor, deterministic=deterministic)
        
        # 转换回numpy
        action = action.cpu().numpy().flatten()
        
        # 限制动作范围
        action = np.clip(action, -1.0, 1.0)
        
        original_action = action.copy() if self.enable_cognitive_visualization else None
        
        # === 认知延迟模块：处理动作延迟 ===
        delay_applied = False
        if self.use_cognitive_modules and self.cognitive_delay_module:
            try:
                action = self.cognitive_delay_module.process_action(action, is_ppo_mode=True)
                delay_applied = True
            except Exception as e:
                print(f"⚠️ 认知延迟模块处理失败: {e}")
        
        # === 收集可视化数据 ===
        if self.enable_cognitive_visualization and self.cognitive_viz_data is not None:
            # 感知模块数据 - 获取正前方雷达束的实时数据
            front_beam_data = {'original_distance': 0.0, 'noisy_distance': 0.0, 'noise_level': 0.0}
            
            if self.cognitive_perception_module and perception_applied:
                # 获取正前方雷达束的实时数据
                front_beam_data = self.cognitive_perception_module.get_front_beam_info()
                noise_level = front_beam_data.get('noise_level', 0.0)
            else:
                noise_level = 0.0

            # 延迟模块数据
            if self.cognitive_delay_module and delay_applied:
                # 尝试获取延迟信息，如果方法不存在则使用默认值
                if hasattr(self.cognitive_delay_module, 'get_delay_info'):
                    delay_info = self.cognitive_delay_module.get_delay_info()
                    current_delay = delay_info.get('current_delay', 0)
                else:
                    current_delay = self.cognitive_delay_module.delay_steps
            else:
                current_delay = 0

            
            # 记录数据
            self.cognitive_viz_data['timestamps'].append(time.time())
            self.cognitive_viz_data['step_count'].append(step_count)
            self.cognitive_viz_data['perception_noise'].append(noise_level)
            self.cognitive_viz_data['perception_applied'].append(perception_applied)
            self.cognitive_viz_data['delay_steps'].append(current_delay)
            self.cognitive_viz_data['delay_applied'].append(delay_applied)
            
            # 记录正前方雷达的原始距离和加噪距离（米）
            self.cognitive_viz_data['original_observations'].append(front_beam_data['original_distance'])
            self.cognitive_viz_data['noisy_observations'].append(front_beam_data['noisy_distance'])
            
            if original_action is not None:
                self.cognitive_viz_data['original_actions'].append(original_action.copy())
                self.cognitive_viz_data['delayed_actions'].append(action.copy())
        
        return action
    
    def run_single_episode(self, render: bool = True, max_steps: int = 1000, 
                          scenario_seed: Optional[int] = None, deterministic: bool = True) -> Dict:
        """
        运行单个仿真episode
        
        Args:
            render: 是否渲染
            max_steps: 最大步数
            scenario_seed: 场景种子
            deterministic: 是否使用确定性策略
            
        Returns:
            episode统计信息
        """
        # 创建环境
        env = self.create_environment(render=render, scenario_seed=scenario_seed)
        
        # 重置环境
        obs, info = env.reset()
        
        # === 认知模块：重置状态和附加到环境 ===
        if self.use_cognitive_modules:
            if self.cognitive_bias_module:
                self.cognitive_bias_module.reset()
            if self.cognitive_delay_module:  
                self.cognitive_delay_module.reset()
            if self.cognitive_perception_module:
                self.cognitive_perception_module.reset()
                # 🔧 关键：将噪声雷达附加到环境，替换原始雷达传感器
                self.cognitive_perception_module.attach_to_env(env)
                print("🔗 认知感知模块已附加到环境 - 噪声将在传感器层自动注入")
                
                # 🔍 验证环境配置，确保避免双重噪声
                lidar_config = env.config.get("vehicle_config", {}).get("lidar", {})
                gaussian_noise = lidar_config.get("gaussian_noise", 0.0)
                dropout_prob = lidar_config.get("dropout_prob", 0.0)
                
                if gaussian_noise > 0.0 or dropout_prob > 0.0:
                    print(f"⚠️ 警告：检测到环境lidar配置中存在额外噪声！")
                    print(f"   gaussian_noise: {gaussian_noise}")
                    print(f"   dropout_prob: {dropout_prob}")
                    print(f"   这可能导致双重噪声问题，建议设置为0.0")
                else:
                    print(f"✅ 环境lidar噪声配置正确 (gaussian_noise=0.0, dropout_prob=0.0)")
            
            # 🔗 认知偏差模块附加到环境
            if self.cognitive_bias_module:
                success = self.cognitive_bias_module.attach_to_env(env)
                if success:
                    print("🔗 认知偏差模块已附加到环境 - 将基于TTA动态调整奖励")
                else:
                    print("⚠️ 认知偏差模块附加失败")
        
        # Episode统计
        episode_stats = {
            "total_reward": 0.0,
            "episode_length": 0,
            "success": False,
            "collision": False,
            "out_of_road": False,
            "max_speed": 0.0,
            "avg_speed": 0.0,
            "path_completion": 0.0,
            "actions": [],
            "speeds": [],
            "rewards": [],
            # 🔧 新增：变道惩罚统计变量
            "lane_change_penalties": [],
            "lane_change_speed_ratios": [],
            "cooldown_violations": [],
            "lane_changes": 0,
            # 🧠 新增：认知模块统计
            "cognitive_bias_info": [],
            "cognitive_delay_info": [],
            "cognitive_perception_info": []
        }
        
        speeds = []
        step_count = 0
        
        try:
            for step in range(max_steps):
                # 获取PPO动作
                action = self.get_action(obs, deterministic=deterministic, env=env, step_count=step_count)
                episode_stats["actions"].append(action.copy())
                
                # 调试：打印前几步的动作
                if step < 5:
                    print(f"  步骤 {step}: 动作=[{action[0]:.3f}, {action[1]:.3f}]")
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                
                original_reward = reward if self.enable_cognitive_visualization and self.cognitive_viz_data is not None else None
                
                # === 认知偏差模块：处理奖励偏差 ===
                bias_applied = False
                if self.use_cognitive_modules and self.cognitive_bias_module:
                    try:
                        # 根据文档使用正确的参数调用 process_reward
                        if hasattr(self.cognitive_bias_module, 'process_reward'):
                            reward_result = self.cognitive_bias_module.process_reward(
                                original_reward=reward,
                                env=env,
                                info=info,
                                is_ppo_mode=True
                            )
                            
                            # 处理返回值 - 根据文档是 (adjusted_reward, bias_info)
                            orig_reward_debug = reward
                            if isinstance(reward_result, (tuple, list)) and len(reward_result) >= 2:
                                adjusted_reward, bias_info = reward_result[0], reward_result[1]
                                reward = float(adjusted_reward)
                                
                                # 记录偏差信息用于可视化
                                if isinstance(bias_info, dict):
                                    bias_amount = bias_info.get('bias_applied', 0.0)
                                    inverse_tta = bias_info.get('inverse_tta', 0.0)
                                    bias_active = bias_info.get('bias_active', False)
                                    
                                    if bias_active and abs(bias_amount) > 1e-6:
                                        print(f"🧠 认知偏差: {orig_reward_debug:.3f} → {reward:.3f} (偏差: {bias_amount:+.3f}, TTA⁻¹: {inverse_tta:.3f})")
                                        bias_applied = True
                                    else:
                                        # 如果偏差不活跃或很小，不应用测试偏差
                                        bias_applied = False
                                else:
                                    bias_applied = True
                            else:
                                # 兼容性处理 - 单一返回值
                                reward = float(reward_result) if reward_result is not None else reward
                                bias_applied = abs(reward - orig_reward_debug) > 1e-6
                        else:
                            # 如果没有process_reward方法，应用简单偏差
                            orig_reward_debug = reward
                            if reward > 0:
                                reward *= 0.9  # 轻微减少正奖励
                            else:
                                reward *= 1.1  # 增加负奖励影响
                            

                            
                            bias_applied = True
                    except Exception as e:
                        print(f"⚠️ 认知偏差模块处理失败: {e}")
                
                # === 收集偏差可视化数据 ===
                if self.enable_cognitive_visualization and self.cognitive_viz_data is not None:
                    # 偏差模块数据
                    try:
                        if self.cognitive_bias_module and bias_applied:
                            if hasattr(self.cognitive_bias_module, 'get_bias_info'):
                                bias_info = self.cognitive_bias_module.get_bias_info()
                                # 处理复杂的bias_info对象
                                if isinstance(bias_info, dict):
                                    bias_strength = float(bias_info.get('bias_strength', 0.0))
                                elif isinstance(bias_info, (list, tuple)):
                                    bias_strength = float(bias_info[0]) if len(bias_info) > 0 else 0.0
                                else:
                                    bias_strength = float(bias_info) if bias_info is not None else 0.0
                            else:
                                # 如果没有get_bias_info方法，计算简单的偏差强度
                                bias_strength = abs(reward - original_reward) if original_reward is not None else 0.0
                        else:
                            bias_strength = 0.0
                    except Exception as e:
                        print(f"⚠️ 偏差数据收集失败: {e}")
                        bias_strength = 0.0
                    
                    # 记录偏差相关数据
                    self.cognitive_viz_data['bias_strength'].append(bias_strength)
                    self.cognitive_viz_data['bias_applied'].append(bias_applied)
                    
                    if original_reward is not None:
                        self.cognitive_viz_data['original_rewards'].append(float(original_reward))
                        self.cognitive_viz_data['modified_rewards'].append(float(reward))
                
                # 🚀 新增：收集速度控制奖励数据
                if self.enable_speed_control_visualization and self.speed_control_viz_data is not None:
                    # 记录步数
                    self.speed_control_viz_data['step_count'].append(step_count)
                    
                    try:
                        # 🔧 修复：检查环境类型和step_infos属性
                        if hasattr(env, 'step_infos') and isinstance(env.step_infos, dict):
                            # 🔧 修复：获取正确的agent ID
                            agent_id = None
                            if hasattr(env, 'agent') and hasattr(env.agent, 'id'):
                                agent_id = env.agent.id
                            elif hasattr(env, 'agents'):
                                # 如果没有agent属性，尝试从agents中获取第一个
                                agent_ids = list(env.agents.keys())
                                if agent_ids:
                                    agent_id = agent_ids[0]
                            
                            # 🔧 修复：如果找不到agent_id，尝试从step_infos中获取
                            if agent_id is None and env.step_infos:
                                # 使用step_infos中的第一个agent ID
                                agent_id = list(env.step_infos.keys())[0]
                                print(f"    🔍 从step_infos中获取agent_id: {agent_id}")
                            
                            if agent_id and agent_id in env.step_infos:
                                step_info = env.step_infos[agent_id]
                                
                                # 调试：打印step_info的内容（前几步）
                                if step < 5:
                                    print(f"    🔍 速度控制调试 - agent_id: {agent_id}")
                                    print(f"    🔍 速度控制调试 - step_info键: {list(step_info.keys())}")
                                    if 'sc_r_total' in step_info:
                                        print(f"    ✅ 找到速度控制奖励: {step_info['sc_r_total']:.4f}")
                                    else:
                                        print(f"    ❌ 未找到速度控制奖励键")
                                
                                # 收集速度控制奖励数据
                                if 'sc_r_total' in step_info:
                                    self.speed_control_viz_data['speed_control_total'].append(float(step_info['sc_r_total']))
                                    self.speed_control_viz_data['speed_control_tracking'].append(float(step_info.get('sc_r_track', 0.0)))
                                    self.speed_control_viz_data['speed_control_soft_wall'].append(float(step_info.get('sc_r_wall', 0.0)))
                                    self.speed_control_viz_data['speed_control_behavior_guidance'].append(float(step_info.get('sc_r_act_over', 0.0)))
                                    
                                    # 记录速度相关信息
                                    if 'sc_v' in step_info:
                                        self.speed_control_viz_data['vehicle_speeds'].append(float(step_info['sc_v']))
                                    if 'sc_v_ref' in step_info:
                                        self.speed_control_viz_data['speed_references'].append(float(step_info['sc_v_ref']))
                                    if 'sc_dv' in step_info:
                                        self.speed_control_viz_data['speed_deviations'].append(float(step_info['sc_dv']))
                                    
                                    # 🔧 修复：打印成功收集的数据
                                    if step < 5:
                                        print(f"    ✅ 成功收集速度控制数据: sc_r_total={step_info['sc_r_total']:.4f}")
                                else:
                                    # 🔧 修复：如果没有速度控制奖励，尝试从环境直接计算
                                    if hasattr(env, '_compute_speed_control_reward') and hasattr(env, 'agent'):
                                        try:
                                            # 直接调用速度控制奖励计算
                                            sc_reward = env._compute_speed_control_reward(env.agent, action)
                                            # 从step_infos中获取最新数据
                                            if agent_id in env.step_infos:
                                                step_info = env.step_infos[agent_id]
                                                if 'sc_r_total' in step_info:
                                                    self.speed_control_viz_data['speed_control_total'].append(float(step_info['sc_r_total']))
                                                    self.speed_control_viz_data['speed_control_tracking'].append(float(step_info.get('sc_r_track', 0.0)))
                                                    self.speed_control_viz_data['speed_control_soft_wall'].append(float(step_info.get('sc_r_wall', 0.0)))
                                                    self.speed_control_viz_data['speed_control_behavior_guidance'].append(float(step_info.get('sc_r_act_over', 0.0)))
                                                    
                                                    if 'sc_v' in step_info:
                                                        self.speed_control_viz_data['vehicle_speeds'].append(float(step_info['sc_v']))
                                                    if 'sc_v_ref' in step_info:
                                                        self.speed_control_viz_data['speed_references'].append(float(step_info['sc_v_ref']))
                                                    if 'sc_dv' in step_info:
                                                        self.speed_control_viz_data['speed_deviations'].append(float(step_info['sc_dv']))
                                                else:
                                                    # 填充默认值
                                                    self._fill_default_speed_control_data()
                                            else:
                                                self._fill_default_speed_control_data()
                                        except Exception as e:
                                            print(f"    ⚠️ 直接计算速度控制奖励失败: {e}")
                                            self._fill_default_speed_control_data()
                                    else:
                                        # 填充默认值
                                        self._fill_default_speed_control_data()
                            else:
                                # 调试：打印环境信息
                                if step < 5:
                                    print(f"    🔍 速度控制调试 - 环境信息:")
                                    print(f"       hasattr(env, 'step_infos'): {hasattr(env, 'step_infos')}")
                                    print(f"       env.step_infos类型: {type(env.step_infos)}")
                                    print(f"       env.step_infos内容: {env.step_infos}")
                                    if hasattr(env, 'agent') and hasattr(env.agent, 'id'):
                                        print(f"       agent.id: {env.agent.id}")
                                    elif hasattr(env, 'agents'):
                                        print(f"       agents键: {list(env.agents.keys())}")
                                
                                # 如果没有step_info，填充默认值
                                self._fill_default_speed_control_data()
                        else:
                            # 调试：打印环境类型信息
                            if step < 5:
                                print(f"    🔍 速度控制调试 - 环境类型检查:")
                                print(f"       环境类型: {type(env)}")
                                print(f"       环境类名: {env.__class__.__name__}")
                                print(f"       是否SpeedControlMetaDriveEnv: {hasattr(env, 'step_infos')}")
                                print(f"       step_infos类型: {type(getattr(env, 'step_infos', None))}")
                            
                            # 如果环境不支持速度控制，填充默认值
                            self._fill_default_speed_control_data()
                    except Exception as e:
                        print(f"⚠️ 速度控制奖励数据收集失败: {e}")
                        # 异常时填充默认值
                        self._fill_default_speed_control_data()
                
                # 更新统计
                episode_stats["total_reward"] += reward
                episode_stats["episode_length"] += 1
                episode_stats["rewards"].append(reward)
                
                # 记录速度 - 从agent对象获取
                if hasattr(env.agent, 'speed'):
                    speed = env.agent.speed
                    speeds.append(speed)
                    episode_stats["speeds"].append(speed)
                    episode_stats["max_speed"] = max(episode_stats["max_speed"], speed)
                    
                    # 调试：打印前几步的速度信息
                    if step < 5:
                        print(f"    速度: {speed:.3f} m/s, 奖励: {reward:.3f}")
                else:
                    if step < 5:
                        print(f"    ⚠️ 无法获取速度信息, 奖励: {reward:.3f}")
                
                # 🔧 新增：变道检测和惩罚计算
                lane_change_detected = False
                
                # 检查车道变更（基于车道索引变化）
                if hasattr(env.agent, 'lane_index'):
                    current_lane_index = env.agent.lane_index
                    agent_id = getattr(env.agent, 'id', id(env.agent))
                    
                    # 检查是否存储了上一个车道索引
                    if agent_id in self._last_lane_index:
                        if self._last_lane_index[agent_id] != current_lane_index:
                            lane_change_detected = True
                            episode_stats["lane_changes"] += 1
                    else:
                        # 第一次运行，记录初始车道索引
                        self._last_lane_index[agent_id] = current_lane_index
                
                # 变道惩罚计算
                if lane_change_detected:
                    # 获取当前速度
                    current_speed = abs(speed) if 'speed' in locals() else 0.0
                    
                    # 计算速度比例 - 使用命令行参数或默认值
                    v_limit = getattr(self, 'args', None) and getattr(self.args, 'v_limit', 15.0) or 15.0
                    speed_ratio = current_speed / v_limit
                    speed_ratio = min(speed_ratio, 2.0)  # 限制最大比例
                    
                    # 基础变道惩罚 - 使用命令行参数或默认值
                    w_lc = getattr(self, 'args', None) and getattr(self.args, 'w_lc', 0.6) or 0.6
                    base_penalty = w_lc
                    
                    # 高速放大惩罚 - 使用命令行参数或默认值
                    k_speed = getattr(self, 'args', None) and getattr(self.args, 'k_speed', 1.0) or 1.0
                    speed_penalty = base_penalty * (1 + k_speed * speed_ratio)
                    
                    # 检查冷却时间
                    current_step = step_count
                    
                    if agent_id in self._last_lane_change_step:
                        steps_since_last_change = current_step - self._last_lane_change_step[agent_id]
                        if steps_since_last_change < self._lane_change_cooldown_steps:
                            # 冷却期内，附加惩罚 - 使用命令行参数或默认值
                            w_lc_cool = getattr(self, 'args', None) and getattr(self.args, 'w_lc_cool', 0.6) or 0.6
                            speed_penalty += w_lc_cool
                            episode_stats["cooldown_violations"].append(1)
                        else:
                            episode_stats["cooldown_violations"].append(0)
                    else:
                        episode_stats["cooldown_violations"].append(0)
                    
                    # 更新最后变道时间
                    self._last_lane_change_step[agent_id] = current_step
                    
                    # 记录变道惩罚统计
                    episode_stats["lane_change_penalties"].append(speed_penalty)
                    episode_stats["lane_change_speed_ratios"].append(speed_ratio)
                    
                    # 将惩罚应用到奖励中
                    episode_stats["total_reward"] -= speed_penalty
                    
                    # 调试：打印变道惩罚信息
                    if step < 5:
                        print(f"    🚗 变道检测！惩罚: {speed_penalty:.3f}, 速度比: {speed_ratio:.3f}")
                
                # 更新车道索引（无论是否变道）
                if hasattr(env.agent, 'lane_index'):
                    current_lane_index = env.agent.lane_index
                    agent_id = getattr(env.agent, 'id', id(env.agent))
                    self._last_lane_index[agent_id] = current_lane_index
                
                # === 认知模块：收集统计信息 ===
                if self.use_cognitive_modules:
                    if self.cognitive_bias_module:
                        try:
                            if hasattr(self.cognitive_bias_module, 'get_bias_info'):
                                bias_info = self.cognitive_bias_module.get_bias_info()
                                # 只保存简化版本，避免复杂对象
                                if isinstance(bias_info, dict):
                                    simplified_info = {
                                        'bias_strength': float(bias_info.get('bias_strength', 0.0)),
                                        'bias_active': bool(bias_info.get('bias_active', False))
                                    }
                                else:
                                    simplified_info = {'status': 'active', 'value': str(bias_info)[:50]}
                                episode_stats["cognitive_bias_info"].append(simplified_info)
                            else:
                                episode_stats["cognitive_bias_info"].append({'status': 'active'})
                        except Exception as e:
                            print(f"⚠️ 认知偏差统计收集失败: {e}")
                    
                    if self.cognitive_delay_module:
                        try:
                            if hasattr(self.cognitive_delay_module, 'get_delay_info'):
                                delay_info = self.cognitive_delay_module.get_delay_info()
                                if isinstance(delay_info, dict):
                                    simplified_info = {k: v for k, v in delay_info.items() if isinstance(v, (int, float, str, bool))}
                                else:
                                    simplified_info = {'delay_steps': self.cognitive_delay_module.delay_steps}
                                episode_stats["cognitive_delay_info"].append(simplified_info)
                            else:
                                episode_stats["cognitive_delay_info"].append({'delay_steps': self.cognitive_delay_module.delay_steps})
                        except Exception as e:
                            print(f"⚠️ 认知延迟统计收集失败: {e}")
                    
                    if self.cognitive_perception_module:
                        try:
                            if hasattr(self.cognitive_perception_module, 'get_perception_info'):
                                perception_info = self.cognitive_perception_module.get_perception_info()
                                if isinstance(perception_info, dict):
                                    simplified_info = {k: v for k, v in perception_info.items() if isinstance(v, (int, float, str, bool))}
                                else:
                                    simplified_info = {'status': 'active'}
                                episode_stats["cognitive_perception_info"].append(simplified_info)
                            else:
                                episode_stats["cognitive_perception_info"].append({'status': 'active'})
                        except Exception as e:
                            print(f"⚠️ 认知感知统计收集失败: {e}")
                
                # 检查终止条件
                if terminated or truncated:
                    # 成功到达
                    if info.get("arrive_dest", False):
                        episode_stats["success"] = True
                    
                    # 碰撞
                    if info.get("crash", False) or info.get("crash_vehicle", False):
                        episode_stats["collision"] = True
                    
                    # 冲出道路
                    if info.get("out_of_road", False):
                        episode_stats["out_of_road"] = True
                    
                    break
                
                step_count += 1
                
                # 渲染时添加延迟
                if render:
                    time.sleep(0.05)  # 50ms延迟，便于观察
        
        finally:
            # 获取最终路径完成度信息
            try:
                # 首先尝试从最后一个info中获取
                if "route_completion" in info:
                    episode_stats["path_completion"] = info["route_completion"]
                # 然后尝试从导航模块直接获取
                elif hasattr(env.agent, 'navigation') and hasattr(env.agent.navigation, 'route_completion'):
                    episode_stats["path_completion"] = env.agent.navigation.route_completion
                # 最后尝试其他方法
                elif hasattr(env.agent, 'navigation'):
                    nav = env.agent.navigation
                    if hasattr(nav, 'get_current_lane_progress'):
                        episode_stats["path_completion"] = nav.get_current_lane_progress()
                    
            except Exception as e:
                print(f"⚠️  路径完成度获取失败: {e}")
                episode_stats["path_completion"] = 0.0
            
            # 计算平均速度
            if speeds:
                episode_stats["avg_speed"] = np.mean(speeds)
            
            # === 生成认知可视化 ===
            # 定义保存目录（为认知模块可视化使用）
            str_save_dir = str(f"cognitive_visualization/cognitive_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            if self.enable_cognitive_visualization and self.cognitive_viz_data:
                try:
                    viz_path = self.generate_cognitive_visualization(episode_stats, str_save_dir)
                    episode_stats["cognitive_visualization_path"] = viz_path
                    
                    # 清空数据为下一个episode准备
                    self.clear_cognitive_visualization_data()
                except Exception as e:
                    print(f"⚠️ 认知可视化生成失败: {e}")
            
            # === 生成速度控制可视化 ===
            if self.enable_speed_control_visualization and self.speed_control_viz_data:
                try:
                    viz_path = self.generate_speed_control_visualization(episode_stats)
                    episode_stats["speed_control_visualization_path"] = viz_path
                    
                    # 清空数据为下一个episode准备
                    self.clear_speed_control_visualization_data()
                except Exception as e:
                    print(f"⚠️ 速度控制可视化生成失败: {e}")
            
            # === 认知模块：分离环境 ===
            if self.use_cognitive_modules and self.cognitive_perception_module:
                try:
                    # 🔧 在环境关闭前生成可视化（需要在分离前完成）
                    self.cognitive_perception_module.generate_visualization(save_dir=str_save_dir, env=env)
                    print("📊 认知感知模块可视化已生成")
                except Exception as e:
                    print(f"⚠️ 认知感知模块可视化生成失败: {e}")
                finally:
                    # 🔧 分离噪声雷达，恢复原始传感器
                    self.cognitive_perception_module.detach_from_env()
                    print("🔗 认知感知模块已从环境分离")
            
            # 🔗 认知偏差模块可视化和分离
            if self.cognitive_bias_module:
                try:
                    # 生成认知偏差模块可视化
                    self.cognitive_bias_module.generate_visualization(env=env, save_dir=str_save_dir)
                    print("📊 认知偏差模块可视化已生成")
                except Exception as e:
                    print(f"⚠️ 认知偏差模块可视化生成失败: {e}")
                finally:
                    try:
                        self.cognitive_bias_module.detach_from_env()
                        print("🔗 认知偏差模块已从环境分离")
                    except Exception as e:
                        print(f"⚠️ 认知偏差模块分离失败: {e}")
            
            env.close()
        
        return episode_stats
    
    def run_simulation(self, num_episodes: int = 5, render: bool = True, 
                      max_steps: int = 1000, deterministic: bool = True) -> List[Dict]:
        """
        运行多个仿真episodes
        
        Args:
            num_episodes: episode数量
            render: 是否渲染
            max_steps: 每个episode最大步数
            deterministic: 是否使用确定性策略
            
        Returns:
            所有episodes的统计信息列表
        """
        print(f"\n🚗 开始PPO仿真 ({num_episodes} episodes)")
        print("=" * 60)
        
        all_stats = []
        
        for episode in range(num_episodes):
            print(f"\n🎮 Episode {episode + 1}/{num_episodes}")
            
            # 固定第一个场景种子，确保背景车初始状态可重复；后续场景可以不同
            if episode == 0:
                scenario_seed = 8888  # 第一个场景始终使用固定种子
            else:
                scenario_seed = 8888 + episode  # 后续场景使用不同种子
            
            # 运行episode
            stats = self.run_single_episode(
                render=render,
                max_steps=max_steps,
                scenario_seed=scenario_seed,
                deterministic=deterministic
            )
            
            all_stats.append(stats)
            
            # 打印episode结果
            print(f"📊 Episode {episode + 1} 结果:")
            print(f"   💰 总奖励: {stats['total_reward']:.2f}")
            print(f"   📏 Episode长度: {stats['episode_length']}")
            print(f"   🏁 成功到达: {'✅' if stats['success'] else '❌'}")
            print(f"   💥 发生碰撞: {'❌' if stats['collision'] else '✅'}")
            print(f"   🛣️  冲出道路: {'❌' if stats['out_of_road'] else '✅'}")
            print(f"   🚀 最高速度: {stats['max_speed']:.2f} m/s")
            print(f"   📈 平均速度: {stats['avg_speed']:.2f} m/s")
            print(f"   🎯 路径完成度: {stats['path_completion']:.1%}")
            
            # 🔧 新增：变道惩罚统计输出
            if stats['lane_changes'] > 0:
                avg_penalty = np.mean(stats['lane_change_penalties']) if stats['lane_change_penalties'] else 0
                avg_speed_ratio = np.mean(stats['lane_change_speed_ratios']) if stats['lane_change_speed_ratios'] else 0
                cooldown_violations = np.sum(stats['cooldown_violations']) if stats['cooldown_violations'] else 0
                print(f"   🚗 变道次数: {stats['lane_changes']}")
                print(f"   💸 平均变道惩罚: {avg_penalty:.3f}")
                print(f"   ⚡ 平均变道速度比: {avg_speed_ratio:.3f}")
                print(f"   ⏰ 冷却期违规: {cooldown_violations}")
            else:
                print(f"   🚗 变道次数: 0")
            
            # 🎨 新增：认知可视化输出
            if 'cognitive_visualization_path' in stats and stats['cognitive_visualization_path']:
                print(f"   🎨 认知可视化: {stats['cognitive_visualization_path']}")
            
            # 🚀 新增：速度控制可视化输出
            if 'speed_control_visualization_path' in stats and stats['speed_control_visualization_path']:
                print(f"   🚀 速度控制可视化: {stats['speed_control_visualization_path']}")
        
        # 打印总体统计
        self._print_summary_stats(all_stats)
        
        return all_stats
    
    def _print_summary_stats(self, all_stats: List[Dict]):
        """打印总体统计信息"""
        if not all_stats:
            return
        
        print(f"\n📈 总体统计 ({len(all_stats)} episodes)")
        print("=" * 60)
        
        # 计算统计指标
        success_rate = np.mean([s['success'] for s in all_stats])
        collision_rate = np.mean([s['collision'] for s in all_stats])
        out_of_road_rate = np.mean([s['out_of_road'] for s in all_stats])
        
        avg_reward = np.mean([s['total_reward'] for s in all_stats])
        avg_length = np.mean([s['episode_length'] for s in all_stats])
        avg_speed = np.mean([s['avg_speed'] for s in all_stats])
        avg_completion = np.mean([s['path_completion'] for s in all_stats])
        
        # 🔧 新增：变道统计计算
        total_lane_changes = np.sum([s['lane_changes'] for s in all_stats])
        avg_lane_change_penalty = np.mean([np.mean(s['lane_change_penalties']) if s['lane_change_penalties'] else 0 for s in all_stats])
        avg_lane_change_speed_ratio = np.mean([np.mean(s['lane_change_speed_ratios']) if s['lane_change_speed_ratios'] else 0 for s in all_stats])
        total_cooldown_violations = np.sum([np.sum(s['cooldown_violations']) if s['cooldown_violations'] else 0 for s in all_stats])
        
        print(f"🏆 成功率: {success_rate:.1%}")
        print(f"💥 碰撞率: {collision_rate:.1%}")
        print(f"🛣️  冲出道路率: {out_of_road_rate:.1%}")
        print(f"💰 平均奖励: {avg_reward:.2f}")
        print(f"📏 平均Episode长度: {avg_length:.1f}")
        print(f"🚀 平均速度: {avg_speed:.2f} m/s")
        print(f"🎯 平均路径完成度: {avg_completion:.1%}")
        
        # 🔧 新增：变道统计输出
        print(f"🚗 总变道次数: {total_lane_changes}")
        print(f"💸 平均变道惩罚: {avg_lane_change_penalty:.3f}")
        print(f"⚡ 平均变道速度比: {avg_lane_change_speed_ratio:.3f}")
        print(f"⏰ 总冷却期违规: {total_cooldown_violations}")
        
        # 🧠 新增：认知模块统计输出
        if hasattr(self, 'use_cognitive_modules') and self.use_cognitive_modules:
            print(f"\n🧠 认知模块统计:")
            
            if self.cognitive_bias_module:
                try:
                    if hasattr(self.cognitive_bias_module, 'get_statistics'):
                        bias_stats = self.cognitive_bias_module.get_statistics()
                        print(f"   💭 认知偏差:")
                        print(f"      平均偏差强度: {bias_stats.get('average_bias', 0.0):.3f}")
                        print(f"      偏差应用次数: {bias_stats.get('active_steps', 0)}")
                    else:
                        print(f"   💭 认知偏差: 基本模式运行")
                except Exception as e:
                    print(f"   ⚠️ 认知偏差统计获取失败: {e}")
            
            if self.cognitive_delay_module:
                try:
                    # 延迟模块没有get_statistics方法，显示基本信息
                    delay_steps = getattr(self.cognitive_delay_module, 'delay_steps', 0)
                    print(f"   ⏰ 认知延迟:")
                    print(f"      配置延迟步数: {delay_steps}")
                    print(f"      状态: 活跃")
                except Exception as e:
                    print(f"   ⚠️ 认知延迟统计获取失败: {e}")
            
            if self.cognitive_perception_module:
                try:
                    # 感知模块没有get_statistics方法，显示配置信息
                    noise_config = getattr(self.cognitive_perception_module, 'noise_config', {})
                    sigma0_value = noise_config.get('sigma0', 0.01)
                    # 确保获取数值类型
                    if isinstance(sigma0_value, (int, float)):
                        base_noise = float(sigma0_value)
                    else:
                        base_noise = 0.01  # 默认值
                    print(f"   👁️ 认知感知:")
                    print(f"      基准噪声水平: {base_noise:.3f}")
                    print(f"      状态: 活跃")
                except Exception as e:
                    print(f"   ⚠️ 认知感知统计获取失败: {e}")
    
    def evaluate_model(self, num_episodes: int = 20, render: bool = False) -> Dict:
        """
        评估模型性能
        
        Args:
            num_episodes: 评估episodes数量
            render: 是否渲染
            
        Returns:
            评估结果字典
        """
        print(f"\n🔍 模型性能评估 ({num_episodes} episodes)")
        print("=" * 60)
        
        all_stats = self.run_simulation(
            num_episodes=num_episodes,
            render=render,
            max_steps=1000,
            deterministic=True
        )
        
        # 计算详细评估指标
        evaluation = {
            "num_episodes": len(all_stats),
            "success_rate": np.mean([s['success'] for s in all_stats]),
            "collision_rate": np.mean([s['collision'] for s in all_stats]),
            "out_of_road_rate": np.mean([s['out_of_road'] for s in all_stats]),
            "avg_reward": np.mean([s['total_reward'] for s in all_stats]),
            "std_reward": np.std([s['total_reward'] for s in all_stats]),
            "avg_episode_length": np.mean([s['episode_length'] for s in all_stats]),
            "avg_speed": np.mean([s['avg_speed'] for s in all_stats]),
            "avg_path_completion": np.mean([s['path_completion'] for s in all_stats]),
            # 🔧 新增：变道统计评估指标
            "total_lane_changes": np.sum([s['lane_changes'] for s in all_stats]),
            "avg_lane_change_penalty": np.mean([np.mean(s['lane_change_penalties']) if s['lane_change_penalties'] else 0 for s in all_stats]),
            "avg_lane_change_speed_ratio": np.mean([np.mean(s['lane_change_speed_ratios']) if s['lane_change_speed_ratios'] else 0 for s in all_stats]),
            "total_cooldown_violations": np.sum([np.sum(s['cooldown_violations']) if s['cooldown_violations'] else 0 for s in all_stats]),
            "checkpoint_path": self.checkpoint_path,
            "training_iteration": self.checkpoint.get('iteration', 'Unknown'),
            "training_global_step": self.checkpoint.get('global_step', 'Unknown')
        }
        
        # 🧠 新增：认知模块评估指标
        if hasattr(self, 'use_cognitive_modules') and self.use_cognitive_modules:
            evaluation["cognitive_modules"] = {}
            
            if self.cognitive_bias_module:
                try:
                    bias_stats = self.cognitive_bias_module.get_statistics()
                    evaluation["cognitive_modules"]["bias"] = bias_stats
                except Exception as e:
                    print(f"⚠️ 认知偏差评估统计获取失败: {e}")
            
            if self.cognitive_delay_module:
                try:
                    delay_stats = self.cognitive_delay_module.get_statistics()
                    evaluation["cognitive_modules"]["delay"] = delay_stats
                except Exception as e:
                    print(f"⚠️ 认知延迟评估统计获取失败: {e}")
            
            if self.cognitive_perception_module:
                try:
                    perception_stats = self.cognitive_perception_module.get_statistics()
                    evaluation["cognitive_modules"]["perception"] = perception_stats
                except Exception as e:
                    print(f"⚠️ 认知感知评估统计获取失败: {e}")
        
        return evaluation

    def generate_cognitive_visualization(self, episode_data: Dict, save_dir: str = None) -> str:
        """
        生成认知模块可视化图表
        
        Args:
            episode_data: episode统计数据
            save_dir: 保存目录
            
        Returns:
            保存的图表文件路径
        """
        if not self.enable_cognitive_visualization or self.cognitive_viz_data is None:
            print("⚠️ 认知可视化未启用")
            return None
        
        if not self.cognitive_viz_data['step_count']:
            print("⚠️ 没有可视化数据")
            return None
        
        # 创建保存目录
        if save_dir is None:
            save_dir = "cognitive_visualization"
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 设置字体为英文，避免中文字体问题
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建综合可视化图表 - 改为3x2布局（移除速度控制奖励可视化）
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Cognitive Modules Visualization Analysis - {timestamp}', fontsize=16, fontweight='bold')
        
        steps = self.cognitive_viz_data['step_count']
        
        # 数据收集情况检查
        print(f"🔍 认知可视化数据: {len(steps)}步, 奖励{len(self.cognitive_viz_data['original_rewards'])}个, 观测{len(self.cognitive_viz_data['original_observations'])}个")
        
        # 1. 认知偏差强度时间序列
        ax1 = axes[0, 0]
        if self.cognitive_viz_data['bias_strength']:
            bias_strength = self.cognitive_viz_data['bias_strength']
            bias_applied = self.cognitive_viz_data['bias_applied']
            
            # 确保偏差数据长度和步数长度匹配
            min_len = min(len(steps), len(bias_strength), len(bias_applied))
            if min_len > 0:
                steps_subset = steps[:min_len]
                bias_strength_subset = bias_strength[:min_len]
                bias_applied_subset = bias_applied[:min_len]
                
                ax1.plot(steps_subset, bias_strength_subset, 'r-', linewidth=2, label='Bias Strength')
                ax1.fill_between(steps_subset, 0, bias_strength_subset, where=[x for x in bias_applied_subset], 
                               alpha=0.3, color='red', label='Bias Applied')
                ax1.set_title('Cognitive Bias Strength', fontweight='bold')
                ax1.set_xlabel('Steps')
                ax1.set_ylabel('Bias Strength')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'Insufficient Bias Data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Cognitive Bias Strength', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'Cognitive Bias Module Disabled', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Cognitive Bias Strength', fontweight='bold')
        
        # 2. 奖励对比
        ax2 = axes[0, 1]
        if self.cognitive_viz_data['original_rewards'] and self.cognitive_viz_data['modified_rewards']:
            original_rewards = self.cognitive_viz_data['original_rewards']
            modified_rewards = self.cognitive_viz_data['modified_rewards']
            
            # 确保奖励数据长度和步数长度匹配
            min_len = min(len(steps), len(original_rewards), len(modified_rewards))
            if min_len > 0:
                steps_subset = steps[:min_len]
                original_rewards_subset = original_rewards[:min_len]
                modified_rewards_subset = modified_rewards[:min_len]
                
                # 检查奖励差异程度
                reward_diff = [abs(orig - mod) for orig, mod in zip(original_rewards_subset, modified_rewards_subset)]
                max_diff = max(reward_diff) if reward_diff else 0
                
                ax2.plot(steps_subset, original_rewards_subset, 'g-', linewidth=3, label='Original Reward', alpha=0.8, marker='o', markersize=4)
                ax2.plot(steps_subset, modified_rewards_subset, 'b-', linewidth=3, label='Modified Reward', marker='s', markersize=4)
                
                # 只有当存在明显差异时才显示填充区域
                if max_diff > 1e-6:  # 只有当差异大于阈值时才显示
                    ax2.fill_between(steps_subset, original_rewards_subset, modified_rewards_subset, 
                                   alpha=0.3, color='orange', label='Bias Effect')
                
                ax2.set_title('Reward Signal Comparison', fontweight='bold')
                ax2.set_xlabel('Steps')
                ax2.set_ylabel('Reward Value')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Insufficient Reward Data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Reward Signal Comparison', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Insufficient Reward Data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Reward Signal Comparison', fontweight='bold')
        
        # 3. 认知延迟步数
        ax3 = axes[1, 0]
        if self.cognitive_viz_data['delay_steps']:
            delay_steps = self.cognitive_viz_data['delay_steps']
            delay_applied = self.cognitive_viz_data['delay_applied']
            
            # 确保延迟数据长度和步数长度匹配
            min_len = min(len(steps), len(delay_steps), len(delay_applied))
            if min_len > 0:
                steps_subset = steps[:min_len]
                delay_steps_subset = delay_steps[:min_len]
                delay_applied_subset = delay_applied[:min_len]
                
                ax3.plot(steps_subset, delay_steps_subset, 'purple', linewidth=2, marker='o', markersize=3, label='Delay Steps')
                ax3.fill_between(steps_subset, 0, delay_steps_subset, where=[x for x in delay_applied_subset], 
                               alpha=0.3, color='purple', label='Delay Applied')
                ax3.set_title('Cognitive Delay Steps', fontweight='bold')
                ax3.set_xlabel('Steps')
                ax3.set_ylabel('Delay Steps')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Insufficient Delay Data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Cognitive Delay Steps', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Cognitive Delay Module Disabled', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Cognitive Delay Steps', fontweight='bold')
        
        # 4. 动作对比（转向）
        ax4 = axes[1, 1]
        if self.cognitive_viz_data['original_actions'] and self.cognitive_viz_data['delayed_actions']:
            original_actions = np.array(self.cognitive_viz_data['original_actions'])
            delayed_actions = np.array(self.cognitive_viz_data['delayed_actions'])
            
            # 确保动作数据长度和步数长度匹配
            min_len = min(len(steps), len(original_actions), len(delayed_actions))
            if min_len > 0:
                steps_subset = steps[:min_len]
                original_actions_subset = original_actions[:min_len]
                delayed_actions_subset = delayed_actions[:min_len]
                
                ax4.plot(steps_subset, original_actions_subset[:, 0], 'g-', linewidth=2, label='Original Steering', alpha=0.7)
                ax4.plot(steps_subset, delayed_actions_subset[:, 0], 'orange', linewidth=2, label='Delayed Steering')
                ax4.set_title('Steering Action Comparison', fontweight='bold')
                ax4.set_xlabel('Steps')
                ax4.set_ylabel('Steering Value')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Insufficient Action Data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Steering Action Comparison', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Insufficient Action Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Steering Action Comparison', fontweight='bold')
        
        # 5. 感知噪声水平
        ax5 = axes[2, 0]
        if self.cognitive_viz_data['perception_noise']:
            perception_noise = self.cognitive_viz_data['perception_noise']
            perception_applied = self.cognitive_viz_data['perception_applied']
            
            # 确保感知噪声数据长度和步数长度匹配
            min_len = min(len(steps), len(perception_noise), len(perception_applied))
            if min_len > 0:
                steps_subset = steps[:min_len]
                perception_noise_subset = perception_noise[:min_len]
                perception_applied_subset = perception_applied[:min_len]
                
                ax5.plot(steps_subset, perception_noise_subset, 'cyan', linewidth=2, label='Front Beam Noise', marker='o', markersize=3)
                ax5.fill_between(steps_subset, 0, perception_noise_subset, where=[x for x in perception_applied_subset], 
                               alpha=0.3, color='cyan', label='Noise Applied')
                ax5.set_title('Front Radar Beam Noise Level (Real-time)', fontweight='bold')
                ax5.set_xlabel('Steps')
                ax5.set_ylabel('Noise Magnitude (meters)')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'Insufficient Perception Data', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Front Radar Beam Noise Level (Real-time)', fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'Cognitive Perception Module Disabled', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Front Radar Beam Noise Level (Real-time)', fontweight='bold')
        
        # 6. 观测对比
        ax6 = axes[2, 1]
        if self.cognitive_viz_data['original_observations'] and self.cognitive_viz_data['noisy_observations']:
            original_obs = self.cognitive_viz_data['original_observations']
            noisy_obs = self.cognitive_viz_data['noisy_observations']
            
            # 确保观测数据长度和步数长度匹配
            min_len = min(len(steps), len(original_obs), len(noisy_obs))
            if min_len > 0:
                steps_subset = steps[:min_len]
                original_obs_subset = original_obs[:min_len]
                noisy_obs_subset = noisy_obs[:min_len]
                
                # 检查观测差异程度  
                obs_diff = [abs(orig - noise) for orig, noise in zip(original_obs_subset, noisy_obs_subset)]
                max_diff = max(obs_diff) if obs_diff else 0
                
                ax6.plot(steps_subset, original_obs_subset, 'g-', linewidth=3, label='Original Distance', alpha=0.8, marker='o', markersize=4)
                ax6.plot(steps_subset, noisy_obs_subset, 'red', linewidth=3, label='Noisy Distance', marker='s', markersize=4)
                
                # 只有当存在明显差异时才显示填充区域
                if max_diff > 1e-6:  # 只有当差异大于阈值时才显示
                    ax6.fill_between(steps_subset, original_obs_subset, noisy_obs_subset, 
                                   alpha=0.3, color='yellow', label='Noise Effect')
                
                ax6.set_title('Front Radar Distance: Before vs After Noise', fontweight='bold')
                ax6.set_xlabel('Steps')
                ax6.set_ylabel('Distance (meters)')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'Insufficient Observation Data', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Front Radar Distance: Before vs After Noise', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'Insufficient Observation Data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Front Radar Distance: Before vs After Noise', fontweight='bold')
        

        
        plt.tight_layout()
        
        # 保存图表
        viz_filename = f"cognitive_visualization_{timestamp}.png"
        viz_path = os.path.join(save_dir, viz_filename)
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成统计报告
        self._generate_cognitive_report(episode_data, save_dir, timestamp)
        
        print(f"✅ 认知可视化已保存: {viz_path}")
        return viz_path
    
    def _generate_cognitive_report(self, episode_data: Dict, save_dir: str, timestamp: str):
        """生成认知模块统计报告"""
        report_path = os.path.join(save_dir, f"cognitive_report_{timestamp}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 认知模块分析报告\n\n")
            f.write(f"**生成时间**: {timestamp}\n\n")
            
            # Episode基本信息
            f.write(f"## Episode基本信息\n\n")
            f.write(f"- **总步数**: {episode_data.get('episode_length', 0)}\n")
            f.write(f"- **总奖励**: {episode_data.get('total_reward', 0):.3f}\n")
            f.write(f"- **成功到达**: {'是' if episode_data.get('success', False) else '否'}\n")
            f.write(f"- **平均速度**: {episode_data.get('avg_speed', 0):.2f} m/s\n\n")
            
            # 认知偏差统计
            if self.cognitive_viz_data['bias_strength']:
                bias_data = self.cognitive_viz_data['bias_strength']
                bias_applied_count = sum(self.cognitive_viz_data['bias_applied'])
                
                f.write(f"## 认知偏差模块\n\n")
                f.write(f"- **偏差生效次数**: {bias_applied_count}\n")
                f.write(f"- **平均偏差强度**: {np.mean(bias_data):.4f}\n")
                f.write(f"- **最大偏差强度**: {np.max(bias_data):.4f}\n")
                f.write(f"- **偏差生效率**: {bias_applied_count/len(bias_data)*100:.1f}%\n\n")
            
            # 认知延迟统计
            if self.cognitive_viz_data['delay_steps']:
                delay_data = self.cognitive_viz_data['delay_steps']
                delay_applied_count = sum(self.cognitive_viz_data['delay_applied'])
                
                f.write(f"## 认知延迟模块\n\n")
                f.write(f"- **延迟生效次数**: {delay_applied_count}\n")
                f.write(f"- **平均延迟步数**: {np.mean(delay_data):.2f}\n")
                f.write(f"- **最大延迟步数**: {np.max(delay_data)}\n")
                f.write(f"- **延迟生效率**: {delay_applied_count/len(delay_data)*100:.1f}%\n\n")
            
            # 认知感知统计
            if self.cognitive_viz_data['perception_noise']:
                noise_data = self.cognitive_viz_data['perception_noise']
                perception_applied_count = sum(self.cognitive_viz_data['perception_applied'])
                
                f.write(f"## 认知感知模块\n\n")
                f.write(f"- **噪声生效次数**: {perception_applied_count}\n")
                f.write(f"- **平均噪声水平**: {np.mean(noise_data):.4f}\n")
                f.write(f"- **最大噪声水平**: {np.max(noise_data):.4f}\n")
                f.write(f"- **噪声生效率**: {perception_applied_count/len(noise_data)*100:.1f}%\n\n")
            
            # 影响分析
            if (self.cognitive_viz_data['original_rewards'] and 
                self.cognitive_viz_data['modified_rewards']):
                orig_rewards = np.array(self.cognitive_viz_data['original_rewards'])
                mod_rewards = np.array(self.cognitive_viz_data['modified_rewards'])
                reward_diff = mod_rewards - orig_rewards
                
                f.write(f"## 认知影响分析\n\n")
                f.write(f"- **平均奖励变化**: {np.mean(reward_diff):.4f}\n")
                f.write(f"- **最大负面影响**: {np.min(reward_diff):.4f}\n")
                f.write(f"- **最大正面影响**: {np.max(reward_diff):.4f}\n")
                f.write(f"- **奖励标准差变化**: {np.std(mod_rewards) - np.std(orig_rewards):.4f}\n\n")
            

        
        print(f"✅ 认知报告已保存: {report_path}")
    
    def clear_cognitive_visualization_data(self):
        """清空认知可视化数据"""
        if self.cognitive_viz_data:
            for key in self.cognitive_viz_data:
                self.cognitive_viz_data[key].clear()
            print("🗑️ 认知可视化数据已清空")
    
    def generate_speed_control_visualization(self, episode_data: Dict, save_dir: str = None) -> str:
        """
        生成速度控制奖励可视化图表
        
        Args:
            episode_data: episode统计数据
            save_dir: 保存目录
            
        Returns:
            保存的图表文件路径
        """
        if not self.enable_speed_control_visualization or self.speed_control_viz_data is None:
            print("⚠️ 速度控制可视化未启用")
            return None
        
        if not self.speed_control_viz_data['step_count']:
            print("⚠️ 没有速度控制可视化数据")
            return None
        
        # 创建保存目录
        if save_dir is None:
            save_dir = "fig_cog/speed_control_visualization"
        else:
            save_dir = os.path.join("fig_cog", "speed_control_visualization")
        
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 设置字体为英文，避免中文字体问题
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建速度控制可视化图表 - 2x2布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Speed Control Reward Visualization Analysis - {timestamp}', fontsize=16, fontweight='bold')
        
        steps = self.speed_control_viz_data['step_count']
        
        # 数据收集情况检查
        print(f"🔍 速度控制可视化数据: {len(steps)}步")
        
        # 1. 速度控制奖励子模块对比
        ax1 = axes[0, 0]
        if (self.speed_control_viz_data['speed_control_tracking'] and 
            self.speed_control_viz_data['speed_control_soft_wall'] and 
            self.speed_control_viz_data['speed_control_behavior_guidance']):
            
            tracking_rewards = self.speed_control_viz_data['speed_control_tracking']
            soft_wall_rewards = self.speed_control_viz_data['speed_control_soft_wall']
            behavior_rewards = self.speed_control_viz_data['speed_control_behavior_guidance']
            
            # 确保数据长度和步数长度匹配
            min_len = min(len(steps), len(tracking_rewards), len(soft_wall_rewards), len(behavior_rewards))
            if min_len > 0:
                steps_subset = steps[:min_len]
                tracking_subset = tracking_rewards[:min_len]
                soft_wall_subset = soft_wall_rewards[:min_len]
                behavior_subset = behavior_rewards[:min_len]
                
                ax1.plot(steps_subset, tracking_subset, 'g-', linewidth=2, label='Speed Tracking', marker='o', markersize=3)
                ax1.plot(steps_subset, soft_wall_subset, 'r-', linewidth=2, label='Soft Wall', marker='s', markersize=3)
                ax1.plot(steps_subset, behavior_subset, 'b-', linewidth=2, label='Behavior Guidance', marker='^', markersize=3)
                
                ax1.set_title('Speed Control Reward Sub-modules', fontweight='bold')
                ax1.set_xlabel('Steps')
                ax1.set_ylabel('Reward Value')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'Insufficient Speed Control Data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Speed Control Reward Sub-modules', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'Speed Control Module Disabled', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Speed Control Reward Sub-modules', fontweight='bold')
        
        # 2. 速度控制总奖励和车辆速度
        ax2 = axes[0, 1]
        if (self.speed_control_viz_data['speed_control_total'] and 
            self.speed_control_viz_data['vehicle_speeds'] and 
            self.speed_control_viz_data['speed_references']):
            
            total_rewards = self.speed_control_viz_data['speed_control_total']
            vehicle_speeds = self.speed_control_viz_data['vehicle_speeds']
            speed_refs = self.speed_control_viz_data['speed_references']
            
            # 确保数据长度和步数长度匹配
            min_len = min(len(steps), len(total_rewards), len(vehicle_speeds), len(speed_refs))
            if min_len > 0:
                steps_subset = steps[:min_len]
                total_subset = total_rewards[:min_len]
                speeds_subset = vehicle_speeds[:min_len]
                refs_subset = speed_refs[:min_len]
                
                # 创建双y轴
                ax2_twin = ax2.twinx()
                
                # 左y轴：速度控制总奖励
                line1 = ax2.plot(steps_subset, total_subset, 'purple', linewidth=2, label='Total SC Reward', marker='o', markersize=3)
                ax2.set_ylabel('Total Speed Control Reward', color='purple')
                ax2.tick_params(axis='y', labelcolor='purple')
                
                # 右y轴：车辆速度和参考速度
                line2 = ax2_twin.plot(steps_subset, speeds_subset, 'orange', linewidth=2, label='Vehicle Speed', marker='s', markersize=3)
                line3 = ax2_twin.plot(steps_subset, refs_subset, 'cyan', linewidth=2, label='Reference Speed', marker='^', markersize=3, linestyle='--')
                ax2_twin.set_ylabel('Speed (m/s)', color='orange')
                ax2_twin.tick_params(axis='y', labelcolor='orange')
                
                # 合并图例
                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                ax2.legend(lines, labels, loc='upper left')
                
                ax2.set_title('Speed Control Total Reward & Vehicle Speed', fontweight='bold')
                ax2.set_xlabel('Steps')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Insufficient Speed Control Data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Speed Control Total Reward & Vehicle Speed', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Speed Control Module Disabled', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Speed Control Total Reward & Vehicle Speed', fontweight='bold')
        
        # 3. 速度偏差分析
        ax3 = axes[1, 0]
        if self.speed_control_viz_data['speed_deviations']:
            deviations = self.speed_control_viz_data['speed_deviations']
            
            # 确保数据长度和步数长度匹配
            min_len = min(len(steps), len(deviations))
            if min_len > 0:
                steps_subset = steps[:min_len]
                deviations_subset = deviations[:min_len]
                
                ax3.plot(steps_subset, deviations_subset, 'red', linewidth=2, marker='o', markersize=3, label='Speed Deviation')
                ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Reference Line')
                
                ax3.set_title('Speed Deviation Analysis', fontweight='bold')
                ax3.set_xlabel('Steps')
                ax3.set_ylabel('Speed Deviation (m/s)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Insufficient Deviation Data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Speed Deviation Analysis', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Deviation Data Available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Speed Deviation Analysis', fontweight='bold')
        
        # 4. 速度控制奖励统计
        ax4 = axes[1, 1]
        if self.speed_control_viz_data['speed_control_total']:
            total_rewards = self.speed_control_viz_data['speed_control_total']
            tracking_rewards = self.speed_control_viz_data['speed_control_tracking']
            soft_wall_rewards = self.speed_control_viz_data['speed_control_soft_wall']
            behavior_rewards = self.speed_control_viz_data['speed_control_behavior_guidance']
            
            # 计算统计信息
            total_mean = np.mean(total_rewards)
            tracking_mean = np.mean(tracking_rewards)
            soft_wall_mean = np.mean(soft_wall_rewards)
            behavior_mean = np.mean(behavior_rewards)
            
            # 创建柱状图
            categories = ['Total', 'Tracking', 'Soft Wall', 'Behavior']
            values = [total_mean, tracking_mean, soft_wall_mean, behavior_mean]
            colors = ['purple', 'green', 'red', 'blue']
            
            bars = ax4.bar(categories, values, color=colors, alpha=0.7)
            ax4.set_title('Average Speed Control Rewards', fontweight='bold')
            ax4.set_ylabel('Average Reward Value')
            ax4.grid(True, alpha=0.3)
            
            # 在柱子上添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'No Reward Data Available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Average Speed Control Rewards', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        viz_filename = f"speed_control_visualization_{timestamp}.png"
        viz_path = os.path.join(save_dir, viz_filename)
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成统计报告
        self._generate_speed_control_report(episode_data, save_dir, timestamp)
        
        print(f"✅ 速度控制可视化已保存: {viz_path}")
        return viz_path
    
    def _generate_speed_control_report(self, episode_data: Dict, save_dir: str, timestamp: str):
        """生成速度控制奖励统计报告"""
        report_path = os.path.join(save_dir, f"speed_control_report_{timestamp}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 速度控制奖励分析报告\n\n")
            f.write(f"**生成时间**: {timestamp}\n\n")
            
            # Episode基本信息
            f.write(f"## Episode基本信息\n\n")
            f.write(f"- **总步数**: {episode_data.get('episode_length', 0)}\n")
            f.write(f"- **总奖励**: {episode_data.get('total_reward', 0):.3f}\n")
            f.write(f"- **成功到达**: {'是' if episode_data.get('success', False) else '否'}\n")
            f.write(f"- **平均速度**: {episode_data.get('avg_speed', 0):.2f} m/s\n\n")
            
            # 速度控制奖励统计
            if (self.speed_control_viz_data['speed_control_total'] and 
                any(abs(x) > 1e-6 for x in self.speed_control_viz_data['speed_control_total'])):
                
                total_rewards = np.array(self.speed_control_viz_data['speed_control_total'])
                tracking_rewards = np.array(self.speed_control_viz_data['speed_control_tracking'])
                soft_wall_rewards = np.array(self.speed_control_viz_data['speed_control_soft_wall'])
                behavior_rewards = np.array(self.speed_control_viz_data['speed_control_behavior_guidance'])
                
                f.write(f"## 速度控制奖励分析\n\n")
                f.write(f"- **总速度控制奖励**:\n")
                f.write(f"  - 平均: {np.mean(total_rewards):.4f}\n")
                f.write(f"  - 标准差: {np.std(total_rewards):.4f}\n")
                f.write(f"  - 最小值: {np.min(total_rewards):.4f}\n")
                f.write(f"  - 最大值: {np.max(total_rewards):.4f}\n\n")
                
                f.write(f"- **速度跟踪子模块**:\n")
                f.write(f"  - 平均奖励: {np.mean(tracking_rewards):.4f}\n")
                f.write(f"  - 标准差: {np.std(tracking_rewards):.4f}\n\n")
                
                f.write(f"- **超速软墙子模块**:\n")
                f.write(f"  - 平均奖励: {np.mean(soft_wall_rewards):.4f}\n")
                f.write(f"  - 标准差: {np.std(soft_wall_rewards):.4f}\n\n")
                
                f.write(f"- **行为导向子模块**:\n")
                f.write(f"  - 平均奖励: {np.mean(behavior_rewards):.4f}\n")
                f.write(f"  - 标准差: {np.std(behavior_rewards):.4f}\n\n")
                
                # 速度统计
                if self.speed_control_viz_data['vehicle_speeds']:
                    speeds = np.array(self.speed_control_viz_data['vehicle_speeds'])
                    refs = np.array(self.speed_control_viz_data['speed_references'])
                    deviations = np.array(self.speed_control_viz_data['speed_deviations'])
                    
                    f.write(f"- **速度统计**:\n")
                    f.write(f"  - 平均车辆速度: {np.mean(speeds):.2f} m/s\n")
                    f.write(f"  - 平均参考速度: {np.mean(refs):.2f} m/s\n")
                    f.write(f"  - 平均速度偏差: {np.mean(deviations):.2f} m/s\n")
                    f.write(f"  - 速度偏差标准差: {np.std(deviations):.2f} m/s\n\n")
        
        print(f"✅ 速度控制报告已保存: {report_path}")
    
    def clear_speed_control_visualization_data(self):
        """清空速度控制可视化数据"""
        if self.speed_control_viz_data:
            for key in self.speed_control_viz_data:
                self.speed_control_viz_data[key].clear()
            print("🗑️ 速度控制可视化数据已清空")
    
    def _fill_default_speed_control_data(self):
        """填充默认的速度控制数据"""
        if self.speed_control_viz_data:
            self.speed_control_viz_data['speed_control_total'].append(0.0)
            self.speed_control_viz_data['speed_control_tracking'].append(0.0)
            self.speed_control_viz_data['speed_control_soft_wall'].append(0.0)
            self.speed_control_viz_data['speed_control_behavior_guidance'].append(0.0)
            self.speed_control_viz_data['vehicle_speeds'].append(0.0)
            self.speed_control_viz_data['speed_references'].append(0.0)
            self.speed_control_viz_data['speed_deviations'].append(0.0)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PPO检查点仿真控制器")
    
    parser.add_argument("--checkpoint", type=str,
                       default="/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/ckpt_0827/AB_1_13_latest_model.pt",
                       help="检查点文件路径")
    
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径（可选）")
    
    parser.add_argument("--episodes", type=int, default=5,
                       help="仿真episodes数量")
    
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="每个episode最大步数")
    
    parser.add_argument("--no_render", action="store_true",
                       help="禁用渲染")
    
    parser.add_argument("--stochastic", action="store_true",
                       help="使用随机策略（默认确定性）")
    
    parser.add_argument("--evaluate", action="store_true",
                       help="运行模型评估")
    
    parser.add_argument("--eval_episodes", type=int, default=20,
                       help="评估episodes数量")
    
    # ===== 新增：变道惩罚配置参数 =====
    parser.add_argument("--w_lc", type=float, default=0.6,
                       help="基础变道成本 (默认: 0.6)")
    parser.add_argument("--k_speed", type=float, default=1.0,
                       help="高速放大系数 (默认: 1.0)")
    parser.add_argument("--v_limit", type=float, default=15.0,
                       help="用于速度归一的限速 (默认: 15.0)")
    parser.add_argument("--lc_cooldown_s", type=float, default=4.0,
                       help="变道冷却时间，秒 (默认: 4.0)")
    parser.add_argument("--w_lc_cool", type=float, default=1,
                       help="冷却期内附加惩罚 (默认: 1)")
    
    # ===== 新增：认知模块配置参数 =====
    parser.add_argument("--use_cognitive_modules", action="store_true",
                       help="启用认知模块 (默认禁用)")
    parser.add_argument("--use_cognitive_bias", action="store_true",
                       help="启用认知偏差模块 (默认禁用)")
    parser.add_argument("--bias_visual_aversion", action="store_true",
                       help="认知偏差模块启用视觉厌恶 (默认启用)")
    parser.add_argument("--bias_visual_distance", type=float, default=50.0,
                       help="认知偏差模块视觉距离 (默认: 50.0)")
    parser.add_argument("--bias_inverse_tta_coef", type=float, default=1.5,
                       help="认知偏差模块looming penalty系数 c (默认: 1.5)")
    parser.add_argument("--bias_tta_threshold", type=float, default=0.1,
                       help="认知偏差模块TTA阈值 (默认: 0.1)")
    parser.add_argument("--use_cognitive_delay", action="store_true",
                       help="启用认知延迟模块 (默认禁用)")
    parser.add_argument("--delay_steps", type=int, default=2,
                       help="认知延迟模块延迟步数 (默认: 2)")  # 一个step是0.1s
    parser.add_argument("--use_cognitive_perception", action="store_true",
                       help="启用认知感知模块 (默认禁用)")
    parser.add_argument("--perception_noise_std", type=float, default=0.01,
                       help="认知感知模块观测噪声标准差 (默认: 0.01)")
    parser.add_argument("--perception_attention_bias", type=float, default=0.1,
                       help="认知感知模块注意力偏置 (默认: 0.1)")
    parser.add_argument("--perception_enable_lidar_noise", action="store_true",
                       help="认知感知模块启用激光雷达噪声 (默认启用)")
    parser.add_argument("--perception_enable_state_noise", action="store_true",
                       help="认知感知模块启用状态噪声 (默认启用)")
    parser.add_argument("--perception_enable_attention_bias", action="store_true",
                       help="认知感知模块启用注意力偏置 (默认启用)")
    
    # ===== 新增：速度控制奖励配置参数 =====
    parser.add_argument("--use_speed_control_reward", action="store_true",
                       help="启用速度控制奖励 (默认禁用)")
    parser.add_argument("--speed_control_k", type=float, default=1.0,
                       help="速度跟踪系数 (默认: 1.0)")
    parser.add_argument("--speed_control_kappa", type=float, default=0.5,
                       help="超速软墙系数 (默认: 0.5)")
    parser.add_argument("--speed_control_mu", type=float, default=0.3,
                       help="超速刹车奖励系数 (默认: 0.3)")
    parser.add_argument("--speed_control_nu", type=float, default=0.2,
                       help="超速加速惩罚系数 (默认: 0.2)")
    parser.add_argument("--speed_control_v_tolerance", type=float, default=1.0,
                       help="速度跟踪容差 (默认: 1.0)")
    parser.add_argument("--speed_control_v_ref", type=float, default=15.0,
                       help="目标参考速度 (默认: 15.0)")
    parser.add_argument("--speed_control_enable_tracking", action="store_true",
                       help="启用速度跟踪子模块 (默认禁用)")
    parser.add_argument("--speed_control_enable_soft_wall", action="store_true",
                       help="启用超速软墙子模块 (默认禁用)")
    parser.add_argument("--speed_control_enable_behavior_guidance", action="store_true",
                       help="启用行为导向子模块 (默认禁用)")
    
    # ===== 新增：认知可视化配置参数 =====
    parser.add_argument("--enable_cognitive_viz", action="store_true",
                       help="启用认知模块可视化 (默认禁用)")
    parser.add_argument("--enable_radar_beam_viz", action="store_true",
                       help="启用雷达束可视化 (默认禁用)")
    
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["auto", "cpu", "cuda"],
                       help="计算设备")
    
    args = parser.parse_args()
    
    # 检查检查点文件
    if not os.path.exists(args.checkpoint):
        print(f"❌ 检查点文件不存在: {args.checkpoint}")
        sys.exit(1)
    

    # 创建仿真控制器
    simulator = PPOCheckpointSimulator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        args=args  # 🔧 新增：传递命令行参数
    )
    
    # 🚀 新增：更新速度控制配置
    if args.use_speed_control_reward:
        # 只更新MetaDrive兼容的配置键
        simulator.config.update({
            "use_speed_control_reward": True
        })
        print("🚀 速度控制奖励配置已更新")
        print(f"   📊 速度跟踪系数: {args.speed_control_k}")
        print(f"   🧱 超速软墙系数: {args.speed_control_kappa}")
        print(f"   🎯 行为导向系数: μ={args.speed_control_mu}, ν={args.speed_control_nu}")
        print(f"   ⚡ 目标速度: {args.speed_control_v_ref} m/s")
        print(f"   📏 速度容差: {args.speed_control_v_tolerance} m/s")
        print(f"   ✅ 子模块状态: 跟踪={args.speed_control_enable_tracking}, 软墙={args.speed_control_enable_soft_wall}, 行为={args.speed_control_enable_behavior_guidance}")
        print("   📝 注意：速度控制参数将在环境创建时通过命令行参数设置")
    
    if args.evaluate:
        # 运行模型评估
        evaluation = simulator.evaluate_model(
            num_episodes=args.eval_episodes,
            render=not args.no_render
        )
        
        print(f"\n📊 评估结果总结:")
        print(f"🏆 成功率: {evaluation['success_rate']:.1%}")
        print(f"💰 平均奖励: {evaluation['avg_reward']:.2f} ± {evaluation['std_reward']:.2f}")
        print(f"📏 平均Episode长度: {evaluation['avg_episode_length']:.1f}")
        print(f"🎯 平均路径完成度: {evaluation['avg_path_completion']:.1%}")
        
        # 🔧 新增：变道统计评估输出
        print(f"🚗 总变道次数: {evaluation['total_lane_changes']}")
        print(f"💸 平均变道惩罚: {evaluation['avg_lane_change_penalty']:.3f}")
        print(f"⚡ 平均变道速度比: {evaluation['avg_lane_change_speed_ratio']:.3f}")
        print(f"⏰ 总冷却期违规: {evaluation['total_cooldown_violations']}")
        
    else:
        # 运行常规仿真
        simulator.run_simulation(
            num_episodes=args.episodes,
            render=not args.no_render,
            max_steps=args.max_steps,
            deterministic=not args.stochastic
        )
    
    print(f"\n✅ 仿真完成！")
    



if __name__ == "__main__":
    main() 