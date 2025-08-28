#!/usr/bin/env python3
"""
MetaDrive PPO Expert 复现训练系统
严格对齐MetaDrive PPO expert的配置，仅关键超参数可调整
支持TensorBoard可视化、完整产物落地和详细说明文档生成
可实现断点续训、读取ckpt结合新参数训
集成认知模块：认知偏差、认知延迟、认知感知
"""

import os
import sys
import json
import argparse
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from typing import Dict, Any, Tuple, List
from collections import deque
import csv
import time
from pathlib import Path

# 添加Stable Baselines3向量化环境支持
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# 添加metadrive到路径
current_dir = Path(__file__).parent.absolute()
metadrive_root = current_dir.parent.parent.parent
sys.path.insert(0, str(metadrive_root))


from metadrive.envs.metadrive_env import MetaDriveEnv
# from metadrive.obs.state_obs import LidarStateObservation
from torch.utils.tensorboard import SummaryWriter

# 导入认知模块
from cognitive_module.cognitive_bias_module import CognitiveBiasModule
from cognitive_module.cognitive_delay_module import CognitiveDelayModule
from cognitive_module.cognitive_perception_module import CognitivePerceptionModule
from cognitive_module.cognitive_parameter_sampler import CognitiveParameterSampler
# === 新增：导入离散认知参数采样器 ===
from cognitive_module.discrete_cognitive_parameter_sampler import DiscreteCognitiveParameterSampler


class SpeedControlMetaDriveEnv(MetaDriveEnv):
    def __init__(self, config, args=None):
        # 提取速度控制参数 - 优先使用config中的值，如果没有则使用命令行参数的默认值
        if args is not None:
            self.k = config.get("speed_control_k", args.speed_control_k)
            self.kappa = config.get("speed_control_kappa", args.speed_control_kappa)
            self.mu = config.get("speed_control_mu", args.speed_control_mu)
            self.nu = config.get("speed_control_nu", args.speed_control_nu)
            self.v_tol = config.get("speed_control_v_tolerance", args.speed_control_v_tolerance)
            self.v_ref = config.get("speed_control_v_ref", args.speed_control_v_ref)
            
            # 提取子模块启用状态参数
            self.enable_tracking = config.get("speed_control_enable_tracking", args.speed_control_enable_tracking)
            self.enable_soft_wall = config.get("speed_control_enable_soft_wall", args.speed_control_enable_soft_wall)
            self.enable_behavior_guidance = config.get("speed_control_enable_behavior_guidance", args.speed_control_enable_behavior_guidance)
        else:
            # 向后兼容：如果没有传入args，使用硬编码的默认值
            self.k = config.get("speed_control_k", 1.0)
            self.kappa = config.get("speed_control_kappa", 0.5)
            self.mu = config.get("speed_control_mu", 0.3)
            self.nu = config.get("speed_control_nu", 0.2)
            self.v_tol = config.get("speed_control_v_tolerance", 1.0)
            self.v_ref = config.get("speed_control_v_ref", 10.0)
            
            self.enable_tracking = config.get("speed_control_enable_tracking", True)
            self.enable_soft_wall = config.get("speed_control_enable_soft_wall", True)
            self.enable_behavior_guidance = config.get("speed_control_enable_behavior_guidance", True)
        
        # 创建MetaDrive兼容的配置（移除所有自定义参数）
        metadrive_config = config.copy()
        speed_control_keys = [
            "use_speed_control_reward", "speed_control_k", "speed_control_kappa",
            "speed_control_mu", "speed_control_nu", "speed_control_v_tolerance", "speed_control_v_ref",
            "speed_control_enable_tracking", "speed_control_enable_soft_wall", "speed_control_enable_behavior_guidance"
        ]
        for key in speed_control_keys:
            if key in metadrive_config:
                del metadrive_config[key]
        
        # 调用父类构造函数
        super().__init__(metadrive_config)
        
        # dt：尽量从引擎或配置取（下面只是兜底）
        self._dt = getattr(self.engine, "controller_step_interval", None) \
                   or getattr(self, "control_interval", None) \
                   or 0.05

        self._last_speed = None

        # 建议在外部 config 中关闭原速度奖励，例如：
        # config["reward_config"]["speed_reward"] = 0.0

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self._last_speed = None
        return obs

    @staticmethod
    def _huber(x, delta):
        ax = abs(x)
        return 0.5*ax*ax if ax <= delta else delta*(ax - 0.5*delta)

    def _compute_speed_control_reward(self, vehicle, action):
        import math

        v = float(vehicle.speed)  # m/s
        # 用速度差商估计加速度，更稳
        if self._last_speed is None:
            a = 0.0
        else:
            a = (v - self._last_speed) / self._dt
        a_pos = max(a, 0.0)
        a_neg = max(-a, 0.0)

        dv = v - self.v_ref
        is_over = 1.0 if dv > 0.0 else 0.0

        # === 子模块奖励计算（根据启用状态） ===
        r_track = 0.0
        r_wall = 0.0
        r_act_over = 0.0

        # A: 目标速度跟踪子模块
        if self.enable_tracking:
            r_track = -self.k * self._huber(dv, self.v_tol)

        # B1: 速度软墙子模块
        if self.enable_soft_wall:
            # softplus = log(1+exp(x))；数值稳定可加裁剪
            dv_clip = max(min(dv, 20.0), -20.0)
            softp = math.log1p(math.exp(dv_clip))
            r_wall = -self.kappa * (softp ** 2) * is_over

        # B2: 行为导向子模块
        if self.enable_behavior_guidance:
            # 超速区动作导向（鼓励减速、惩罚继续加速）
            r_act_over = is_over * (self.mu * a_neg - self.nu * a_pos)

        r_total = r_track + r_wall + r_act_over

        # 记录供下一步使用
        self._last_speed = v

        # 可选：把各分量写进 step_infos，供 info 查看
        vid = vehicle.id
        if hasattr(self, "step_infos"):
            self.step_infos.setdefault(vid, {})
            self.step_infos[vid].update({
                "sc_r_total": r_total,
                "sc_r_track": r_track,
                "sc_r_wall": r_wall,
                "sc_r_act_over": r_act_over,
                "sc_v": v, "sc_v_ref": self.v_ref, "sc_dv": dv, "sc_a": a,
                # === 新增：子模块启用状态记录 ===
                "sc_enable_tracking": self.enable_tracking,
                "sc_enable_soft_wall": self.enable_soft_wall,
                "sc_enable_behavior_guidance": self.enable_behavior_guidance
            })

        return r_total

    def reward_function(self, vehicle_id: str) -> float:
        base_reward = super().reward_function(vehicle_id)

        # 注意：这里仍然使用config.get，因为这是环境内部的配置检查
        # 如果需要使用命令行参数，需要在创建环境时传入
        if self.config.get("use_speed_control_reward", False):
            vehicle = self.agents[vehicle_id]
            current_action = getattr(vehicle, 'current_action', [0.0, 0.0])
            sc_reward = self._compute_speed_control_reward(vehicle, current_action)

            # ——重要——
            # 若已在 config 中把原"速度奖励"权重设为 0，则此处可直接相加。
            # 若无法从配置剥离原速度项，这里应显式减去原速度项（需知道原项的实现/权重）。
            return base_reward + sc_reward

        return base_reward



class PPONetwork(nn.Module):
    """PPO网络结构 - 严格对齐MetaDrive expert + 认知参数集成 + 渐进式训练支持"""
    
    def __init__(self, obs_dim: int = 279, action_dim: int = 2, hidden_dim: int = 256, 
                 checkpoint_obs_dim: int = None):
        super(PPONetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.checkpoint_obs_dim = checkpoint_obs_dim
        self.is_progressive_training = checkpoint_obs_dim == 275 and obs_dim == 279
        
        # Actor网络 (观测维度从275扩展到279，包含认知参数)
        self.actor_fc1 = nn.Linear(obs_dim, hidden_dim)      # 279 → 256
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)    # 256 → 256
        self.actor_out = nn.Linear(hidden_dim, action_dim * 2) # 256 → 4
        
        # Critic网络 (观测维度从275扩展到279，包含认知参数)
        self.critic_fc1 = nn.Linear(obs_dim, hidden_dim)      # 279 → 256
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)   # 256 → 256
        self.critic_out = nn.Linear(hidden_dim, 1)            # 256 → 1
        
        # 激活函数
        self.tanh = nn.Tanh()
        
        # 渐进式训练相关属性
        self.training_stage = 1  # 1: 冻结阶段, 2: 解冻阶段
        self.freeze_threshold_steps = 3000000  # 3M步切换点
        
        # 初始化权重
        self._init_weights()
        
        # 如果是从275维检查点恢复，设置冻结状态
        if self.is_progressive_training:
            self._setup_progressive_training()
    
    def _init_weights(self):
        # 隐层：正交 + gain=1.0（tanh稳定）
        for layer in [self.actor_fc1, self.actor_fc2, self.critic_fc1, self.critic_fc2]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)

        # critic 输出正常尺度
        nn.init.orthogonal_(self.critic_out.weight, gain=1.0)
        nn.init.constant_(self.critic_out.bias, 0.0)

        # actor 输出降温：小得多的增益，避免一开始把动作打到饱和
        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)
        nn.init.constant_(self.actor_out.bias, 0.0)

        # 给"均值"的 throttle 维度一个小正偏置；给 log_std 维度一个较小的初值
        # actor_out 的前 action_dim 是 mean，后 action_dim 是 log_std
        with torch.no_grad():
            action_dim = self.actor_out.out_features // 2
            # log_std 初值更稳一些（比如 -0.5）
            self.actor_out.bias[action_dim:].fill_(-1.0)
            # throttle/brake 是动作的第2维（索引1）：给 mean 一个+0.3 的轻微偏置，鼓励先动起来
            self.actor_out.bias[1] = 0.1

    def forward(self, obs):
        """前向传播"""
        # Actor前向
        x_actor = self.tanh(self.actor_fc1(obs))
        x_actor = self.tanh(self.actor_fc2(x_actor))
        action_logits = self.actor_out(x_actor)
        
        # Critic前向
        x_critic = self.tanh(self.critic_fc1(obs))
        x_critic = self.tanh(self.critic_fc2(x_critic))
        value = self.critic_out(x_critic)
        
        return action_logits, value
    
    
    def get_action_and_value(self, obs, action=None, eps: float = 1e-6):
        """使用 tanh-squashed Gaussian，确保环境执行的动作与计算log_prob完全一致"""
        action_logits, value = self.forward(obs)
        action_mean, action_log_std = torch.chunk(action_logits, 2, dim=-1)
        action_log_std = torch.clamp(action_log_std, -5.0, 2.0)
        action_std = torch.exp(action_log_std)

        base_dist = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            # 训练时用 reparameterization 更稳定
            u = base_dist.rsample()
        else:
            # 来自回放/旧动作：它已经是 tanh 后的 a，需要反挤压回 u=atanh(a)
            a = action.clamp(-1 + eps, 1 - eps)
            u = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh(a)

        a = torch.tanh(u)

        # log_prob 需要加上 tanh 的雅可比修正项
        log_prob = base_dist.log_prob(u) - torch.log(1 - a.pow(2) + eps)
        log_prob = log_prob.sum(dim=-1)

        # 熵用 base_dist 的熵作近似（足够做熵正则/监控）
        entropy = base_dist.entropy().sum(dim=-1)

        return a, log_prob, entropy, value.squeeze(-1)
    
    def act_deterministic(self, obs_tensor):
        """ 修复2: 为评估提供确定性动作（使用均值）"""
        action_logits, _ = self.forward(obs_tensor)
        action_mean, _ = torch.chunk(action_logits, 2, dim=-1)
        return torch.tanh(action_mean)
    
    def get_action_stats(self, obs_tensor):
        """获取动作统计信息（均值）"""
        action_logits, _ = self.forward(obs_tensor)
        action_mean, _ = torch.chunk(action_logits, 2, dim=-1)
        # 应用tanh变换
        action_tanh = torch.tanh(action_mean)
        # 返回steer和throttle的均值
        return action_tanh[:, 0], action_tanh[:, 1]  # steer, throttle
    
    def _setup_progressive_training(self):
        """设置渐进式训练 - 冻结前275维权重"""
        if not self.is_progressive_training:
            return
            
        print("🧠 设置渐进式训练模式：")
        print(f"   检查点维度: {self.checkpoint_obs_dim}")
        print(f"   目标维度: {self.obs_dim}")
        print(f"   冻结前275维权重，仅训练新增4维和下游层")
        
        # 冻结前275维权重（actor_fc1和critic_fc1的前275维）
        with torch.no_grad():
            # 冻结actor_fc1的前275维权重
            self.actor_fc1.weight[:, :275].requires_grad_(False)
            print(f"   ✅ 已冻结actor_fc1前275维权重")
            
            # 冻结critic_fc1的前275维权重
            self.critic_fc1.weight[:, :275].requires_grad_(False)
            print(f"   ✅ 已冻结critic_fc1前275维权重")
        
        # 设置训练阶段
        self.training_stage = 1
        print(f"   🎯 当前训练阶段: {self.training_stage} (冻结阶段)")
    
    def update_training_stage(self, global_step: int):
        """根据训练步数更新训练阶段"""
        if not self.is_progressive_training:
            return
            
        if self.training_stage == 1 and global_step >= self.freeze_threshold_steps:
            self.training_stage = 2
            self._unfreeze_weights()
            print(f"🎯 训练阶段切换: {self.training_stage} (解冻阶段)")
            print(f"   已解冻前275维权重，允许学习认知参数协同作用")
    
    def _unfreeze_weights(self):
        """解冻前275维权重"""
        if not self.is_progressive_training:
            return
            
        # 解冻前275维权重
        with torch.no_grad():
            # 解冻actor_fc1的前275维权重
            self.actor_fc1.weight[:, :275].requires_grad_(True)
            
            # 解冻critic_fc1的前275维权重
            self.critic_fc1.weight[:, :275].requires_grad_(True)
        
        print(f"   ✅ 已解冻前275维权重")
    
    def _load_and_extend_weights(self, checkpoint_state_dict):
        """从275维检查点加载权重并扩展到279维"""
        print("🔧 正在扩展275维权重到279维...")
        
        # 创建新的状态字典
        new_state_dict = {}
        
        for key, value in checkpoint_state_dict.items():
            if key in ['actor_fc1.weight', 'critic_fc1.weight']:
                # 对于输入层，需要扩展权重
                if key == 'actor_fc1.weight':
                    # 原始: [256, 275], 目标: [256, 279]
                    new_weight = torch.zeros(256, 279, device=value.device)
                    new_weight[:, :275] = value  # 复制前275维权重
                    # 初始化新增4维权重（使用较小的随机值）
                    torch.nn.init.orthogonal_(new_weight[:, 275:], gain=0.01)
                    new_state_dict[key] = new_weight
                    print(f"   ✅ 已扩展actor_fc1权重: 275 → 279")
                    
                elif key == 'critic_fc1.weight':
                    # 原始: [256, 275], 目标: [256, 279]
                    new_weight = torch.zeros(256, 279, device=value.device)
                    new_weight[:, :275] = value  # 复制前275维权重
                    # 初始化新增4维权重
                    torch.nn.init.orthogonal_(new_weight[:, 275:], gain=0.01)
                    new_state_dict[key] = new_weight
                    print(f"   ✅ 已扩展critic_fc1权重: 275 → 279")
            else:
                # 其他层直接复制
                new_state_dict[key] = value
        
        # 返回扩展后的状态字典
        print("✅ 权重扩展完成")
        return new_state_dict
    
    def _load_and_truncate_weights(self, checkpoint_state_dict):
        """从279维检查点截取前275维权重"""
        print("🔧 正在截取279维权重到275维...")
        
        # 创建新的状态字典
        new_state_dict = {}
        
        for key, value in checkpoint_state_dict.items():
            if key in ['actor_fc1.weight', 'critic_fc1.weight']:
                # 对于输入层，需要截取权重
                if key == 'actor_fc1.weight':
                    # 原始: [256, 279], 目标: [256, 275]
                    new_weight = value[:, :275]
                    new_state_dict[key] = new_weight
                    print(f"   ✅ 已截取actor_fc1权重: 279 → 275")
                    
                elif key == 'critic_fc1.weight':
                    # 原始: [256, 279], 目标: [256, 275]
                    new_weight = value[:, :275]
                    new_state_dict[key] = new_weight
                    print(f"   ✅ 已截取critic_fc1权重: 279 → 275")
            else:
                # 其他层直接复制
                new_state_dict[key] = value
        
        # 返回截取后的状态字典
        print("✅ 权重截取完成")
        return new_state_dict
    
    def get_frozen_weight_stats(self):
        """获取冻结权重的统计信息"""
        if not self.is_progressive_training:
            return {}
            
        stats = {}
        
        # 获取前275维权重的统计信息
        with torch.no_grad():
            # Actor网络前275维权重统计
            actor_frozen_weights = self.actor_fc1.weight[:, :275]
            stats['actor_frozen_mean'] = actor_frozen_weights.mean().item()
            stats['actor_frozen_std'] = actor_frozen_weights.std().item()
            stats['actor_frozen_norm'] = torch.norm(actor_frozen_weights).item()
            
            # Critic网络前275维权重统计
            critic_frozen_weights = self.critic_fc1.weight[:, :275]
            stats['critic_frozen_mean'] = critic_frozen_weights.mean().item()
            stats['critic_frozen_std'] = critic_frozen_weights.std().item()
            stats['critic_frozen_norm'] = torch.norm(critic_frozen_weights).item()
            
            # 新增4维权重统计
            actor_new_weights = self.actor_fc1.weight[:, 275:]
            stats['actor_new_mean'] = actor_new_weights.mean().item()
            stats['actor_new_std'] = actor_new_weights.std().item()
            stats['actor_new_norm'] = torch.norm(actor_new_weights).item()
            
            critic_new_weights = self.critic_fc1.weight[:, 275:]
            stats['critic_new_mean'] = critic_new_weights.mean().item()
            stats['critic_new_std'] = critic_new_weights.std().item()
            stats['critic_new_norm'] = torch.norm(critic_new_weights).item()
        
        return stats


def make_env(rank: int, config: Dict[str, Any], args):
    """
    环境工厂函数 - 用于创建向量化环境
    每个子进程将运行独立的MetaDrive环境实例
    支持直线场景的动态生成，确保每个环境都有不同的道路长度和交通配置
    
    Args:
        rank: 环境索引
        config: 环境配置字典
        args: 命令行参数对象，包含所有add_arguments的默认值
    
    Returns:
        环境创建函数
    """
    def _init():
        # 为每个环境设置不同的随机种子，确保样本多样性
        env_config = config.copy()
        base_seed = config.get("start_seed", 0)
        env_seed = base_seed + rank * 10000
        env_config["start_seed"] = env_seed
        
        # 为每个环境生成不同的直线场景参数
        scenario_index = env_seed % 1000  # 确保在0-999范围内
        
        # 动态直线道路长度：每个环境不同数量的S段
        min_segments, max_segments = 10, 12
        num_segments = min_segments + (scenario_index * (max_segments - min_segments)) // 1000
        map_string = "S" * num_segments
        
        # 动态交通密度：每个环境不同（使用命令行参数的默认值）
        # 注意：交通密度会在训练过程中通过课程学习动态调整
        min_density, max_density = config.get("traffic_density_min", args.traffic_density_min), config.get("traffic_density_max", args.traffic_density_max)
        base_traffic_density = config.get("traffic_density", min_density)  # 使用配置中的基础值
        
        # 为了保持环境多样性，每个环境在基础密度上增加小的扰动
        density_perturbation = (scenario_index % 100) / 1000.0 * 0.02  # 最大2%的扰动
        traffic_density = base_traffic_density + density_perturbation
        
        # 更新环境配置
        env_config["map"] = map_string
        env_config["traffic_density"] = traffic_density
        
        # 确保子进程环境不使用渲染（避免显示冲突）
        # 强制覆盖渲染相关参数，确保多进程环境安全
        env_config["use_render"] = False
        env_config["debug"] = False
        env_config["image_observation"] = False
        
        # 只保留MetaDrive真正必需的渲染参数
        # 其他参数让MetaDrive使用默认值
        
        # 移除自定义参数（MetaDrive不识别的参数）
        custom_params = ["traffic_density_min", "traffic_density_max"]
        for param in custom_params:
            if param in env_config:
                del env_config[param]
        
        # 调试：打印关键配置参数
        print(f" 环境{rank}配置验证: use_render={env_config.get('use_render')}, image_observation={env_config.get('image_observation')}")
        
        # 根据配置选择环境类
        if env_config.get("use_speed_control_reward", False):
            # 使用自定义的速度控制环境
            # 从命令行参数获取速度控制参数
            speed_control_config = env_config.copy()
            speed_control_config.update({
                "speed_control_k": config.get("speed_control_k", args.speed_control_k),  # 使用命令行参数默认值
                "speed_control_kappa": config.get("speed_control_kappa", args.speed_control_kappa),  # 使用命令行参数默认值
                "speed_control_mu": config.get("speed_control_mu", args.speed_control_mu),  # 使用命令行参数默认值
                "speed_control_nu": config.get("speed_control_nu", args.speed_control_nu),  # 使用命令行参数默认值
                "speed_control_v_tolerance": config.get("speed_control_v_tolerance", args.speed_control_v_tolerance),  # 使用命令行参数默认值
                "speed_control_v_ref": config.get("speed_control_v_ref", args.speed_control_v_ref),  # 使用命令行参数默认值
                # === 新增：子模块启用控制参数 ===
                "speed_control_enable_tracking": config.get("speed_control_enable_tracking", args.speed_control_enable_tracking),  # 使用命令行参数默认值
                "speed_control_enable_soft_wall": config.get("speed_control_enable_soft_wall", args.speed_control_enable_soft_wall),  # 使用命令行参数默认值
                "speed_control_enable_behavior_guidance": config.get("speed_control_enable_behavior_guidance", args.speed_control_enable_behavior_guidance),  # 使用命令行参数默认值
            })
            env = SpeedControlMetaDriveEnv(speed_control_config, args)
            
            # 输出子模块启用状态信息
            tracking_status = "✅" if speed_control_config["speed_control_enable_tracking"] else "❌"
            soft_wall_status = "✅" if speed_control_config["speed_control_enable_soft_wall"] else "❌"
            behavior_status = "✅" if speed_control_config["speed_control_enable_behavior_guidance"] else "❌"
            
            print(f" 环境{rank}: 使用SpeedControlMetaDriveEnv (速度控制奖励已启用)")
            print(f"   子模块状态: 跟踪{tracking_status} 软墙{soft_wall_status} 行为{behavior_status}")
        else:
            # 使用标准MetaDrive环境
            env = MetaDriveEnv(env_config)
            print(f" 环境{rank}: 使用MetaDriveEnv (标准奖励)")
        
        # 为环境添加动态更新交通密度的能力
        def update_traffic_density(new_density):
            """动态更新交通密度"""
            if hasattr(env, 'config'):
                env.config['traffic_density'] = new_density + density_perturbation
                # 如果环境支持运行时配置更新，在这里实现
                
        env.update_traffic_density = update_traffic_density
        
        return env
    
    return _init


class PPOExpertReproduction:
    """PPO Expert复现训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        
        # 设置随机种子
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # 创建实验目录
        self.exp_dir = self._create_experiment_dir()
        
        # 保存配置
        self.config = self._build_config()
        self._save_config()
        
        # 初始化训练统计 - 必须在创建环境之前初始化
        self.global_step = 0
        self.episode_count = 0
        self.train_stats = []
        
        # 存储交通密度参数供其他方法使用
        self.traffic_density_min = args.traffic_density_min if args.traffic_density_min > 0 else 0.1
        self.traffic_density_max = args.traffic_density_max if args.traffic_density_max > 0 else 0.15
        
        # === 检测并记录交通密度采样模式 ===
        if (hasattr(args, 'resume_from') and args.resume_from and 
            args.traffic_density_min > 0 and args.traffic_density_max > 0):
            print(f"🎯 恢复训练模式 - 交通密度采样:")
            print(f"   模式: 随机采样（覆盖课程学习）")
            print(f"   密度范围: [{args.traffic_density_min:.3f}, {args.traffic_density_max:.3f}]")
            self.traffic_density_mode = "random_sampling"
        else:
            print(f"🎯 交通密度采样:")
            print(f"   模式: 课程学习（默认/未指定密度参数）") 
            print(f"   默认范围: [{self.traffic_density_min:.3f}, {self.traffic_density_max:.3f}]")
            self.traffic_density_mode = "curriculum_learning"
        
        # 课程学习状态 - 必须在创建环境之前初始化
        self.use_curriculum = self.args.use_curriculum
        self.curriculum_mode = self.args.curriculum_mode
        self.curriculum_alpha = self.args.curriculum_alpha
        self.curriculum_stage = 0  # gate模式使用；progress模式按进度算，不用这个
        
        # === 初始化认知模块 ===
        self.use_cognitive_modules = args.use_cognitive_modules
        self.cognitive_bias_module = None
        self.cognitive_delay_module = None
        self.cognitive_perception_module = None
        self.cognitive_parameter_sampler = None
        
        if self.use_cognitive_modules:
            print("易 初始化认知模块...")
            
            # 先初始化认知感知模块（因为认知偏差模块需要引用它）
            if args.use_cognitive_perception:
                perception_config = {
                    'sigma0': args.perception_sigma0 * 10,  # 转换为米制噪声
                    'k': args.perception_k,
                    'p_miss0': args.perception_p_miss0,
                    'far_distance': 50.0,
                    'p_false': args.perception_p_false,
                    'use_ar1': True,  # 启用AR(1)过程
                    'rho': 0.8,       # AR(1)相关系数
                    'use_kf': args.perception_use_kf,
                    'kf_dt': args.perception_kf_dt,
                    'kf_q_scale': args.perception_kf_q_scale
                }
                self.cognitive_perception_module = CognitivePerceptionModule(perception_config)
                
                # 启用雷达束可视化（如果指定）
                if getattr(args, 'enable_radar_beam_viz', False):
                    self.cognitive_perception_module.enable_radar_visualization(True)
                
                print(f"   ✅ 认知感知模块已启用")
            
            # 初始化认知偏差模块（传入认知感知模块引用）
            if args.use_cognitive_bias:
                bias_config = {
                    'inverse_tta_coef': args.bias_inverse_tta_coef,
                    'tta_threshold': args.bias_tta_threshold,
                    'adaptive_bias': args.bias_adaptive,
                    'adaptation_rate': args.bias_adaptation_rate,
                    'visual_detection_distance': args.bias_visual_distance,
                    'visual_detection_angle': args.bias_visual_angle,
                    'visual_aversion_strength': args.bias_visual_strength
                }
                #  关键改进：传入认知感知模块引用
                self.cognitive_bias_module = CognitiveBiasModule(
                    bias_config=bias_config,
                    cognitive_perception_module=self.cognitive_perception_module
                )
                print(f"   ✅ 认知偏差模块已启用")
                if self.cognitive_perception_module:
                    print(f"       已连接到认知感知模块")
            
            # 初始化认知延迟模块
            if args.use_cognitive_delay:
                self.cognitive_delay_module = CognitiveDelayModule(
                    delay_steps=int(args.delay_steps),  # 确保是整数类型
                    enable_smoothing=args.delay_smoothing,
                    smoothing_factor=args.delay_smoothing_factor,
                    enable_visualization=args.cognitive_visualization
                )
                print(f"   ✅ 认知延迟模块已启用 (延迟{args.delay_steps}步)")
            
            # === 新增：初始化认知参数采样器 ===
            if args.use_cognitive_parameter_sampling:
                # 根据参数选择采样器类型
                if args.cognitive_sampler_type == "discrete":
                    print("   使用离散认知参数采样器")
                    self.cognitive_parameter_sampler = DiscreteCognitiveParameterSampler(
                        update_steps=args.cognitive_param_update_steps,
                        bias_inverse_tta_coef_range=args.bias_inverse_tta_coef_range,
                        perception_sigma0_range=args.perception_sigma0_range,
                        perception_k_range=args.perception_k_range,
                        delay_steps_range=args.delay_steps_range,
                        # === 新增：离散采样密度参数 ===
                        bias_inverse_tta_coef_density=args.bias_inverse_tta_coef_density,
                        perception_sigma0_density=args.perception_sigma0_density,
                        perception_k_density=args.perception_k_density,
                        delay_steps_density=args.delay_steps_density,
                        enable_visualization=args.cognitive_visualization,
                        save_history=True
                    )
                elif args.cognitive_sampler_type == "continuous":
                    print("   使用连续认知参数采样器")
                    self.cognitive_parameter_sampler = CognitiveParameterSampler(
                        update_steps=args.cognitive_param_update_steps,
                        bias_inverse_tta_coef_range=args.bias_inverse_tta_coef_range,
                        perception_sigma0_range=args.perception_sigma0_range,
                        perception_k_range=args.perception_k_range,
                        delay_steps_range=args.delay_steps_range,
                        enable_visualization=args.cognitive_visualization,
                        save_history=True
                    )
                else:
                    raise ValueError(f"不支持的认知采样器类型: {args.cognitive_sampler_type}")

            else:
                self.cognitive_parameter_sampler = None

        
        # === 认知可视化数据收集 ===
        self.enable_cognitive_visualization = args.cognitive_visualization
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
            print(f" 认知可视化: 已启用")
        else:
            self.cognitive_viz_data = None
        
        # 创建环境
        self.envs = self._create_environments()
        
        # === 认知模块：附加到环境 ===
        if self.use_cognitive_modules:
            # 对于向量化环境，附加到第一个环境实例（用于可视化等）
            try:
                if hasattr(self.envs, 'envs') and len(self.envs.envs) > 0:
                    self._attach_cognitive_modules_to_env(self.envs.envs[0])
                elif hasattr(self.envs, 'venv') and hasattr(self.envs.venv, 'envs'):
                    if len(self.envs.venv.envs) > 0:
                        self._attach_cognitive_modules_to_env(self.envs.venv.envs[0])
            except Exception as e:
                print(f"⚠️ 认知模块附加失败: {e}")
        
        
        # 渐进式训练相关属性
        self.checkpoint_obs_dim = None  # 将在load_checkpoint中设置
        self.is_progressive_training = False  # 是否启用渐进式训练
        
        # 创建网络 - 根据是否启用认知模块动态设置观测维度
        # 注意：如果是从检查点恢复，网络维度将在load_checkpoint中确定
        if args.resume_from:
            # 延迟创建网络，在load_checkpoint中根据检查点维度创建
            self.network = None
            self.optimizer = None
        else:
            # 正常模式：创建网络
            obs_dim = 279 if self.use_cognitive_modules else 275
            self.network = PPONetwork(obs_dim=obs_dim).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=args.lr)
        
        # 学习率调度参数缓存
        self.lr_init = self.args.lr              # 初始学习率
        self.lr_min = self.args.lr_min           # 衰减的下限
        self.warmup_ratio = self.args.warmup_ratio
        self.lr_schedule = self.args.lr_schedule

        # 验证和记录直线场景生成
        self._validate_and_log_scenarios()
        
        # 创建TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "tensorboard"))
        
        # 恢复训练逻辑
        if args.resume_from:
            self.load_checkpoint(args.resume_from)
            print(f"✅ 从检查点恢复训练完成")
        
        # 熵系数衰减设置
        self.entropy_coef_start = args.entropy_coef_start
        self.entropy_coef_end = args.entropy_coef_end
        self.entropy_decay_end_ratio = args.entropy_decay_end_ratio
        self.current_entropy_coef = self.entropy_coef_start  # 当前熵系数
        
        # 添加episode统计缓冲区
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_speeds = deque(maxlen=100)
        self.episode_lane_deviations = deque(maxlen=100)
        self.episode_lane_changes = deque(maxlen=100)
        self.episode_min_ttcs = deque(maxlen=100)
        self.episode_path_completions = deque(maxlen=100)
        self.episode_timeouts = deque(maxlen=100)
        
        #  新增：动作统计缓冲区
        self.episode_steer_means = deque(maxlen=100)
        self.episode_throttle_means = deque(maxlen=100)
        
        # === 新增：初始化车道变更检测相关属性 ===
        self._last_lane_index = {}  # 存储每个agent的上一个车道索引
        self.debug_lane_change = False  # 车道变更调试开关
        

        # 创建CSV日志
        self.csv_path = os.path.join(self.exp_dir, "training_logs.csv")
        self._init_csv_log()
        
        
        print(f" PPO Expert复现训练初始化完成")
        print(f" 实验目录: {self.exp_dir}")
        print(f" 设备: {self.device}")
        print(f" 随机种子: {args.seed}")
        
        # === 渐进式训练信息 ===
        if self.is_progressive_training:
            self.network.training_stage = 1
            print(f"🧠 渐进式训练模式已启用:")
            print(f"   检查点维度: {self.checkpoint_obs_dim}")
            print(f"   目标维度: {self.network.obs_dim}")
            print(f"   冻结阈值: {self.network.freeze_threshold_steps:,} 步")
            print(f"   当前阶段: {self.network.training_stage} (1=冻结, 2=解冻)")
        elif self.checkpoint_obs_dim:
            print(f"📥 从{self.checkpoint_obs_dim}维检查点恢复训练")
        
        # === 认知参数集成调试信息 ===
        if self.use_cognitive_modules:
            print(f"易 认知模块已启用，观测维度已扩展:")
            print(f"   原始观测维度: 275 (Lidar: 240 + State: 35)")
            print(f"   认知参数维度: 4 (bias_coef, sigma0, k, delay)")
            print(f"   扩展后观测维度: 279")
            print(f"   网络结构: 279 → 256 → 256 → 4/1")
            
            if self.cognitive_parameter_sampler:
                sampler_type_name = "离散" if args.cognitive_sampler_type == "discrete" else "连续"
                print(f"   认知参数采样器: 已启用 ({sampler_type_name})")
                print(f"   参数更新频率: {self.cognitive_parameter_sampler.update_steps} 步")
                if args.cognitive_sampler_type == "discrete":
                    print(f"   离散采样密度: 偏差系数={args.bias_inverse_tta_coef_density}, "
                          f"感知sigma0={args.perception_sigma0_density}, "
                          f"感知k={args.perception_k_density}, "
                          f"延迟步数={args.delay_steps_density}")
                print(f"   当前认知参数: {self.cognitive_parameter_sampler.get_current_parameters()}")
            else:
                print(f"   认知参数采样器: 未启用 (使用固定参数)")
        else:
            print(f" 认知模块未启用，使用标准观测维度: 275")
    
  
    
    def _create_experiment_dir(self) -> str:
        """创建实验目录"""
        # 恢复训练时的逻辑
        if hasattr(self.args, 'resume_from') and self.args.resume_from:
            checkpoint_path = Path(self.args.resume_from)
            
            # 检查用户是否显式指定了save_dir
            if self.args.save_dir is not None:
                # 用户指定了新的save_dir，在新位置创建继续训练目录
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                exp_name = f"resumed_training_{timestamp}"
                exp_dir = os.path.join(self.args.save_dir, "runs", exp_name)
                os.makedirs(exp_dir, exist_ok=True)
                os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
                print(f"📁 恢复训练模式 - 使用新指定目录: {exp_dir}")
                print(f"   原检查点路径: {self.args.resume_from}")
                return exp_dir
            else:
                # 用户未指定save_dir，使用原检查点目录
                if checkpoint_path.parent.name == "checkpoints":
                    exp_dir = str(checkpoint_path.parent.parent)
                    print(f"📥 恢复训练模式 - 使用原实验目录: {exp_dir}")
                    return exp_dir
                else:
                    # 兜底：检查点不在标准目录结构中
                    exp_dir = str(checkpoint_path.parent)
                    print(f"📥 恢复训练模式 - 使用检查点所在目录: {exp_dir}")
                    return exp_dir
        
        # 正常模式：创建新实验目录
        if self.args.save_dir is None:
            # 设置默认save_dir
            self.args.save_dir = "/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction"
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"ppo_expert_reproduction_{timestamp}"
        exp_dir = os.path.join(self.args.save_dir, "runs", exp_name)
        
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        
        print(f"📁 实验目录已创建: {exp_dir}")
        return exp_dir
    
    def _compute_scheduled_lr(self):
        # 进度 p ∈ [0,1]
        p = min(1.0, float(self.global_step) / float(self.args.total_timesteps + 1e-8))
        # warmup
        if self.warmup_ratio > 0.0:
            wu = self.warmup_ratio
            if p < wu:
                # 从 0 → lr_init 线性升温
                return self.lr_init * (p / wu)
            # 去掉 warmup 后的重新归一化进度
            p = (p - wu) / max(1e-8, (1.0 - wu))
            p = max(0.0, min(1.0, p))

        if self.lr_schedule == "constant":
            return self.lr_init

        if self.lr_schedule == "linear":
            # 线性从 lr_init → lr_min，80% 进度到达 lr_min
            end_ratio = 0.8
            q = min(1.0, p / end_ratio)
            return self.lr_init + (self.lr_min - self.lr_init) * q

        if self.lr_schedule == "cosine":
            # 余弦退火从 lr_init → lr_min
            import math
            cos_term = 0.5 * (1 + math.cos(math.pi * p))
            return self.lr_min + (self.lr_init - self.lr_min) * cos_term

        if self.lr_schedule == "stage":
            # === 兜底：没开课程时直接返回初始学习率 ===
            if not self.use_curriculum:
                return self.lr_init
            # === 正常分段调度 ===
            if self.curriculum_stage == 0:
                return self.lr_init
            elif self.curriculum_stage == 1:
                return max(self.lr_min, self.lr_init * 0.7)
            elif self.curriculum_stage == 2:
                return max(self.lr_min, self.lr_init * 0.5)
            else:  # stage 3
                return max(self.lr_min, self.lr_init * 0.3)

        # 兜底
        return self.lr_init


    def _build_config(self) -> Dict[str, Any]:
        """构建完整配置"""
        return {
            # ===== 复现设定 =====
            "reproduction_target": "MetaDrive PPO Expert",
            "experiment_name": os.path.basename(self.exp_dir),
            "timestamp": datetime.now().isoformat(),
            "random_seed": self.args.seed,
            "device": str(self.device),
            
            # ===== 系统设置 =====
            "system": {
                "device": str(self.device),
                "seed": self.args.seed,
                "save_dir": self.args.save_dir,
                "resume_from": self.args.resume_from
            },
            
            # ===== 网络结构 (严格对齐expert + 认知参数集成) =====
            "network": {
                "observation_dim": 279,  # 275(原始) + 4(认知参数)
                "action_dim": 2,
                "hidden_dim": 256,
                "activation": "tanh",
                "cognitive_params_integration": True,
                "cognitive_params_dim": 4
            },
            
            # ===== 环境配置 (直线场景生成) =====
            "env_config": {
                "num_scenarios": 1000,
                "map_type": "straight_road",     # 直线道路类型
                "dynamic_road_length": True,     # 启用动态道路长度
                "road_length_range": [500, 1000], # 道路长度范围(米)
                "dynamic_traffic": True,         # 启用动态交通配置
                "traffic_density_range": [self.args.traffic_density_min, self.args.traffic_density_max], # 交通密度范围
                "random_traffic": True,          # 交通随机化
                "horizon": 10000,
                
                # 优化的奖励配置
                "reward_config": {
                    "success_reward": self.args.success_reward,
                    "driving_reward": self.args.driving_reward,
                    "speed_reward": self.args.speed_reward,
                    "use_lateral_reward": self.args.use_lateral_reward,
                    "out_of_road_penalty": self.args.out_of_road_penalty,
                    "crash_vehicle_penalty": self.args.crash_penalty,
                    "crash_object_penalty": self.args.crash_penalty,
                    "crash_sidewalk_penalty": self.args.crash_sidewalk_penalty,
                    
                    # === 速度控制奖励配置 (新增) ===
                    "use_speed_control_reward": self.args.use_speed_control_reward,  # 是否启用速度控制奖励
                    
                    # === 子模块启用控制配置 ===
                    "speed_control_enable_tracking": self.args.speed_control_enable_tracking,      # 启用速度跟踪子模块
                    "speed_control_enable_soft_wall": self.args.speed_control_enable_soft_wall,    # 启用速度软墙子模块
                    "speed_control_enable_behavior_guidance": self.args.speed_control_enable_behavior_guidance, # 启用行为导向子模块
                    
                    # === 子模块参数配置 ===
                    "speed_control_k": self.args.speed_control_k,                    # 速度跟踪奖励系数
                    "speed_control_kappa": self.args.speed_control_kappa,            # 超速软墙惩罚系数
                    "speed_control_mu": self.args.speed_control_mu,                  # 超速刹车奖励系数
                    "speed_control_nu": self.args.speed_control_nu,                  # 超速加速惩罚系数
                    "speed_control_v_tolerance": self.args.speed_control_v_tolerance, # 速度跟踪容差
                    "speed_control_v_ref": self.args.speed_control_v_ref,            # 目标参考速度
                },
                
                # 终止条件配置
                "termination_config": {
                    "out_of_road_done": True,
                    "crash_vehicle_done": True,
                    "crash_object_done": True,
                    "on_continuous_line_done": False,
                    "on_broken_line_done": False
                },
                
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
                    },
                    "random_agent_model": False
                }
            },
            
            # ===== 关键超参数 (可调整) =====
            "hyperparameters": {
                "learning_rate": self.args.lr,
                "n_steps": self.args.n_steps,
                "n_envs": self.args.n_envs,
                "batch_size": self.args.batch_size,
                "n_epochs": self.args.n_epochs,
                "gamma": self.args.gamma,
                "gae_lambda": self.args.gae_lambda,
                "clip_range": self.args.clip_range,
                "entropy_coef": self.args.entropy_coef,
                "vf_coef": self.args.vf_coef,
                "max_grad_norm": self.args.max_grad_norm,
                "target_kl": self.args.target_kl,
                "action_penalty_coef": self.args.action_penalty_coef
            },
            
            # ===== 学习率调度配置 =====
            "learning_rate_schedule": {
                "schedule_type": self.args.lr_schedule,
                "lr_min": self.args.lr_min,
                "warmup_ratio": self.args.warmup_ratio
            },
            
            # ===== 熵系数衰减配置 =====
            "entropy_decay": {
                "entropy_coef_start": self.args.entropy_coef_start,
                "entropy_coef_end": self.args.entropy_coef_end,
                "entropy_decay_end_ratio": self.args.entropy_decay_end_ratio
            },
            
            # ===== 训练设定 =====
            "training": {
                "total_timesteps": self.args.total_timesteps,
                "checkpoint_freq": self.args.checkpoint_freq,
                "eval_freq": self.args.eval_freq,
                "log_freq": self.args.log_freq
            },
            
            # ===== 课程学习配置 =====
            "curriculum_learning": {
                "enabled": self.args.use_curriculum,
                "mode": self.args.curriculum_mode,
                "alpha": self.args.curriculum_alpha,
                "gate_succ_threshold": self.args.gate_succ_threshold,
                "gate_coll_threshold": self.args.gate_coll_threshold
            },
            
            # ===== 认知模块配置 =====
            "cognitive_modules": {
                "enabled": self.args.use_cognitive_modules,
                "visualization": self.args.cognitive_visualization,
                
                # 认知偏差模块（风险厌恶）
                "cognitive_bias": {
                    "enabled": self.args.use_cognitive_bias,
                    "inverse_tta_coef": self.args.bias_inverse_tta_coef,
                    "tta_threshold": self.args.bias_tta_threshold,
                    "adaptive_bias": self.args.bias_adaptive,
                    "adaptation_rate": self.args.bias_adaptation_rate,
                    "visual_detection_distance": self.args.bias_visual_distance,
                    "visual_detection_angle": self.args.bias_visual_angle,
                    "visual_aversion_strength": self.args.bias_visual_strength
                },
                
                # 认知延迟模块（动作延迟）
                "cognitive_delay": {
                    "enabled": self.args.use_cognitive_delay,
                    "delay_steps": self.args.delay_steps,
                    "enable_smoothing": self.args.delay_smoothing,
                    "smoothing_factor": self.args.delay_smoothing_factor
                },
                
                # 认知感知模块（观测噪声）
                "cognitive_perception": {
                    "enabled": self.args.use_cognitive_perception,
                    "sigma0": self.args.perception_sigma0,
                    "k": self.args.perception_k,
                    "p_miss0": self.args.perception_p_miss0,
                    "p_false": self.args.perception_p_false,
                    "use_kf": self.args.perception_use_kf,
                    "kf_dt": self.args.perception_kf_dt,
                    "kf_q_scale": self.args.perception_kf_q_scale,
                    "enable_radar_beam_viz": getattr(self.args, 'enable_radar_beam_viz', False)
                },
                
                # 认知参数采样器
                "cognitive_parameter_sampler": {
                    "enabled": self.args.use_cognitive_parameter_sampling,
                    "sampler_type": self.args.cognitive_sampler_type,  # 采样器类型
                    "update_steps": self.args.cognitive_param_update_steps,
                    "bias_inverse_tta_coef_range": self.args.bias_inverse_tta_coef_range,
                    "perception_sigma0_range": self.args.perception_sigma0_range,
                    "perception_k_range": self.args.perception_k_range,
                    "delay_steps_range": self.args.delay_steps_range,
                    # === 新增：离散采样密度参数 ===
                    "bias_inverse_tta_coef_density": self.args.bias_inverse_tta_coef_density,
                    "perception_sigma0_density": self.args.perception_sigma0_density,
                    "perception_k_density": self.args.perception_k_density,
                    "delay_steps_density": self.args.delay_steps_density
                }
            }
        }
    
    def _save_config(self):
        """保存配置文件"""
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def _curriculum_schedule(self):
        """计算课程学习下的 map_string 和 traffic_density"""
        # 计算训练进度 p ∈ [0,1]
        p = min(1.0, float(self.global_step) / float(self.args.total_timesteps + 1e-8))

        if self.use_curriculum:
            if self.curriculum_mode == "progress":
                # 四阶段分段 - 修改分界点
                if p < 0.02:
                    stage = 0
                elif p < 0.05:
                    stage = 1
                elif p < 0.1:
                    stage = 2
                else:
                    stage = 3

                # 分段内归一化进度（用于插值）
                seg_edges = [0.0, 0.02, 0.05, 0.1, 1.0]
                seg_p = (p - seg_edges[stage]) / max(1e-8, (seg_edges[stage + 1] - seg_edges[stage]))
                seg_p = max(0.0, min(1.0, seg_p)) ** self.curriculum_alpha
            else:
                # gate 模式：直接用固定 stage
                stage = getattr(self, "curriculum_stage", 0)
                seg_p = 1.0
        else:
            # 未启用课程：直接用最高难度
            stage = 3
            seg_p = 1.0

        # 每阶段地图模板
        stage_maps = {
            0: "SSS",
            1: "SSSSS",
            2: "SCSCS",
            3: None   # None → 原始动态直线逻辑
        }

        # 每阶段目标交通密度区间
        stage_density = {
            0: (0.03, 0.06),
            1: (0.06, 0.09),
            2: (0.09, 0.12),
            3: (0.12, self.args.traffic_density_max)
        }

        # 计算交通密度
        d0, d1 = stage_density[stage]
        traffic_density_cur = d0 + (d1 - d0) * seg_p

        # 计算地图字符串
        if stage_maps[stage] is not None:
            map_string = stage_maps[stage]
        else:
            # 动态直线逻辑
            scenario_seed = self.args.seed if hasattr(self, 'current_scenario_seed') else self.args.seed
            scenario_index = scenario_seed % 1000
            min_segments, max_segments = 2, 10
            num_segments = min_segments + (scenario_index * (max_segments - min_segments)) // 1000
            map_string = "S" * num_segments

        return map_string, traffic_density_cur
    def _compute_dynamic_straight_map(self):
        """保持你原来的动态直线地图逻辑，仅返回 map_string"""
        scenario_seed = self.args.seed if hasattr(self, 'current_scenario_seed') else self.args.seed
        scenario_index = scenario_seed % 1000
        min_segments, max_segments = 2, 10
        num_segments = min_segments + (scenario_index * (max_segments - min_segments)) // 1000
        return "S" * num_segments

    def _curriculum_density(self):
        """
        根据训练模式计算交通密度：
        1. 恢复训练且用户指定密度范围 → 随机采样
        2. 其他情况 → 课程学习
        返回: float traffic_density
        """
        # 检测恢复训练 + 用户指定密度范围的情况
        if (hasattr(self.args, 'resume_from') and self.args.resume_from and 
            self._is_traffic_density_user_specified()):
            # 在用户指定范围内随机采样
            return self._sample_traffic_density_randomly()
        
        # 否则使用原有的课程学习逻辑
        return self._curriculum_density_original()
    
    def _is_traffic_density_user_specified(self) -> bool:
        """检测用户是否显式指定了交通密度参数"""
        return (self.args.traffic_density_min > 0 and 
                self.args.traffic_density_max > 0 and
                self.args.traffic_density_min < self.args.traffic_density_max)
    
    def _sample_traffic_density_randomly(self) -> float:
        """在用户指定范围内随机采样交通密度"""
        import random
        traffic_density = random.uniform(self.args.traffic_density_min, self.args.traffic_density_max)
        
        # 定期输出采样值（避免过度输出）
        if hasattr(self, '_density_sample_count'):
            self._density_sample_count += 1
        else:
            self._density_sample_count = 1
            
        if self._density_sample_count % 1000 == 1:  # 每1000次输出一次
            print(f"🎲 随机交通密度采样: {traffic_density:.4f} "
                  f"(范围: [{self.args.traffic_density_min:.3f}, {self.args.traffic_density_max:.3f}])")
        
        return traffic_density
    
    def _curriculum_density_original(self):
        """
        原始课程学习计算交通密度逻辑
        返回: float traffic_density
        """
        # 基本进度 p ∈ [0,1]
        p = min(1.0, float(self.global_step) / float(self.args.total_timesteps + 1e-8))

        # 默认目标阶段 = 3（最难），未开课程学习直接用目标分布
        stage = 3
        seg_p = 1.0

        if self.use_curriculum:
            if self.curriculum_mode == "progress":
                # 四段式：[0.0, 0.02, 0.05, 0.1, 1.0] - 修改分界点
                if p < 0.02:
                    stage = 0
                elif p < 0.05:
                    stage = 1
                elif p < 0.1:
                    stage = 2
                else:
                    stage = 3
                seg_edges = [0.0, 0.02, 0.05, 0.1, 1.0]
                seg_p = (p - seg_edges[stage]) / max(1e-8, (seg_edges[stage+1] - seg_edges[stage]))
                seg_p = max(0.0, min(1.0, seg_p)) ** self.curriculum_alpha
            else:
                # gate：由 self.curriculum_stage 控制
                stage = getattr(self, "curriculum_stage", 0)
                seg_p = 1.0

        # 仅密度分布按阶段变化；最后阶段使用默认范围
        default_max = 0.15  # 原始默认最大值
        stage_density = {
            0: (0.03, 0.06),
            1: (0.06, 0.09),
            2: (0.09, 0.12),
            3: (0.12, default_max)
        }

        d0, d1 = stage_density[stage]
        return d0 + (d1 - d0) * seg_p

    def _get_base_env_config(self):
        """获取基础环境配置 - 不包含动态参数"""
        # 地图：保持直线（动态段数），不受课程学习影响
        map_string = self._compute_dynamic_straight_map()

        # 使用初始交通密度，后续会动态更新
        initial_traffic_density = self._curriculum_density()
        
        return {
            # === 直线场景配置 ===
            "num_scenarios": 1000,
            "map": map_string,                   # 使用动态数量的直线段
            
            # === 动态交通配置 ===
            "traffic_density": initial_traffic_density,  # 初始交通密度
            "random_traffic": True,              # 启用交通随机化
            "horizon": 10000,
            "start_seed": self.args.seed,
            
            # === 渲染和观测配置（MetaDrive必需） ===
            "use_render": False,                 # 关闭渲染（训练时不需要）
            "debug": False,                      # 关闭调试模式
            "image_observation": False,          # 关闭图像观测（使用激光雷达）
            
            # === 优化的奖励配置 ===
            # 成功奖励 - 增加以鼓励完成任务
            "success_reward": self.args.success_reward,  # 默认10.0 -> 20.0
            
            # 前进奖励 - 鼓励沿道路前进
            "driving_reward": self.args.driving_reward,   # 默认1.0 -> 2.0
            
            # 速度奖励 - 鼓励合理速度
            "speed_reward": self.args.speed_reward,     # 默认0.1 -> 0.3
            
            # 车道保持奖励 - 关键！帮助学会沿车道行驶
            "use_lateral_reward": self.args.use_lateral_reward,  # 默认False -> True
            
            # 惩罚设置 - 平衡学习难度
            "out_of_road_penalty": self.args.out_of_road_penalty,      # 默认5.0 -> 8.0 (增加冲出道路惩罚)
            "crash_vehicle_penalty": self.args.crash_penalty,    # 默认5.0 -> 8.0
            "crash_object_penalty": self.args.crash_penalty,     # 默认5.0 -> 8.0
            "crash_sidewalk_penalty": 2.0,   # 默认0.0 -> 2.0 (增加撞人行道惩罚)
            

            
            # 终止条件 - 让智能体有更多机会学习
            "out_of_road_done": True,         # 确保冲出道路会终止
            "crash_vehicle_done": True,       # 确保撞车会终止
            "crash_object_done": True,        # 确保撞物体会终止
            "on_continuous_line_done": True, # 不允许压线，降低学习难度
            "on_broken_line_done": False,     # 允许压虚线
            
            "vehicle_config": {
                "lidar": dict(
                    num_lasers=240, 
                    distance=50, 
                    num_others=4, 
                    gaussian_noise=0.0, 
                    dropout_prob=0.0
                ),
                "side_detector": dict(
                    num_lasers=0, 
                    distance=50, 
                    gaussian_noise=0.0, 
                    dropout_prob=0.0
                ),
                "lane_line_detector": dict(
                    num_lasers=0, 
                    distance=20, 
                    gaussian_noise=0.0, 
                    dropout_prob=0.0
                )
            }
        }
    
    def _update_env_curriculum(self):
        """动态更新环境的课程学习参数"""
        if not self.use_curriculum:
            return
            
        # 计算当前交通密度
        current_traffic_density = self._curriculum_density()
        
        # 更新所有环境的交通密度
        # SB3 的 VecEnv 都支持 env_method；SubprocVecEnv 也支持
        if hasattr(self.envs, "env_method"):
            self.envs.env_method("update_traffic_density", current_traffic_density)
        elif hasattr(self.envs, "envs"):
            # DummyVecEnv 情况：拿到真实 env 实例逐个更新
            for env in self.envs.envs:
                if hasattr(env, "update_traffic_density"):
                    env.update_traffic_density(current_traffic_density)
                elif hasattr(env, "config"):
                    env.config["traffic_density"] = current_traffic_density

        # # 对于向量化环境，需要特殊处理
        # if hasattr(self.envs, 'set_attr'):
        #     # Stable Baselines3 的向量化环境
        #     self.envs.set_attr('config', {'traffic_density': current_traffic_density})
        # elif hasattr(self.envs, 'envs'):
        #     # 手动更新每个环境
        #     for env in self.envs.envs:
        #         if hasattr(env, 'config'):
        #             env.config['traffic_density'] = current_traffic_density
        #         # 如果环境有内部配置更新方法，也调用它
        #         if hasattr(env, 'update_traffic_density'):
        #             env.update_traffic_density(current_traffic_density)

    
    def _get_env_config(self):
        """获取环境配置 - 保持兼容性，实际使用 _get_base_env_config"""
        return self._get_base_env_config()
    
    def _create_environments(self):
        """创建向量化环境 - 支持真正的多进程并行"""
        # 注意：这里不直接调用 _get_env_config()，因为课程学习需要动态更新
        # 我们传递基础配置，交通密度在运行时动态计算
        base_env_config = self._get_base_env_config()
        
        # 添加速度控制参数到环境配置中
        if self.args.use_speed_control_reward:
            base_env_config.update({
                "use_speed_control_reward": True,
                "speed_control_k": self.args.speed_control_k,
                "speed_control_kappa": self.args.speed_control_kappa,
                "speed_control_mu": self.args.speed_control_mu,
                "speed_control_nu": self.args.speed_control_nu,
                "speed_control_v_tolerance": self.args.speed_control_v_tolerance,
                "speed_control_v_ref": self.args.speed_control_v_ref,
                # === 新增：子模块启用控制参数 ===
                "speed_control_enable_tracking": self.args.speed_control_enable_tracking,
                "speed_control_enable_soft_wall": self.args.speed_control_enable_soft_wall,
                "speed_control_enable_behavior_guidance": self.args.speed_control_enable_behavior_guidance,
            })
        
        if self.args.n_envs > 1:
            print(f" 创建 {self.args.n_envs} 个并行环境 (SubprocVecEnv)")
            print(f"   每个环境运行在独立子进程中，避免MetaDrive Engine单例限制")
            
            # 使用SubprocVecEnv创建多进程并行环境
            envs = SubprocVecEnv([
                make_env(rank, base_env_config, self.args) 
                for rank in range(self.args.n_envs)
            ])
            
            print(f"✅ 成功创建 {self.args.n_envs} 个并行环境")
            return envs
        else:
            print(" 创建单个环境 (DummyVecEnv)")
            
            # 单环境也使用向量化接口保持一致性
            envs = DummyVecEnv([make_env(0, base_env_config, self.args)])
            return envs
    
    def _update_entropy_coef(self):
        """更新熵系数 - 线性衰减逻辑"""
        # 计算训练进度
        progress = self.global_step / self.args.total_timesteps
        
        if progress <= self.entropy_decay_end_ratio:
            # 在衰减区间内进行线性插值
            decay_progress = progress / self.entropy_decay_end_ratio
            self.current_entropy_coef = (
                self.entropy_coef_start - 
                (self.entropy_coef_start - self.entropy_coef_end) * decay_progress
            )
        else:
            # 超过衰减区间，保持最终值
            self.current_entropy_coef = self.entropy_coef_end
        
        return self.current_entropy_coef
    
    def _init_csv_log(self):
        """初始化CSV日志文件"""
        # 恢复训练模式：检查CSV文件是否存在
        if hasattr(self.args, 'resume_from') and self.args.resume_from and os.path.exists(self.csv_path):
            print(f" 恢复CSV日志记录: {self.csv_path}")
            return
        
        # 正常模式：创建新的CSV文件
        headers = [
            "step", "episode", "ep_reward_mean", "ep_len_mean",
            "policy_loss", "value_loss", "entropy", "approx_kl",
            "learning_rate", "entropy_coef", "collision_rate", "offroad_rate", 
            "success_rate", "fps", "clipfrac", "explained_variance",
            "grad_norm", "avg_speed", "lane_deviation", "lane_change_count",
            "min_ttc", "path_completion",
            "steer_mean", "steer_std", "throttle_mean", "throttle_std",  # 新增动作统计列
            "action_penalty",  # === 新增：动作正则化惩罚列 ===
            # === 新增：认知参数列 ===
            "bias_inverse_tta_coef", "perception_sigma0", "perception_k", "delay_steps"
        ]
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f" CSV日志文件已创建: {self.csv_path}")
    
    def collect_rollouts(self) -> Tuple[torch.Tensor, ...]:
        """收集rollout数据 - 使用向量化环境的真实并行采样"""
    
        # 存储rollout数据
        obs_batch = []
        actions_batch = []
        log_probs_batch = []
        rewards_batch = []
        dones_batch = []
        values_batch = []
        
        # 使用向量化环境进行真实并行采样
        obs = self.envs.reset()  # 返回shape: (n_envs, obs_dim)
        
        # === 认知模块：重置状态 ===
        if self.use_cognitive_modules:
            if self.cognitive_delay_module:
                self.cognitive_delay_module.reset()
            if self.cognitive_perception_module:
                self.cognitive_perception_module.reset()
            if self.cognitive_bias_module:
                self.cognitive_bias_module.reset()
        
        # Episode统计变量 - 支持多环境
        episode_rewards = np.zeros(self.args.n_envs)
        episode_lengths = np.zeros(self.args.n_envs)
        episode_speeds = [[] for _ in range(self.args.n_envs)]
        lane_deviations = [[] for _ in range(self.args.n_envs)]
        lane_changes = np.zeros(self.args.n_envs)
        min_ttcs = [[] for _ in range(self.args.n_envs)]
        
        #  新增：动作统计变量
        episode_steer_means = [[] for _ in range(self.args.n_envs)]
        episode_throttle_means = [[] for _ in range(self.args.n_envs)]
        
        # 收集n_steps步数据
        for step in range(self.args.n_steps):
            # === 认知参数更新：基于真实仿真步数 ===
            if self.use_cognitive_modules and self.cognitive_parameter_sampler:
                current_sim_step = self.global_step + step
                if self.cognitive_parameter_sampler.should_update_parameters(current_sim_step):
                    new_params = self.cognitive_parameter_sampler.update_parameters(current_sim_step)
                    self._apply_cognitive_parameters(new_params)

            
            # === 认知感知模块：处理观测噪声 ===
            if self.use_cognitive_modules and self.cognitive_perception_module:
                # 对每个环境的观测应用感知噪声
                obs_processed = []
                for env_idx in range(self.args.n_envs):
                    single_obs = obs[env_idx]
                    # 应用感知噪声
                    noisy_obs = self.cognitive_perception_module.process_observation(
                        single_obs, 
                        is_ppo_mode=True
                    )
                    obs_processed.append(noisy_obs)
                obs = np.array(obs_processed)
            
            # === 认知参数集成：将认知参数拼接到观测中 ===
            current_cognitive_params = {}
            if self.use_cognitive_modules:
                if self.cognitive_parameter_sampler:
                    current_cognitive_params = self.cognitive_parameter_sampler.get_current_parameters()
   
                else:
                    # 如果没有参数采样器，使用命令行参数的默认值
                    current_cognitive_params = {
                        'bias_inverse_tta_coef': self.args.bias_inverse_tta_coef,
                        'perception_sigma0': self.args.perception_sigma0,
                        'perception_k': self.args.perception_k,
                        'delay_steps': self.args.delay_steps
                    }
            
            # 将认知参数拼接到观测中
            obs_with_cognitive = self._concatenate_cognitive_params(obs, current_cognitive_params)
            
            # 将观测转换为tensor (现在包含认知参数)
            obs_tensor = torch.as_tensor(obs_with_cognitive, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                actions, log_probs, _, values = self.network.get_action_and_value(obs_tensor)
                #  新增：获取动作统计信息
                steer_means, throttle_means = self.network.get_action_stats(obs_tensor)
            
            # 执行动作 - 向量化环境会自动处理多个环境
            actions_np = actions.cpu().numpy()
            
            # === 认知延迟模块：处理动作延迟 ===
            if self.use_cognitive_modules and self.cognitive_delay_module:
                # 对每个环境的动作应用延迟
                actions_delayed = []
                for env_idx in range(self.args.n_envs):
                    single_action = actions_np[env_idx]
                    # 应用动作延迟
                    delayed_action = self.cognitive_delay_module.process_action(
                        single_action,
                        is_ppo_mode=True
                    )
                    actions_delayed.append(delayed_action)
                actions_np = np.array(actions_delayed)
            
            next_obs, rewards, dones, infos = self.envs.step(actions_np)
            
            # === 认知偏差模块：处理奖励偏差 ===
            if self.use_cognitive_modules and self.cognitive_bias_module:
                # 对每个环境的奖励应用认知偏差
                for env_idx in range(self.args.n_envs):
                    # 获取当前环境实例
                    current_env = None
                    try:
                        if hasattr(self.envs, 'envs') and len(self.envs.envs) > env_idx:
                            current_env = self.envs.envs[env_idx]
                        elif hasattr(self.envs, 'venv') and hasattr(self.envs.venv, 'envs'):
                            current_env = self.envs.venv.envs[env_idx] if len(self.envs.venv.envs) > env_idx else None
                    except:
                        pass
                    
                    if current_env:
                        try:
                            # 根据文档使用正确的参数调用 process_reward
                            if hasattr(self.cognitive_bias_module, 'process_reward'):
                                reward_result = self.cognitive_bias_module.process_reward(
                                    original_reward=rewards[env_idx],
                                    env=current_env,
                                    info=infos[env_idx],
                                    is_ppo_mode=True
                                )
                                
                                # 处理返回值 - 根据文档是 (adjusted_reward, bias_info)
                                orig_reward_debug = rewards[env_idx]
                                if isinstance(reward_result, (tuple, list)) and len(reward_result) >= 2:
                                    adjusted_reward, bias_info = reward_result[0], reward_result[1]
                                    rewards[env_idx] = float(adjusted_reward)
                                    
                                    # 记录偏差信息用于可视化
                                    if isinstance(bias_info, dict):
                                        bias_amount = bias_info.get('bias_applied', 0.0)
                                        inverse_tta = bias_info.get('inverse_tta', 0.0)
                                        bias_active = bias_info.get('bias_active', False)
                                        
                                        # 只在第一个环境和偶尔输出调试信息，避免刷屏
                                        if bias_active and abs(bias_amount) > 1e-6 and env_idx == 0 and step % 100 == 0:
                                            print(f"易 认知偏差: {orig_reward_debug:.3f} → {rewards[env_idx]:.3f} (偏差: {bias_amount:+.3f}, TTA⁻¹: {inverse_tta:.3f})")
                                else:
                                    # 兼容性处理 - 单一返回值
                                    rewards[env_idx] = float(reward_result) if reward_result is not None else rewards[env_idx]
                        except Exception as e:
                            if env_idx == 0 and step == 0:  # 只在第一次报错
                                print(f"⚠️ 认知偏差模块处理失败: {e}")
            
            # 收集episode统计信息 - 处理多环境信息
            episode_rewards += rewards
            episode_lengths += 1
            
            # 处理多环境的info信息
            for env_idx, info in enumerate(infos):
                # 速度统计 - 修复：MetaDrive使用'velocity'键而非'speed'
                if 'velocity' in info:
                    episode_speeds[env_idx].append(info['velocity'])
                elif 'speed' in info:
                    episode_speeds[env_idx].append(info['speed'])
                elif hasattr(info, 'speed'):
                    episode_speeds[env_idx].append(info.speed)
                
                # 获取当前环境实例来计算缺失的指标
                try:
                    # 从向量化环境中获取对应的环境实例
                    current_env = None
                    if hasattr(self.envs, 'envs') and len(self.envs.envs) > env_idx:
                        current_env = self.envs.envs[env_idx]
                    elif hasattr(self.envs, 'venv') and hasattr(self.envs.venv, 'envs'):
                        current_env = self.envs.venv.envs[env_idx] if len(self.envs.venv.envs) > env_idx else None
                    
                    # 计算缺失的指标
                    if current_env is not None:
                        missing_metrics = self._calculate_missing_metrics(current_env, info, env_idx=0)
                    else:
                        missing_metrics = {}
                except Exception:
                    missing_metrics = {}
                
                # 车道偏移统计 - 优先使用info，其次使用计算值
                if 'lane_deviation' in info:
                    lane_deviations[env_idx].append(info['lane_deviation'])
                elif 'lane_deviation' in missing_metrics:
                    lane_deviations[env_idx].append(missing_metrics['lane_deviation'])
                
                # TTC统计 - 优先使用info，其次使用计算值
                if 'ttc' in info:
                    min_ttcs[env_idx].append(info['ttc'])
                elif 'min_ttc' in info:
                    min_ttcs[env_idx].append(info['min_ttc'])
                elif 'ttc' in missing_metrics:
                    min_ttcs[env_idx].append(missing_metrics['ttc'])
                
                # 车道变换检测 - 优先使用info，其次使用计算值
                lane_change_detected = False
                if 'lane_change' in info and info['lane_change']:
                    lane_changes[env_idx] += 1
                    lane_change_detected = True
                    if self.debug_lane_change:
                        print(f" [环境{env_idx}] 通过info检测到车道变更")
                elif missing_metrics.get('lane_change', False):
                    lane_changes[env_idx] += 1
                    lane_change_detected = True
                    if self.debug_lane_change:
                        print(f" [环境{env_idx}] 通过missing_metrics检测到车道变更")
                else:
                    # 使用增强的车道变更检测
                    if current_env is not None and hasattr(current_env, 'agent'):
                        lane_change_detected = self._detect_lane_change_enhanced(current_env.agent, env_idx, info)
                        if lane_change_detected:
                            lane_changes[env_idx] += 1
                            if self.debug_lane_change:
                                print(f" [环境{env_idx}] 通过增强检测检测到车道变更")


                #  新增：收集动作统计信息
                episode_steer_means[env_idx].append(steer_means[env_idx].item())
                episode_throttle_means[env_idx].append(throttle_means[env_idx].item())
            
            # 处理episode结束 - 检查每个环境
            for env_idx in range(self.args.n_envs):
                if dones[env_idx]:
                    # 记录episode统计
                    self.episode_rewards.append(episode_rewards[env_idx])
                    self.episode_lengths.append(episode_lengths[env_idx])
                    self.episode_speeds.append(np.mean(episode_speeds[env_idx]) if episode_speeds[env_idx] else 0)
                    self.episode_lane_deviations.append(np.mean(lane_deviations[env_idx]) if lane_deviations[env_idx] else 0)
                    self.episode_lane_changes.append(lane_changes[env_idx])
                    self.episode_min_ttcs.append(np.min(min_ttcs[env_idx]) if min_ttcs[env_idx] else float('inf'))
                    
                    #  新增：记录动作统计
                    self.episode_steer_means.append(np.mean(episode_steer_means[env_idx]) if episode_steer_means[env_idx] else 0)
                    self.episode_throttle_means.append(np.mean(episode_throttle_means[env_idx]) if episode_throttle_means[env_idx] else 0)

                    # 路径完成度计算
                    info = infos[env_idx]
                    path_completion = info.get('route_completion', 0.0)
                    if 'arrive_dest' in info and info['arrive_dest']:
                        path_completion = 1.0
                    self.episode_path_completions.append(path_completion)
                    
                    # 超时检测
                    timeout = (episode_lengths[env_idx] >= self.envs.get_attr('config')[env_idx]['horizon'] and 
                              not info.get('arrive_dest', False) and not info.get('crash', False))
                    self.episode_timeouts.append(1 if timeout else 0)
                    
                    # 重置该环境的统计
                    episode_rewards[env_idx] = 0
                    episode_lengths[env_idx] = 0
                    episode_speeds[env_idx] = []
                    lane_deviations[env_idx] = []
                    lane_changes[env_idx] = 0
                    min_ttcs[env_idx] = []
                    #  新增：重置动作统计
                    episode_steer_means[env_idx] = []
                    episode_throttle_means[env_idx] = []
            
            # 存储数据 - 直接使用向量化环境的真实数据
            # === 修复：存储包含认知参数的观测 ===
            if self.use_cognitive_modules:
                # 存储包含认知参数的观测
                obs_to_store = obs_with_cognitive.copy()
            else:
                # 存储原始观测
                obs_to_store = obs.copy()
            
            obs_batch.append(obs_to_store)
            actions_batch.append(actions_np)
            log_probs_batch.append(log_probs.cpu().numpy())
            rewards_batch.append(rewards)
            dones_batch.append(dones.astype(np.float32))
            values_batch.append(values.cpu().numpy())
            
            obs = next_obs  # 更新观测
        
        # 修复1: 计算last_values作为bootstrap值
        # 确保last_obs也包含认知参数
        current_cognitive_params = {}
        if self.use_cognitive_modules and self.cognitive_parameter_sampler:
            try:
                current_cognitive_params = self.cognitive_parameter_sampler.get_current_parameters()
            except Exception as e:
                print(f"⚠️ 获取认知参数失败: {e}")
                # 使用默认认知参数
                current_cognitive_params = {
                    'bias_inverse_tta_coef': 1.0,
                    'perception_sigma0': 0.1,
                    'perception_k': 0.02,
                    'delay_steps': 2
                }
        elif self.use_cognitive_modules:
            # 使用默认认知参数
            current_cognitive_params = {
                'bias_inverse_tta_coef': 1.0,
                'perception_sigma0': 0.1,
                'perception_k': 0.02,
                'delay_steps': 2
            }
        
        last_obs_with_cognitive = self._concatenate_cognitive_params(obs, current_cognitive_params)
        last_obs_tensor = torch.as_tensor(last_obs_with_cognitive, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            _, _, _, last_values = self.network.get_action_and_value(last_obs_tensor)
        
        # 更新全局步数
        self.global_step += self.args.n_steps * self.args.n_envs
        
        # 转换为tensor - 使用更高效的方法
        obs_batch     = torch.as_tensor(np.asarray(obs_batch,     dtype=np.float32), device=self.device)
        actions_batch = torch.as_tensor(np.asarray(actions_batch, dtype=np.float32), device=self.device)
        log_probs_batch = torch.as_tensor(np.asarray(log_probs_batch, dtype=np.float32), device=self.device)
        rewards_batch = torch.as_tensor(np.asarray(rewards_batch, dtype=np.float32), device=self.device)
        dones_batch   = torch.as_tensor(np.asarray(dones_batch,   dtype=np.float32), device=self.device)
        values_batch  = torch.as_tensor(np.asarray(values_batch,  dtype=np.float32), device=self.device)

        
        #  修复1: 使用last_values计算正确的GAE  
        advantages, returns = self.compute_gae(rewards_batch, values_batch, dones_batch, last_values)
        
        return (obs_batch, actions_batch, log_probs_batch, 
                advantages, returns)
    
    def compute_gae(self, rewards, values, dones, last_values):
        """ 修复1: 计算GAE优势函数 - 使用正确的bootstrap值"""
        # rewards: [n_steps, n_envs]
        # values:  [n_steps, n_envs] 
        # last_values: [n_envs] 对应s_T的V估计（bootstrap）
        n_steps, n_envs = rewards.shape
        advantages = torch.zeros_like(rewards, device=self.device)
        last_values = torch.as_tensor(last_values, dtype=torch.float32, device=self.device)
        last_adv = torch.zeros(n_envs, device=self.device)
        
        for t in reversed(range(n_steps)):
            next_non_terminal = 1.0 - dones[t]
            next_value = last_values if t == n_steps - 1 else values[t + 1]
            delta = rewards[t] + self.args.gamma * next_value * next_non_terminal - values[t]
            last_adv = delta + self.args.gamma * self.args.gae_lambda * next_non_terminal * last_adv
            advantages[t] = last_adv
        
        returns = advantages + values
        return advantages, returns
    
    def update_policy(self, obs, actions, old_log_probs, advantages, returns):
        """更新策略"""
        # === 调试信息：显示观测维度 ===
        print(f" update_policy调试信息:")
        print(f"   输入obs形状: {obs.shape}")
        print(f"   认知模块状态: {self.use_cognitive_modules}")
        if self.use_cognitive_modules and self.cognitive_parameter_sampler:
            try:
                current_params = self.cognitive_parameter_sampler.get_current_parameters()
                print(f"   当前认知参数: {current_params}")
            except Exception as e:
                print(f"   ⚠️ 获取认知参数失败: {e}")
                print(f"   使用默认认知参数")
        
        # 标准化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 准备训练数据
        batch_size = obs.shape[0] * obs.shape[1]  # n_steps * n_envs
        obs = obs.view(batch_size, -1)
        actions = actions.view(batch_size, -1)
        old_log_probs = old_log_probs.view(batch_size)
        advantages = advantages.view(batch_size)
        returns = returns.view(batch_size)
        
        # === 修复：确保观测包含认知参数 ===
        if self.use_cognitive_modules and obs.shape[1] == 275:
            # 如果观测是275维，需要添加认知参数
            print(f"   检测到275维观测，正在添加认知参数...")
            current_cognitive_params = {}
            if self.cognitive_parameter_sampler:
                try:
                    current_cognitive_params = self.cognitive_parameter_sampler.get_current_parameters()
                except Exception as e:
                    print(f"   ⚠️ 获取认知参数失败: {e}")
                    # 使用默认认知参数
                    current_cognitive_params = {
                        'bias_inverse_tta_coef': 1.0,
                        'perception_sigma0': 0.1,
                        'perception_k': 0.02,
                        'delay_steps': 2
                    }
            else:
                # 使用默认认知参数
                current_cognitive_params = {
                    'bias_inverse_tta_coef': 1.0,
                    'perception_sigma0': 0.1,
                    'perception_k': 0.02,
                    'delay_steps': 2
                }
            
            # 为每个样本添加认知参数
            cognitive_vector = np.array([
                [current_cognitive_params['bias_inverse_tta_coef'],
                 current_cognitive_params['perception_sigma0'],
                 current_cognitive_params['perception_k'],
                 current_cognitive_params['delay_steps']] for _ in range(batch_size)
            ], dtype=np.float32)
            
            # 拼接认知参数
            obs = np.concatenate([obs.cpu().numpy(), cognitive_vector], axis=1)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            print(f"   观测维度已扩展: 275 → {obs.shape[1]}")
        elif self.use_cognitive_modules and obs.shape[1] != 279:
            print(f"   ⚠️ 警告：观测维度异常: {obs.shape[1]}，期望279")
        else:
            print(f"   观测维度正常: {obs.shape[1]}")
        
        # 多轮更新
        policy_losses = []
        value_losses = []
        entropy_losses = []
        entropies = []
        approx_kls = []
        clipfracs = []
        grad_norms = []
        
        # 计算初始值函数的解释方差
        with torch.no_grad():
            _, _, _, initial_values = self.network.get_action_and_value(obs)
            y_pred = initial_values
            y_true = returns
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
        
        for epoch in range(self.args.n_epochs):
            # 打乱数据
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, self.args.batch_size):
                end = start + self.args.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算clip fraction
                clip_fraction = torch.mean(((ratio - 1.0).abs() > self.args.clip_range).float())
                clipfracs.append(clip_fraction.item())
                
                # PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value损失
                value_loss = nn.MSELoss()(new_values, batch_returns)
                
                # Entropy损失
                entropy_loss = -entropy.mean()
                
                # === 新增：动作正则化惩罚 ===
                # 惩罚过大的动作值，防止油门/转向饱和
                action_penalty = self.args.action_penalty_coef * torch.mean(torch.abs(batch_actions))
                
                # 总损失
                total_loss = policy_loss + self.args.vf_coef * value_loss + self.current_entropy_coef * entropy_loss + action_penalty
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 计算梯度范数
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)
                grad_norms.append(grad_norm.item())
                
                self.optimizer.step()
                
                # 记录统计
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                entropies.append(entropy.mean().item())
                
                # 计算KL散度
                with torch.no_grad():
                    approx_kl = ((new_log_probs - batch_old_log_probs) ** 2).mean()
                    approx_kls.append(approx_kl.item())
                
                # 早停检查
                if self.args.target_kl and approx_kl > self.args.target_kl:
                    break
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "entropy": np.mean(entropies),
            "approx_kl": np.mean(approx_kls),
            "clipfrac": np.mean(clipfracs),
            "explained_variance": explained_var.item(),
            "grad_norm": np.mean(grad_norms),
            "action_penalty": self.args.action_penalty_coef * torch.mean(torch.abs(actions)).item()  # 记录动作惩罚
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """评估策略 - 使用独立的评估环境"""
        eval_rewards = []
        eval_lengths = []
        eval_collisions = 0
        eval_offroads = 0
        eval_successes = 0
        eval_speeds = []
        eval_lane_deviations = []
        eval_lane_changes = []
        eval_min_ttcs = []
        eval_path_completions = []
        
        print(f" 开始策略评估 ({num_episodes} episodes)...")
        
        for episode in range(num_episodes):
            obs = self.envs.reset()
            episode_reward = 0
            episode_length = 0
            episode_speeds = []
            episode_lane_deviations = []
            episode_lane_changes = 0
            episode_min_ttcs = []
            
            #  从环境获取第一个实例用于指标计算
            try:
                if hasattr(self.envs, 'envs') and len(self.envs.envs) > 0:
                    current_env = self.envs.envs[0]
                elif hasattr(self.envs, 'venv') and hasattr(self.envs.venv, 'envs'):
                    current_env = self.envs.venv.envs[0] if len(self.envs.venv.envs) > 0 else None
                else:
                    current_env = None
            except Exception:
                current_env = None
            
            while True:
                # === 认知参数集成：将认知参数拼接到观测中 ===
                current_cognitive_params = {}
                if self.use_cognitive_modules and self.cognitive_parameter_sampler:
                    try:
                        current_cognitive_params = self.cognitive_parameter_sampler.get_current_parameters()
                    except Exception as e:
                        print(f"⚠️ 获取认知参数失败: {e}")
                        # 使用默认认知参数
                        current_cognitive_params = {
                            'bias_inverse_tta_coef': 1.0,
                            'perception_sigma0': 0.1,
                            'perception_k': 0.02,
                            'delay_steps': 2
                        }
                
                # 将认知参数拼接到观测中
                obs_with_cognitive = self._concatenate_cognitive_params(obs, current_cognitive_params)
                
                obs_tensor = torch.FloatTensor(obs_with_cognitive).to(self.device)
                
                with torch.no_grad():
                    #  修复2: 评估使用确定性动作（均值）
                    action = self.network.act_deterministic(obs_tensor)
                
                #  修复6: 评估时也要clip动作
                action_np = action.cpu().numpy()
                # action_np = np.clip(action_np, -1.0, 1.0)
                obs, reward, done, info = self.envs.step(action_np)
                
                # 处理多环境返回值 - 取第一个环境的数据用于评估
                if isinstance(reward, (list, tuple, np.ndarray)):
                    episode_reward += reward[0]
                    done_flag = done[0]
                    info = info[0]
                else:
                    episode_reward += reward
                    done_flag = done
                
                episode_length += 1
                
                #  修复：更全面的速度获取
                speed_value = 0.0
                if isinstance(info, dict):
                    if 'velocity' in info:
                        speed_value = abs(info['velocity'])  # 取绝对值
                    elif 'speed' in info:
                        speed_value = abs(info['speed'])
                    elif hasattr(info, 'speed'):
                        speed_value = abs(info.speed)
                # 从agent直接获取速度
                if speed_value == 0.0 and current_env is not None:
                    try:
                        if hasattr(current_env, 'agent') and hasattr(current_env.agent, 'speed'):
                            speed_value = abs(current_env.agent.speed)
                    except:
                        pass
                
                if speed_value > 0:
                    episode_speeds.append(speed_value)
                
                #  修复：更完善的指标计算
                missing_metrics = {}
                if current_env is not None:
                    try:
                        missing_metrics = self._calculate_missing_metrics(current_env, info, env_idx=0)  # 评估时使用env_idx=0
                    except Exception as e:
                        # print(f"警告：指标计算失败: {e}")
                        pass
                
                # 车道偏移统计
                if isinstance(info, dict) and 'lane_deviation' in info:
                    episode_lane_deviations.append(abs(info['lane_deviation']))
                elif 'lane_deviation' in missing_metrics:
                    episode_lane_deviations.append(abs(missing_metrics['lane_deviation']))
                else:
                    # 添加默认值避免空列表
                    episode_lane_deviations.append(0.0)
                
                # TTC统计
                if isinstance(info, dict):
                    if 'ttc' in info:
                        episode_min_ttcs.append(info['ttc'])
                    elif 'min_ttc' in info:
                        episode_min_ttcs.append(info['min_ttc'])
                elif 'ttc' in missing_metrics:
                    episode_min_ttcs.append(missing_metrics['ttc'])
                else:
                    # 添加默认值
                    episode_min_ttcs.append(10.0)  # 默认安全TTC
                
                # 车道变换检测
                if isinstance(info, dict) and 'lane_change' in info and info['lane_change']:
                    episode_lane_changes += 1
                    if self.debug_lane_change:
                        print(f" [评估] 通过info检测到车道变更")
                elif missing_metrics.get('lane_change', False):
                    episode_lane_changes += 1
                    if self.debug_lane_change:
                        print(f" [评估] 通过missing_metrics检测到车道变更")
                else:
                    # 使用增强的车道变更检测
                    if current_env is not None and hasattr(current_env, 'agent'):
                        if self._detect_lane_change_enhanced(current_env.agent, 0, info):  # 评估时使用env_idx=0
                            episode_lane_changes += 1
                            if self.debug_lane_change:
                                print(f" [评估] 通过增强检测检测到车道变更")
                
                if done_flag:
                    #  修复：更准确的终止原因统计
                    crash_detected = False
                    offroad_detected = False
                    success_detected = False
                    
                    if isinstance(info, dict):
                        # 检查各种碰撞情况
                        if (info.get("crash", False) or 
                            info.get("crash_vehicle", False) or 
                            info.get("crash_object", False) or
                            info.get("collision", False)):
                            crash_detected = True
                            eval_collisions += 1
                        # 检查冲出道路
                        elif info.get("out_of_road", False):
                            offroad_detected = True
                            eval_offroads += 1
                        # 检查成功到达
                        elif info.get("arrive_dest", False):
                            success_detected = True
                            eval_successes += 1
                    
                    # 计算路径完成度
                    path_completion = 0.0
                    if isinstance(info, dict):
                        path_completion = info.get('route_completion', 0.0)
                        if info.get('arrive_dest', False):
                            path_completion = 1.0
                    eval_path_completions.append(path_completion)
                    
                    # # 调试输出
                    # print(f"   Episode {episode+1}: reward={episode_reward:.2f}, length={episode_length}, "
                    #       f"speed={np.mean(episode_speeds) if episode_speeds else 0:.2f}, "
                    #       f"crash={crash_detected}, offroad={offroad_detected}, success={success_detected}")
                    
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_speeds.append(np.mean(episode_speeds) if episode_speeds else 0)
            eval_lane_deviations.append(np.mean(episode_lane_deviations) if episode_lane_deviations else 0)
            eval_lane_changes.append(episode_lane_changes)
            eval_min_ttcs.append(np.min(episode_min_ttcs) if episode_min_ttcs else 10.0)
        
        # 计算最终统计
        collision_rate = eval_collisions / num_episodes
        offroad_rate = eval_offroads / num_episodes  
        success_rate = eval_successes / num_episodes
        
        print(f"✅ 评估完成: 碰撞率={collision_rate:.3f}, 冲出道路率={offroad_rate:.3f}, 成功率={success_rate:.3f}")
        
        return {
            "eval_reward_mean": np.mean(eval_rewards),
            "eval_reward_std": np.std(eval_rewards),
            "eval_length_mean": np.mean(eval_lengths),
            "eval_collision_rate": collision_rate,
            "eval_offroad_rate": offroad_rate,
            "eval_success_rate": success_rate,
            "eval_avg_speed": np.mean(eval_speeds),
            "eval_lane_deviation": np.mean(eval_lane_deviations),
            "eval_lane_change_count": np.mean(eval_lane_changes),
            "eval_min_ttc": np.mean([ttc for ttc in eval_min_ttcs if ttc != float('inf')]) if any(ttc != float('inf') for ttc in eval_min_ttcs) else 10.0,
            "eval_path_completion": np.mean(eval_path_completions)
        }
    
    def save_checkpoint(self, iteration: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            "iteration": iteration,
            "global_step": self.global_step,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "args": vars(self.args)
        }
        
        # === 保存渐进式训练状态 ===
        if self.is_progressive_training:
            checkpoint['progressive_training'] = {
                'is_progressive_training': True,
                'training_stage': self.network.training_stage,
                'checkpoint_obs_dim': self.checkpoint_obs_dim,
                'freeze_threshold_steps': self.network.freeze_threshold_steps
            }
        
        # 保存认知模块状态
        if self.use_cognitive_modules:
            cognitive_states = {}
            
            if self.cognitive_bias_module:
                cognitive_states['bias_module'] = {
                    'inverse_tta_coef': getattr(self.cognitive_bias_module, 'inverse_tta_coef', 1.0),
                    'tta_threshold': getattr(self.cognitive_bias_module, 'tta_threshold', 1.0),
                    'step_count': getattr(self.cognitive_bias_module, '_step_count', 0),
                    'total_bias': getattr(self.cognitive_bias_module, '_total_bias', 0.0),
                    'active_steps': getattr(self.cognitive_bias_module, '_active_steps', 0)
                }
            
            if self.cognitive_delay_module:
                cognitive_states['delay_module'] = self.cognitive_delay_module.get_status()
            
            if self.cognitive_perception_module:
                cognitive_states['perception_module'] = {
                    'initialized': getattr(self.cognitive_perception_module, 'initialized', False),
                    'sigma0': getattr(self.cognitive_perception_module, 'sigma0', 0.0),
                    'k': getattr(self.cognitive_perception_module, 'k', 0.0)
                }
            
            # === 新增：保存认知参数采样器状态 ===
            if self.cognitive_parameter_sampler:
                cognitive_states['parameter_sampler'] = {
                    'current_parameters': self.cognitive_parameter_sampler.get_current_parameters(),
                    'total_updates': getattr(self.cognitive_parameter_sampler, '_total_updates', 0),
                    'last_update_step': getattr(self.cognitive_parameter_sampler, '_last_update_step', 0),
                    'update_steps': self.cognitive_parameter_sampler.update_steps,
                    'param_history_count': len(self.cognitive_parameter_sampler.param_history)
                }
            
            checkpoint['cognitive_states'] = cognitive_states
        
        # 保存常规检查点
        checkpoint_path = os.path.join(self.exp_dir, "checkpoints", f"checkpoint_{iteration}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.exp_dir, "checkpoints", "best_model.pt")
            torch.save(checkpoint, best_path)
            
        # 保存最新模型
        latest_path = os.path.join(self.exp_dir, "checkpoints", "latest_model.pt")
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """从检查点恢复训练状态"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        print(f"📥 正在加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        required_keys = ["iteration", "global_step", "network_state_dict", "optimizer_state_dict"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"检查点格式不完整，缺少键: {missing_keys}")
        
        ckpt_state = checkpoint["network_state_dict"]
        if "actor_fc1.weight" in ckpt_state:
            checkpoint_obs_dim = ckpt_state["actor_fc1.weight"].shape[1]
        elif "critic_fc1.weight" in ckpt_state:
            checkpoint_obs_dim = ckpt_state["critic_fc1.weight"].shape[1]
        else:
            checkpoint_obs_dim = None

        if self.network is None:
            target_obs_dim = 279 if self.use_cognitive_modules else 275
            print(f"🔧 根据目标配置创建网络: {target_obs_dim}维")
            self.network = PPONetwork(obs_dim=target_obs_dim).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.args.lr)
        
        current_obs_dim = self.network.actor_fc1.in_features
        self.checkpoint_obs_dim = checkpoint_obs_dim

        # ---- 权重加载 ----
        if checkpoint_obs_dim and checkpoint_obs_dim != current_obs_dim:
            if checkpoint_obs_dim == 275 and current_obs_dim == 279:
                print("🧠 检测到 275→279 迁移，调用 _load_and_extend_weights")
                extended_state_dict = self.network._load_and_extend_weights(ckpt_state)
                self.network.load_state_dict(extended_state_dict)
                self.is_progressive_training = True
            elif checkpoint_obs_dim == 279 and current_obs_dim == 275:
                print("🧠 检测到 279→275 迁移，调用 _load_and_truncate_weights")
                truncated_state_dict = self.network._load_and_truncate_weights(ckpt_state)
                self.network.load_state_dict(truncated_state_dict)
            else:
                print("⚠️ 无法自动处理维度不匹配，将尝试直接加载")
                self.network.load_state_dict(ckpt_state, strict=False)
        else:
            print(f"✅ 维度匹配，直接加载权重")
            self.network.load_state_dict(ckpt_state)

        print("✅ 网络权重已恢复")

        # ---- 恢复优化器状态（部分保留） ----
        # try:
        self.safe_load_optimizer_state(self.optimizer, checkpoint["optimizer_state_dict"], self.network)
        # except Exception as e:
        #     print(f"⚠️ 优化器状态部分恢复失败: {e}")
        #     print(f"   将重新初始化优化器参数状态")
        #     self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.lr)

        # 强制覆盖学习率
        if hasattr(self.args, 'lr'):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args.lr
            print(f"🔄 学习率已更新为新设置: {self.args.lr}")

        # ---- 恢复训练进度 ----
        self.global_step = checkpoint["global_step"]
        self.start_iteration = checkpoint["iteration"]

        # === 渐进式训练只依赖 global_step ===
        if self.is_progressive_training:
            self.network.training_stage = 1
            if self.global_step >= self.network.freeze_threshold_steps:
                self.network.update_training_stage(self.global_step)

        print(f"📊 训练状态恢复:")
        print(f"   全局步数: {self.global_step:,}")
        print(f"   迭代次数: {self.start_iteration}")

        # === TensorBoard 重建 ===
        self.writer.close()
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "tensorboard"))

        
    def safe_load_optimizer_state(self, optimizer, state_dict, model):
        """只保留 shape 匹配的动量状态"""
        new_state = optimizer.state_dict()

        # 映射: 参数 id -> 参数对象
        param_map = {id(p): p for group in optimizer.param_groups for p in group['params']}

        for pid, old_state in state_dict["state"].items():
            if pid not in param_map:
                continue
            param = param_map[pid]
            if param.shape != old_state.get("exp_avg", torch.zeros_like(param)).shape:
                print(f"⚠️ 丢弃优化器状态: {param.shape} vs {old_state.get('exp_avg').shape}")
                continue

            # shape 匹配 -> 保留该参数的状态
            new_state["state"][pid] = old_state

        optimizer.load_state_dict(new_state)
        print("✅ 优化器状态已部分恢复（仅保留 shape 匹配的参数）")


    def _validate_checkpoint_compatibility(self, checkpoint: Dict):
        """验证检查点与当前配置的兼容性"""
        if "config" not in checkpoint:
            print("⚠️  检查点中无配置信息，跳过兼容性检查")
            return
        
        checkpoint_config = checkpoint["config"]
        current_config = self.config
        
        # 关键参数兼容性检查
        critical_params = [
            "obs_dim", "action_dim", "num_scenarios", "map",
            "vehicle_config", "success_reward", "driving_reward"
        ]
        
        incompatible_params = []
        for param in critical_params:
            if param in checkpoint_config and param in current_config:
                if checkpoint_config[param] != current_config[param]:
                    incompatible_params.append(f"{param}: {checkpoint_config[param]} -> {current_config[param]}")
        
        if incompatible_params:
            print("⚠️  检测到配置差异:")
            for param in incompatible_params:
                print(f"   {param}")
            print("继续训练可能导致不可预期的结果")
        else:
            print("✅ 配置兼容性检查通过")
    
    def _calculate_missing_metrics(self, env, info, env_idx=None):
        """
        计算MetaDrive info中缺失的指标
        
        Args:
            env: MetaDrive环境实例（用于访问agent）
            info: 环境返回的info字典
            env_idx: 环境索引，用于车道变更检测
            
        Returns:
            dict: 包含计算出的指标的字典
        """
        metrics = {}
        
        try:
            agent = env.agent if hasattr(env, 'agent') else None
            if agent is None:
                return metrics
            
            # 1. 车道偏移计算 (Lane Deviation)
            try:
                # 方法1: 尝试直接从agent获取车道中心距离
                if hasattr(agent, 'lateral_distance_to_lane_center'):
                    metrics['lane_deviation'] = abs(agent.lateral_distance_to_lane_center)
                elif hasattr(agent, 'lane') and hasattr(agent, 'position'):
                    # 方法2: 基于转向角度估算车道偏移
                    # 获取转向角度（steering）作为车道偏移的指标
                    steering = getattr(agent, 'steering', 0.0)
                    speed = getattr(agent, 'speed', 0.0)
                    
                    # 基于转向和速度计算车道偏移的估计值
                    # 这是一个启发式方法，实际偏移应该基于几何计算
                    if hasattr(agent.lane, 'width'):
                        lane_width = agent.lane.width
                        # 转向越大，偏移越大；速度越快，偏移影响越大
                        estimated_deviation = abs(steering) * (1 + speed * 0.1) * lane_width * 0.3
                        # 限制在合理范围内
                        metrics['lane_deviation'] = min(estimated_deviation, lane_width / 2)
                    else:
                        # 简单基于转向角的偏移
                        metrics['lane_deviation'] = abs(steering) * 0.5
                else:
                    metrics['lane_deviation'] = 0.0
            except Exception:
                metrics['lane_deviation'] = 0.0
            
            # 2. 车道变更检测 (Lane Change)
            try:
                # 改进的车道变更检测方法
                lane_change_detected = False
                
                # 方法1: 检查MetaDrive环境info中的车道变更标志
                if isinstance(info, dict) and 'lane_change' in info:
                    lane_change_detected = bool(info['lane_change'])
                    if lane_change_detected:
                        if self.debug_lane_change:
                            print(f" [环境{env_idx}] 通过info检测到车道变更")
                        return True
                
                # 方法2: 基于车道索引变化检测
                if not lane_change_detected and hasattr(agent, 'lane_index'):
                    current_lane_index = agent.lane_index
                    env_agent_id = f"env_{env_idx}_lane_index"
                    
                    if env_agent_id in self._last_lane_index:
                        if self._last_lane_index[env_agent_id] != current_lane_index:
                            if self.debug_lane_change:
                                print(f" [环境{env_idx}] 车道索引变化: {self._last_lane_index[env_agent_id]} → {current_lane_index}")
                            self._last_lane_index[env_agent_id] = current_lane_index
                            return True
                        else:
                            self._last_lane_index[env_agent_id] = current_lane_index
                            return False
                    else:
                        # 首次记录
                        self._last_lane_index[env_agent_id] = current_lane_index
                        if self.debug_lane_change:
                            print(f" [环境{env_idx}] 首次记录车道索引: {current_lane_index}")
                        return False
                
                # 方法3: 基于转向角度和速度启发式检测
                if not lane_change_detected:
                    try:
                        steering = getattr(agent, 'steering', 0.0)
                        speed = getattr(agent, 'speed', 0.0)
                        
                        # 如果转向角度大且速度适中，可能是车道变更
                        if abs(steering) > 0.3 and 5.0 < speed < 20.0:
                            # 检查是否持续了一段时间
                            if env_idx is not None:
                                steering_key = f"env_{env_idx}_steering"
                            else:
                                steering_key = f"agent_{getattr(agent, 'id', id(agent))}_steering"
                                
                            if steering_key not in self._last_lane_index:
                                self._last_lane_index[steering_key] = 0
                            
                            self._last_lane_index[steering_key] += 1
                            
                            # 如果连续3步都有大转向，可能是车道变更
                            if self._last_lane_index[steering_key] >= 3:
                                if self.debug_lane_change:
                                    print(f" [环境{env_idx}] 通过转向模式检测到车道变更: 转向={steering:.3f}, 速度={speed:.3f}, 持续步数={self._last_lane_index[steering_key]}")
                                self._last_lane_index[steering_key] = 0  # 重置计数器
                                return True
                        else:
                            # 重置转向计数器
                            steering_key = f"env_{env_idx}_steering_pattern"
                            if steering_key in self._last_lane_index:
                                self._last_lane_index[steering_key] = 0
                                
                    except Exception as e:
                        if self.debug_lane_change:
                            print(f"⚠️ [环境{env_idx}] 转向模式检测失败: {e}")
                
                # 方法4: 直线道路场景的替代检测（模拟车道变更行为）
                if not lane_change_detected:
                    try:
                        steering = getattr(agent, 'steering', 0.0)
                        speed = getattr(agent, 'speed', 0.0)
                        
                        # 在直线道路场景中，检测"车道变更意图"而不是实际变更
                        # 这基于转向模式、速度和位置变化
                        if abs(steering) > 0.15 and 5.0 < speed < 25.0:  # 降低转向阈值
                            # 检查是否持续了一段时间
                            steering_key = f"env_{env_idx}_steering_intent"
                            
                            if steering_key not in self._last_lane_index:
                                self._last_lane_index[steering_key] = 0
                            
                            self._last_lane_index[steering_key] += 1
                            
                            # 连续3步有转向意图判定为"车道变更意图"
                            if self._last_lane_index[steering_key] >= 3:
                                if self.debug_lane_change:
                                    print(f" [环境{env_idx}] 检测到车道变更意图: 转向={steering:.3f}, 速度={speed:.3f}, 持续步数={self._last_lane_index[steering_key]}")
                                self._last_lane_index[steering_key] = 0  # 重置计数器
                                return True
                        else:
                            # 重置转向意图计数器
                            steering_key = f"env_{env_idx}_steering_intent"
                            if steering_key in self._last_lane_index:
                                self._last_lane_index[steering_key] = 0
                                
                    except Exception as e:
                        if self.debug_lane_change:
                            print(f"⚠️ [环境{env_idx}] 车道变更意图检测失败: {e}")
                
                metrics['lane_change'] = lane_change_detected
                
                # 调试信息
                if lane_change_detected:
                    print(f" 车道变更检测成功! 环境{env_idx if env_idx is not None else 'N/A'}, 车道索引: {getattr(agent, 'lane_index', 'N/A')}")
                
            except Exception as e:
                print(f"⚠️ 车道变更检测异常: {e}")
                metrics['lane_change'] = False
            
            # 3. TTC计算 (Time to Collision) - 简化版本
            try:
                # 由于获取周围车辆信息较复杂，这里用启发式方法
                # 基于速度和环境状态估算
                agent_speed = getattr(agent, 'speed', 0.0)
                
                # 如果即将碰撞，TTC应该很小
                if info.get('crash', False) or info.get('crash_vehicle', False):
                    metrics['ttc'] = 0.1
                elif agent_speed > 0.1:
                    # 简化的TTC计算：基于速度和一些环境因素
                    # 实际应该基于与前车的距离和相对速度
                    base_ttc = 10.0  # 基础TTC
                    speed_factor = min(agent_speed / 10.0, 1.0)  # 速度因子
                    # 增加一些随机性来模拟不同的交通情况
                    random_factor = np.random.uniform(0.5, 1.5)
                    estimated_ttc = base_ttc * (1 - speed_factor * 0.5) * random_factor
                    metrics['ttc'] = max(estimated_ttc, 0.1)
                else:
                    metrics['ttc'] = float('inf')  # 静止时无碰撞风险
            except Exception:
                metrics['ttc'] = float('inf')
            
        except Exception as e:
            # 如果所有计算都失败，返回默认值
            metrics = {
                'lane_deviation': 0.0,
                'lane_change': False,
                'ttc': float('inf')
            }
        
        return metrics
    
    def _show_hyperparameter_override_info(self, checkpoint: Dict):
        """显示超参数覆盖信息"""
        print("\n 超参数覆盖情况:")
        
        # 检查是否有保存的args
        checkpoint_args = checkpoint.get("args", {})
        
        # 关键超参数对比 - 添加total_timesteps
        key_hyperparams = [
            ("total_timesteps", "总训练步数"),
            ("lr", "学习率"),
            ("n_steps", "rollout步数"),
            ("batch_size", "批次大小"),
            ("n_epochs", "训练轮次"),
            ("gamma", "折扣因子"),
            ("gae_lambda", "GAE lambda"),
            ("clip_range", "裁剪范围"),
            ("entropy_coef_start", "初始熵系数"),
            ("entropy_coef_end", "最终熵系数"),
            ("entropy_decay_end_ratio", "熵系数衰减比例"),
            ("vf_coef", "价值函数系数"),
            ("max_grad_norm", "梯度裁剪"),
            ("target_kl", "目标KL散度")
        ]
        
        overridden_params = []
        unchanged_params = []
        
        for param_name, param_desc in key_hyperparams:
            checkpoint_val = checkpoint_args.get(param_name, "N/A")
            current_val = getattr(self.args, param_name, "N/A")
            
            if checkpoint_val != "N/A" and current_val != "N/A":
                if checkpoint_val != current_val:
                    # 特殊处理total_timesteps显示格式
                    if param_name == "total_timesteps":
                        overridden_params.append(f"   {param_desc}: {checkpoint_val:,} → {current_val:,}")
                    else:
                        overridden_params.append(f"   {param_desc}: {checkpoint_val} → {current_val}")
                else:
                    if param_name == "total_timesteps":
                        unchanged_params.append(f"   {param_desc}: {current_val:,}")
                    else:
                        unchanged_params.append(f"   {param_desc}: {current_val}")
            elif current_val != "N/A":
                if param_name == "total_timesteps":
                    overridden_params.append(f"   {param_desc}: (新增) {current_val:,}")
                else:
                    overridden_params.append(f"   {param_desc}: (新增) {current_val}")
        
        if overridden_params:
            print(" 已覆盖的超参数:")
            for param in overridden_params:
                print(param)
        
        if unchanged_params and len(unchanged_params) <= 5:  # 只显示少量未改变的参数
            print(" 保持不变的超参数:")
            for param in unchanged_params[:5]:
                print(param)
            if len(unchanged_params) > 5:
                print(f"   ... 以及其他{len(unchanged_params)-5}个参数")
        
        if not overridden_params:
            print(" 所有超参数保持与检查点一致")
        
        print()
    
    def log_metrics(self, iteration: int, train_stats: Dict, eval_stats: Dict = None):
        """记录指标"""
        # TensorBoard日志 - 训练指标
        for key, value in train_stats.items():
            if key == "policy_loss":
                self.writer.add_scalar("loss/actor_loss", value, self.global_step)
                self.writer.add_scalar("train/policy_loss", value, self.global_step)
            elif key == "entropy_loss":
                self.writer.add_scalar("loss/entropy_loss", value, self.global_step)
            else:
                self.writer.add_scalar(f"train/{key}", value, self.global_step)
        
        # 添加学习率记录
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)
        
        # 添加熵系数记录
        self.writer.add_scalar("train/entropy_coef", self.current_entropy_coef, self.global_step)
        
        # === 渐进式训练监控 ===
        if self.is_progressive_training:
            # 记录训练阶段
            self.writer.add_scalar("progressive_training/stage", self.network.training_stage, self.global_step)
            
            # 记录冻结权重统计
            frozen_stats = self.network.get_frozen_weight_stats()
            if frozen_stats:
                self.writer.add_scalar("progressive_training/actor_frozen_norm", frozen_stats['actor_frozen_norm'], self.global_step)
                self.writer.add_scalar("progressive_training/critic_frozen_norm", frozen_stats['critic_frozen_norm'], self.global_step)
                self.writer.add_scalar("progressive_training/actor_new_norm", frozen_stats['actor_new_norm'], self.global_step)
                self.writer.add_scalar("progressive_training/critic_new_norm", frozen_stats['critic_new_norm'], self.global_step)
        
        # 记录课程阶段（便于可视化）
        if self.use_curriculum:
            stage_for_log = self.curriculum_stage if self.curriculum_mode == "gate" else (
                0 if self.global_step < 0.02 * self.args.total_timesteps else
                1 if self.global_step < 0.05 * self.args.total_timesteps else
                2 if self.global_step < 0.1 * self.args.total_timesteps else 3
            )
        else:
            stage_for_log = 0  # 未启用课程学习时显示stage 0
        self.writer.add_scalar("env/curriculum_stage", stage_for_log, self.global_step)
        
        # 记录当前交通密度（课程学习）
        if self.use_curriculum:
            current_traffic_density = self._curriculum_density()
            self.writer.add_scalar("env/curriculum_traffic_density", current_traffic_density, self.global_step)

        # === 新增：速度控制奖励可视化 ===
        if self.args.use_speed_control_reward:
            # 从环境中收集速度控制奖励数据
            speed_control_data = self._collect_speed_control_metrics()
            
            if speed_control_data:
                # 简化：只保留四个关键指标
                # 1. 总速度控制奖励
                self.writer.add_scalar("speed_control/total_reward", speed_control_data.get('r_total', 0), self.global_step)
                
                # 2. 速度偏差（当前速度与目标速度的差值）
                self.writer.add_scalar("speed_control/speed_deviation", speed_control_data.get('speed_deviation', 0), self.global_step)
                
                # 3. 超速标志（0=正常速度，1=超速）
                self.writer.add_scalar("speed_control/overspeed_flag", speed_control_data.get('overspeed_flag', 0), self.global_step)
                
                # 4. 目标参考速度
                self.writer.add_scalar("speed_control/target_speed", speed_control_data.get('target_speed', 0), self.global_step)

        # Episode环境统计
        if len(self.episode_rewards) > 0:
            self.writer.add_scalar("env/ep_rew_mean", np.mean(self.episode_rewards), self.global_step)
            self.writer.add_scalar("env/ep_rew_max", np.max(self.episode_rewards), self.global_step)
            self.writer.add_scalar("env/ep_rew_min", np.min(self.episode_rewards), self.global_step)
            
        if len(self.episode_lengths) > 0:
            self.writer.add_scalar("env/ep_len_mean", np.mean(self.episode_lengths), self.global_step)
            self.writer.add_scalar("env/ep_len_max", np.max(self.episode_lengths), self.global_step)
            self.writer.add_scalar("env/ep_len_min", np.min(self.episode_lengths), self.global_step)
            
        if len(self.episode_timeouts) > 0:
            self.writer.add_scalar("env/time_outs", np.mean(self.episode_timeouts), self.global_step)
        
        # 新增：动作统计记录
        if len(self.episode_steer_means) > 0:
            self.writer.add_scalar("actions/steer_mean", np.mean(self.episode_steer_means), self.global_step)
            self.writer.add_scalar("actions/steer_std", np.std(self.episode_steer_means), self.global_step)
            self.writer.add_scalar("actions/steer_min", np.min(self.episode_steer_means), self.global_step)
            self.writer.add_scalar("actions/steer_max", np.max(self.episode_steer_means), self.global_step)
        
        if len(self.episode_throttle_means) > 0:
            self.writer.add_scalar("actions/throttle_mean", np.mean(self.episode_throttle_means), self.global_step)
            self.writer.add_scalar("actions/throttle_std", np.std(self.episode_throttle_means), self.global_step)
            self.writer.add_scalar("actions/throttle_min", np.min(self.episode_throttle_means), self.global_step)
            self.writer.add_scalar("actions/throttle_max", np.max(self.episode_throttle_means), self.global_step)

        # === 新增：动作正则化惩罚记录 ===
        if train_stats.get("action_penalty", 0) > 0:
            self.writer.add_scalar("train/action_penalty", train_stats["action_penalty"], self.global_step)

        # 评估指标
        if eval_stats:
            for key, value in eval_stats.items():
                self.writer.add_scalar(f"eval/{key.replace('eval_', '')}", value, self.global_step)
        

        # === 新增：记录认知参数采样器指标 ===
        if self.cognitive_parameter_sampler:
            
            # 记录当前参数值
            current_params = self.cognitive_parameter_sampler.get_current_parameters()
            self.writer.add_scalar("cognitive_params/bias_inverse_tta_coef", 
                                current_params['bias_inverse_tta_coef'], 
                                self.global_step)
            self.writer.add_scalar("cognitive_params/perception_sigma0", 
                                current_params['perception_sigma0'], 
                                self.global_step)
            self.writer.add_scalar("cognitive_params/perception_k", 
                                current_params['perception_k'], 
                                self.global_step)
            self.writer.add_scalar("cognitive_params/delay_steps", 
                                current_params['delay_steps'], 
                                self.global_step)
                
                
             
        # CSV日志
        # === 修复：安全的认知参数获取 ===
        cognitive_params = {
            'bias_inverse_tta_coef': 0.0,
            'perception_sigma0': 0.0,
            'perception_k': 0.0,
            'delay_steps': 1
        }
        
        if self.cognitive_parameter_sampler:
            try:
                cognitive_params = self.cognitive_parameter_sampler.get_current_parameters()
            except Exception as e:
                print(f"⚠️ 从采样器获取认知参数失败: {e}")
                # 使用默认值继续
        # 如果没有采样器，从实际使用的模块获取参数
        elif self.use_cognitive_modules:

            # 从认知偏差模块获取实际使用的偏差系数
            if self.cognitive_bias_module:
                cognitive_params['bias_inverse_tta_coef'] = getattr(self.cognitive_bias_module, 'inverse_tta_coef', self.args.bias_inverse_tta_coef)
            
            # 从认知感知模块获取实际使用的感知参数
            if self.cognitive_perception_module:
                # 注意：感知模块的sigma0需要转换为米制单位
                sigma0_meters = getattr(self.cognitive_perception_module, 'sigma0', self.args.perception_sigma0 * 10)
                cognitive_params['perception_sigma0'] = sigma0_meters / 10.0  # 转换回原始单位
                cognitive_params['perception_k'] = getattr(self.cognitive_perception_module, 'k', self.args.perception_k)
            
            # 从认知延迟模块获取实际使用的延迟参数
            if self.cognitive_delay_module:
                cognitive_params['delay_steps'] = getattr(self.cognitive_delay_module, 'delay_steps', self.args.delay_steps)
            
            print(f"✅ 已获取真实使用的认知参数: {cognitive_params}")


        
        log_data = [
            self.global_step, iteration,
            eval_stats.get("eval_reward_mean", 0) if eval_stats else 0,
            eval_stats.get("eval_length_mean", 0) if eval_stats else 0,
            train_stats.get("policy_loss", 0),
            train_stats.get("value_loss", 0),
            train_stats.get("entropy", 0),
            train_stats.get("approx_kl", 0),
            current_lr,
            self.current_entropy_coef,  # 添加熵系数
            eval_stats.get("eval_collision_rate", 0) if eval_stats else 0,
            eval_stats.get("eval_offroad_rate", 0) if eval_stats else 0,
            eval_stats.get("eval_success_rate", 0) if eval_stats else 0,
            train_stats.get("fps", 0),
            train_stats.get('clipfrac', 0),
            train_stats.get('explained_variance', 0),
            train_stats.get('grad_norm', 0),
            eval_stats.get('eval_avg_speed', 0) if eval_stats else 0,
            eval_stats.get('eval_lane_deviation', 0) if eval_stats else 0,
            eval_stats.get('eval_lane_change_count', 0) if eval_stats else 0,
            eval_stats.get('eval_min_ttc', 0) if eval_stats else 0,
            eval_stats.get('eval_path_completion', 0) if eval_stats else 0,
            # 新增：动作统计数据
            np.mean(self.episode_steer_means) if len(self.episode_steer_means) > 0 else 0,
            np.std(self.episode_steer_means) if len(self.episode_steer_means) > 0 else 0,
            np.mean(self.episode_throttle_means) if len(self.episode_throttle_means) > 0 else 0,
            np.std(self.episode_throttle_means) if len(self.episode_throttle_means) > 0 else 0,
            # 新增：动作正则化惩罚数据
            train_stats.get('action_penalty', 0),

            # === 修复：使用真实使用的认知参数数据 ===
            cognitive_params['bias_inverse_tta_coef'],
            cognitive_params['perception_sigma0'],
            cognitive_params['perception_k'],
            cognitive_params['delay_steps']
        ]
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_data)
        
        # 控制台输出
        reward_str = f"{eval_stats.get('eval_reward_mean', 0):.3f}" if eval_stats else "N/A"
        print(f"Step {self.global_step:8d} | Iter {iteration:4d} | "
              f"Reward: {reward_str} | "
              f"Policy Loss: {train_stats.get('policy_loss', 0):.6f} | "
              f"Value Loss: {train_stats.get('value_loss', 0):.6f}")
        
        # 每隔一定步数输出详细统计
        if iteration % (self.args.eval_freq * 2) == 0 and eval_stats:
            print(f"详细统计 (Step {self.global_step}):")
            print(f"   平均速度: {eval_stats.get('eval_avg_speed', 0):.2f}")
            print(f"   车道偏移: {eval_stats.get('eval_lane_deviation', 0):.3f}")
            print(f"   路径完成: {eval_stats.get('eval_path_completion', 0):.3f}")
            print(f"   成功率: {eval_stats.get('eval_success_rate', 0):.3f}")
            print(f"   当前熵系数: {self.current_entropy_coef:.4f}")
            
            # === 渐进式训练状态输出 ===
            if self.is_progressive_training:
                print(f"🧠 渐进式训练状态:")
                print(f"   训练阶段: {self.network.training_stage} (1=冻结, 2=解冻)")
                print(f"   冻结阈值: {self.network.freeze_threshold_steps:,} 步")
                print(f"   距离解冻: {max(0, self.network.freeze_threshold_steps - self.global_step):,} 步")
                
                # 显示冻结权重统计
                frozen_stats = self.network.get_frozen_weight_stats()
                if frozen_stats:
                    print(f"   冻结权重统计:")
                    print(f"     Actor前275维: 均值={frozen_stats['actor_frozen_mean']:.4f}, 范数={frozen_stats['actor_frozen_norm']:.4f}")
                    print(f"     Critic前275维: 均值={frozen_stats['critic_frozen_mean']:.4f}, 范数={frozen_stats['critic_frozen_norm']:.4f}")
                    print(f"     新增4维: Actor={frozen_stats['actor_new_norm']:.4f}, Critic={frozen_stats['critic_new_norm']:.4f}")
            
            # 新增：动作统计输出
            if len(self.episode_steer_means) > 0:
                print(f"   转向均值: {np.mean(self.episode_steer_means):.3f} ± {np.std(self.episode_steer_means):.3f}")
            if len(self.episode_throttle_means) > 0:
                print(f"   油门均值: {np.mean(self.episode_throttle_means):.3f} ± {np.std(self.episode_throttle_means):.3f}")

            if train_stats.get('clipfrac', 0) > 0:
                print(f"   Clip Fraction: {train_stats.get('clipfrac', 0):.3f}")
                print(f"   Explained Var: {train_stats.get('explained_variance', 0):.3f}")
            
            # === 新增：动作正则化惩罚输出 ===
            if train_stats.get('action_penalty', 0) > 0:
                print(f"   动作正则化惩罚: {train_stats.get('action_penalty', 0):.6f}")

            # 新增：速度控制奖励统计输出
            if self.args.use_speed_control_reward:
                speed_control_data = self._collect_speed_control_metrics()
                if speed_control_data:
                    print(f"   速度控制奖励:")
                    print(f"     总奖励: {speed_control_data.get('r_total', 0):.3f}")
                    print(f"     速度偏差: {speed_control_data.get('speed_deviation', 0):.2f} m/s")
                    print(f"     超速标志: {'是' if speed_control_data.get('overspeed_flag', 0) > 0 else '否'}")
                    print(f"     目标速度: {speed_control_data.get('target_speed', 0):.2f} m/s")
                    
                    # === 新增：子模块状态输出 ===
                    tracking_status = "✅" if speed_control_data.get('speed_control_enable_tracking', True) else "❌"
                    soft_wall_status = "✅" if speed_control_data.get('speed_control_enable_soft_wall', True) else "❌"
                    behavior_status = "✅" if speed_control_data.get('speed_control_enable_behavior_guidance', True) else "❌"
                    print(f"     子模块状态: 跟踪{tracking_status} 软墙{soft_wall_status} 行为{behavior_status}")
                    
                    # 显示各子模块的贡献
                    if speed_control_data.get('speed_control_enable_tracking', True):
                        print(f"     跟踪奖励: {speed_control_data.get('r_track', 0):.3f}")
                    if speed_control_data.get('speed_control_enable_soft_wall', True):
                        print(f"     软墙惩罚: {speed_control_data.get('r_wall', 0):.3f}")
                    if speed_control_data.get('speed_control_enable_behavior_guidance', True):
                        print(f"     行为导向: {speed_control_data.get('r_act_over', 0):.3f}")
    
    def train(self):
        """主训练循环"""
        print(f" 开始PPO训练 - 目标步数: {self.args.total_timesteps:,}")
        
        # 设置起始迭代号
        start_iteration = getattr(self, 'start_iteration', 0)
        iteration = start_iteration
        start_time = time.time()
        best_reward = float('-inf')
        
        # 如果是恢复训练，尝试获取历史最佳奖励
        if hasattr(self, 'start_iteration') and self.start_iteration > 0:
            print(f" 从迭代 {self.start_iteration} 恢复训练")
        
        while self.global_step < self.args.total_timesteps:
            iteration += 1
            
            # 更新熵系数
            current_entropy = self._update_entropy_coef()
            
            # 更新课程学习环境参数
            self._update_env_curriculum()
            
            # === 渐进式训练：更新训练阶段 ===
            if self.is_progressive_training:
                self.network.update_training_stage(self.global_step)
            
            # 收集rollouts
            rollout_start = time.time()
            rollouts = self.collect_rollouts()
            rollout_time = time.time() - rollout_start
            
            # 更新学习率（每个迭代生效）
            new_lr = self._compute_scheduled_lr()
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

            # 更新策略
            update_start = time.time()
            train_stats = self.update_policy(*rollouts)
            update_time = time.time() - update_start
            
            # 计算FPS
            fps = (self.args.n_steps * self.args.n_envs) / (rollout_time + update_time)
            train_stats["fps"] = fps
            train_stats["entropy_coef"] = self.current_entropy_coef  # 记录当前熵系数
            
            # 定期评估
            eval_stats = None
            if iteration % self.args.eval_freq == 0:
                eval_stats = self.evaluate()
                
                # ===== gate 模式晋升判定 =====
                if self.use_curriculum and self.curriculum_mode == "gate":
                    succ = eval_stats.get("eval_success_rate", 0.0)
                    coll = eval_stats.get("eval_collision_rate", 1.0)
                    if self.curriculum_stage < 3 and succ >= self.args.gate_succ_threshold and coll <= self.args.gate_coll_threshold:
                        self.curriculum_stage += 1
                        print(f" 课程晋升 -> Stage {self.curriculum_stage}")

                # ===== 指标阈值达标检查 =====
                reward_mean = eval_stats.get("eval_reward_mean", 0.0)
                success_rate = eval_stats.get("eval_success_rate", 0.0)
                collision_rate = eval_stats.get("eval_collision_rate", 1.0)
                offroad_rate = eval_stats.get("eval_offroad_rate", 1.0)
                
                # 检查是否同时满足所有阈值条件
                # 增加课程学习条件：必须进入stage 2才开始检查指标达标
                if self.curriculum_mode == "gate":
                    current_stage = self.curriculum_stage
                else:
                    # progress模式：根据训练进度计算stage - 修改分界点
                    p = min(1.0, float(self.global_step) / float(self.args.total_timesteps + 1e-8))
                    if p < 0.02:
                        current_stage = 0
                    elif p < 0.05:
                        current_stage = 1
                    elif p < 0.1:
                        current_stage = 2
                    else:
                        current_stage = 3
                
                if (current_stage >= 2 and  # 课程学习必须进入stage 2
                    reward_mean >= 200 and 
                    success_rate >= 0.70 and 
                    collision_rate <= 0.15 and 
                    offroad_rate <= 0.15):
                    
                    print(f"�� 指标达标检测到！")
                    print(f"   当前课程阶段: Stage {current_stage} ✅")
                    print(f"   平均奖励: {reward_mean:.3f} ≥ 200.0 ✅")
                    print(f"   成功率: {success_rate:.3f} ≥ 0.70 ✅")
                    print(f"   碰撞率: {collision_rate:.3f} ≤ 0.15 ✅")
                    print(f"   冲出道路率: {offroad_rate:.3f} ≤ 0.15 ✅")
                    
                    # 保存达标检查点（特殊命名）
                    milestone_checkpoint_path = os.path.join(
                        self.exp_dir, "checkpoints", 
                        f"milestone_checkpoint_iter{iteration}_stage{current_stage}_reward{reward_mean:.1f}_succ{success_rate:.2f}.pt"
                    )
                    
                    milestone_checkpoint = {
                        "iteration": iteration,
                        "global_step": self.global_step,
                        "network_state_dict": self.network.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.config,
                        "args": vars(self.args),
                        "milestone_metrics": {
                            "eval_reward_mean": reward_mean,
                            "eval_success_rate": success_rate,
                            "eval_collision_rate": collision_rate,
                            "eval_offroad_rate": offroad_rate,
                            "curriculum_stage": current_stage,  # 记录达到里程碑时的课程阶段
                            "milestone_achieved": True,
                            "milestone_timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    torch.save(milestone_checkpoint, milestone_checkpoint_path)
                    print(f" 里程碑检查点已保存: {os.path.basename(milestone_checkpoint_path)}")
                    print(f" 完整路径: {milestone_checkpoint_path}")

                # 检查是否为最佳模型
                is_best = eval_stats["eval_reward_mean"] > best_reward
                if is_best:
                    best_reward = eval_stats["eval_reward_mean"]
                
                # 保存检查点
                if iteration % self.args.checkpoint_freq == 0:
                    self.save_checkpoint(iteration, is_best)
            
            # 记录指标
            if iteration % self.args.log_freq == 0:
                self.log_metrics(iteration, train_stats, eval_stats)
        
        # 训练结束
        print(f"✅ 训练完成！总用时: {time.time() - start_time:.2f}秒")
        
        # 最终评估和保存
        final_eval = self.evaluate(num_episodes=20)
        self.save_checkpoint(iteration, False)
        
        # 生成最终报告
        self.generate_final_report(final_eval)
        
        # 关闭资源
        self.writer.close()
        
        # === 认知模块：分离环境 ===
        if self.use_cognitive_modules:
            # 生成认知模块可视化（如果启用）
            if self.enable_cognitive_visualization and self.cognitive_perception_module:
                try:
                    viz_dir = os.path.join(self.exp_dir, "cognitive_visualization")
                    os.makedirs(viz_dir, exist_ok=True)
                    # 注意：需要在环境关闭前生成可视化
                    if hasattr(self.envs, 'envs') and len(self.envs.envs) > 0:
                        self.cognitive_perception_module.generate_visualization(save_dir=viz_dir, env=self.envs.envs[0])
                        print(" 认知感知模块可视化已生成")
                except Exception as e:
                    print(f"⚠️ 认知感知模块可视化生成失败: {e}")
            
            # 生成偏差模块可视化
            if self.cognitive_bias_module:
                try:
                    viz_dir = os.path.join(self.exp_dir, "cognitive_visualization")
                    os.makedirs(viz_dir, exist_ok=True)
                    if hasattr(self.envs, 'envs') and len(self.envs.envs) > 0:
                        self.cognitive_bias_module.generate_visualization(env=self.envs.envs[0], save_dir=viz_dir)
                        print(" 认知偏差模块可视化已生成")
                except Exception as e:
                    print(f"⚠️ 认知偏差模块可视化生成失败: {e}")
            
            # === 新增：生成认知参数采样器可视化 ===
            if self.cognitive_parameter_sampler:
                try:
                    viz_dir = os.path.join(self.exp_dir, "cognitive_visualization")
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    # 添加调试信息
                    sampler_stats = self.cognitive_parameter_sampler.get_statistics()
                 
                    
                    # 如果没有足够的历史数据，强制记录当前参数
                    if len(self.cognitive_parameter_sampler.param_history) < 2:
                        print("⚠️ 历史数据不足，强制记录当前参数...")
                        current_params = self.cognitive_parameter_sampler.get_current_parameters()
                        self.cognitive_parameter_sampler._record_parameter_update(
                            self.global_step, "forced_record"
                        )
                        print(f"✅ 已强制记录当前参数: {current_params}")
                    
                    # 生成参数采样可视化图表
                    viz_file = self.cognitive_parameter_sampler.generate_parameter_visualization(
                        output_dir=os.path.join(viz_dir, "parameter_sampling"),
                        session_name=f"ppo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    if viz_file:
                        print(" 认知参数采样器可视化已生成")
                    else:
                        print("⚠️ 认知参数采样器可视化生成失败")
                    
                    # 保存参数采样历史到文件
                    history_file = os.path.join(viz_dir, "parameter_sampling_history.json")
                    self.cognitive_parameter_sampler.save_history_to_file(history_file)
                    print(" 认知参数采样历史已保存")
                    
                except Exception as e:
                    print(f"⚠️ 认知参数采样器可视化生成失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 分离认知模块
            self._detach_cognitive_modules_from_env()
        
        # 关闭向量化环境
        self.envs.close()
        print(f" 训练环境已安全关闭")
    
    def generate_final_report(self, final_eval: Dict):
        """生成最终报告"""
        report_path = os.path.join(self.exp_dir, "report.md")
        
        # 加载训练数据
        df = pd.read_csv(self.csv_path)
        
        report_content = f"""# MetaDrive PPO Expert 复现训练报告

##  背景与目标

本实验旨在复现MetaDrive PPO Expert的训练过程，严格对齐网络结构、观测空间、动作空间和环境配置，仅对关键超参数进行可控调整。

{f'''##  认知模块集成

本次训练集成了以下认知模块：

### 认知偏差模块（风险厌恶）
- **状态**: {'启用' if self.args.use_cognitive_bias else '禁用'}
- **偏差强度系数**: {self.args.bias_inverse_tta_coef if self.args.use_cognitive_bias else 'N/A'}
- **TTA阈值**: {self.args.bias_tta_threshold if self.args.use_cognitive_bias else 'N/A'}
- **视觉检测距离**: {self.args.bias_visual_distance if self.args.use_cognitive_bias else 'N/A'}米
- **视觉检测角度**: {self.args.bias_visual_angle if self.args.use_cognitive_bias else 'N/A'}度

### 认知延迟模块（动作延迟）
- **状态**: {'启用' if self.args.use_cognitive_delay else '禁用'}
- **延迟步数**: {self.args.delay_steps if self.args.use_cognitive_delay else 'N/A'}
- **动作平滑**: {'启用' if self.args.delay_smoothing else 'N/A'}
- **平滑系数**: {self.args.delay_smoothing_factor if self.args.use_cognitive_delay else 'N/A'}

### 认知感知模块（观测噪声）
- **状态**: {'启用' if self.args.use_cognitive_perception else '禁用'}
- **基础噪声**: {self.args.perception_sigma0 if self.args.use_cognitive_perception else 'N/A'}米
- **卡尔曼滤波**: {'启用' if self.args.perception_use_kf else 'N/A'}
''' if self.args.use_cognitive_modules else '## 认知模块未启用'}

##  实验配置

### 网络结构 (严格对齐Expert + 认知参数集成)
- **观测维度**: 279 (Lidar: 240 + State: 35 + 4维认知参数)
- **动作维度**: 2 (连续控制: 转向 + 油门/刹车)
- **隐藏层**: 256 -> 256
- **激活函数**: Tanh

### 环境配置 (直线场景生成)
- **场景类型**: 动态直线道路
- **场景数量**: 1000个互不相同的直线场景
- **道路长度**: 200-800米动态变化
- **交通密度**: {self.args.traffic_density_min}-{self.args.traffic_density_max}动态变化
- **并行环境**: {self.args.n_envs} (真正的多进程并行)
- **Lidar配置**: 240束激光，50米距离，4个其他车辆
- **随机种子**: {self.args.seed}
- **交通随机化**: 启用

### 关键超参数 (可调整)
- **学习率**: {self.args.lr}
- **rollout步数**: {self.args.n_steps}
- **环境数量**: {self.args.n_envs}
- **批次大小**: {self.args.batch_size}
- **训练轮次**: {self.args.n_epochs}
- **折扣因子**: {self.args.gamma}
- **GAE Lambda**: {self.args.gae_lambda}
- **裁剪范围**: {self.args.clip_range}
- **熵系数**: {self.args.entropy_coef}

##  训练结果

### 最终性能
- **平均奖励**: {final_eval['eval_reward_mean']:.3f} ± {final_eval['eval_reward_std']:.3f}
- **平均长度**: {final_eval['eval_length_mean']:.1f}
- **碰撞率**: {final_eval['eval_collision_rate']:.3f}
- **冲出道路率**: {final_eval['eval_offroad_rate']:.3f}
- **成功率**: {final_eval['eval_success_rate']:.3f}

### 训练统计
- **总训练步数**: {self.global_step:,}
- **最终策略损失**: {df['policy_loss'].iloc[-1]:.6f}
- **最终值函数损失**: {df['value_loss'].iloc[-1]:.6f}
- **最终熵值**: {df['entropy'].iloc[-1]:.6f}

##  使用方法

### 基础训练
```bash
python ppo_expert_reproduction.py
```

### 自定义超参数
```bash
python ppo_expert_reproduction.py \\
    --lr 3e-4 \\
    --n_steps 2048 \\
    --n_envs 8 \\
    --batch_size 256 \\
    --clip_range 0.2
```

### 高性能并行训练
```bash
python ppo_expert_reproduction.py \\
    --n_envs 16 \\
    --n_steps 1024 \\
    --batch_size 512 \\
    --total_timesteps 2000000
```

### 命令行参数一览
- `--lr`: 学习率 (默认: 3e-4)
- `--n_steps`: rollout步数 (默认: 2048)
- `--n_envs`: 并行环境数 (默认: 4, 支持真正的多进程并行)
- `--batch_size`: 批次大小 (默认: 256)
- `--n_epochs`: 训练轮次 (默认: 10)
- `--gamma`: 折扣因子 (默认: 0.99)
- `--gae_lambda`: GAE lambda (默认: 0.95)
- `--clip_range`: PPO裁剪范围 (默认: 0.2)
- `--entropy_coef`: 熵系数 (默认: 0.01)
- `--total_timesteps`: 总训练步数 (默认: 1,000,000)

##  可视化说明

### TensorBoard监控
```bash
tensorboard --logdir {self.exp_dir}/tensorboard
```

关键监控指标:
- `train/policy_loss`: 策略损失
- `train/value_loss`: 值函数损失
- `train/entropy`: 策略熵
- `train/approx_kl`: 近似KL散度
- `eval/eval_reward_mean`: 评估平均奖励
- `eval/eval_collision_rate`: 碰撞率
- `eval/eval_success_rate`: 成功率

##  评估协议

### 验证设置
- **验证环境**: 与训练环境相同配置
- **验证频率**: 每{self.args.eval_freq}次迭代
- **验证episode数**: 10 (最终评估20)
- **确定性策略**: 使用动作均值

### 最优模型选择标准
1. **主要指标**: 验证集平均episode奖励
2. **约束条件**: 碰撞率不劣化
3. **辅助指标**: 成功率、episode长度

##  复现性保证

### 依赖版本
- Python: {sys.version.split()[0]}
- PyTorch: {torch.__version__}
- NumPy: {np.__version__}

### 随机性控制
- **全局种子**: {self.args.seed}
- **PyTorch种子**: 已设置
- **NumPy种子**: 已设置

### 硬件环境
- **计算设备**: {self.device}
- **训练时长**: 预计2-4小时 (取决于硬件)

##  产物说明

```
{os.path.basename(self.exp_dir)}/
├── config.json              # 完整配置文件
├── training_logs.csv         # 训练过程CSV日志
├── tensorboard/              # TensorBoard事件文件
├── checkpoints/              # 模型检查点
│   ├── best_model.pt         # 最佳模型
│   ├── latest_model.pt       # 最新模型
│   └── checkpoint_*.pt       # 定期检查点
└── report.md                 # 本报告文件
```

##  扩展说明

### 添加新超参数
1. 在`add_arguments()`函数中添加参数定义
2. 在`_build_config()`中添加配置项
3. 在相应训练逻辑中使用新参数

### 多场景训练
修改`env_config`中的`num_scenarios`和`map`参数：
```python
env_config.update({{
    "num_scenarios": 5000,  # 更多场景
    "map": "SSSSSSSS"      # 自定义地图
}})
```

### 自定义奖励函数
继承`MetaDriveEnv`并重写`reward_function`方法。

---
**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**实验目录**: `{self.exp_dir}`
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f" 最终报告已生成: {report_path}")
    
    def _validate_and_log_scenarios(self):
        """验证和记录直线场景生成参数"""
        print(f"\n️ 直线场景配置验证:")
        print(f"   场景类型: 动态直线道路")
        print(f"   总场景数: 1000")
        print(f"   道路长度: 2-10个直线段 (每段约50-80米)")
        print(f"   交通密度范围: {self.args.traffic_density_min}-{self.args.traffic_density_max}")
        print(f"   交通随机化: 启用")
        if self.use_curriculum:
            print(f"   课程学习分界点: [0.0, 0.02, 0.05, 0.1, 1.0]")
            print(f"   Stage 0: 0.0-2% 训练进度")
            print(f"   Stage 1: 2%-5% 训练进度")
            print(f"   Stage 2: 5%-10% 训练进度")
            print(f"   Stage 3: 10%-100% 训练进度")
        
        # 生成几个示例场景参数用于验证
        sample_scenarios = []
        for i in [0, 100, 500, 999]:
            scenario_index = (self.args.seed + i) % 1000
            num_segments = 2 + (scenario_index * 8) // 1000  # 2-10段
            traffic_density = self.args.traffic_density_min + (scenario_index * (self.args.traffic_density_max - self.args.traffic_density_min)) / 1000
            map_string = "S" * num_segments
            estimated_length = num_segments * 65  # 估算长度（每段约65米）
            sample_scenarios.append({
                "index": i,
                "seed": self.args.seed + i,
                "num_segments": num_segments,
                "map_string": map_string,
                "estimated_length": estimated_length,
                "traffic_density": traffic_density
            })
        
        print(f"\n 场景示例:")
        for scenario in sample_scenarios:
            print(f"   场景{scenario['index']:3d}: {scenario['num_segments']:2d}段 ({scenario['estimated_length']:3d}m), "
                  f"密度={scenario['traffic_density']:.3f}, 地图='{scenario['map_string']}'")
        
        # 保存场景配置到文件
        scenario_config_path = os.path.join(self.exp_dir, "scenario_config.json")
        scenario_config = {
            "scenario_type": "straight_road_segments",
            "total_scenarios": 1000,
            "segments_range": [2, 10],
            "estimated_length_range": [130, 650],  # 2*65 到 10*65
            "traffic_density_range": [self.args.traffic_density_min, self.args.traffic_density_max],
            "random_traffic": True,
            "sample_scenarios": sample_scenarios,
            "generation_formula": {
                "num_segments": "2 + (scenario_index * 8) // 1000",
                "traffic_density": f"{self.args.traffic_density_min} + (scenario_index * ({self.args.traffic_density_max} - {self.args.traffic_density_min})) / 1000",
                "map_string": "'S' * num_segments"
            }
        }
        
        # 添加课程学习配置信息
        if self.use_curriculum:
            scenario_config["curriculum_learning"] = {
                "enabled": True,
                "stage_boundaries": [0.0, 0.02, 0.05, 0.1, 1.0],
                "stage_descriptions": {
                    "stage_0": "0.0-2% 训练进度: 简单场景",
                    "stage_1": "2%-5% 训练进度: 中等场景", 
                    "stage_2": "5%-10% 训练进度: 复杂场景",
                    "stage_3": "10%-100% 训练进度: 完整场景"
                }
            }
        
        with open(scenario_config_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_config, f, indent=2, ensure_ascii=False)
        
        print(f" 场景配置已保存: {scenario_config_path}")
        print("=" * 50)

    def _create_single_environment(self):
        """创建单个环境实例"""
        return MetaDriveEnv(self._get_base_env_config())
    
    def _attach_cognitive_modules_to_env(self, env):
        """将认知模块附加到环境"""
        if not self.use_cognitive_modules:
            return
            
        # 附加感知模块（必须先于偏差模块）
        if self.cognitive_perception_module:
            self.cognitive_perception_module.reset()
            self.cognitive_perception_module.attach_to_env(env)
            print(" 认知感知模块已附加到环境 - 噪声将在传感器层自动注入")
            
            # 验证环境配置，确保避免双重噪声
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
            
        # 附加偏差模块
        if self.cognitive_bias_module:
            success = self.cognitive_bias_module.attach_to_env(env)
            if success:
                print(" 认知偏差模块已附加到环境 - 将基于TTA动态调整奖励")
            else:
                print("⚠️ 认知偏差模块附加失败")
    
    def _detach_cognitive_modules_from_env(self):
        """从环境分离认知模块"""
        if not self.use_cognitive_modules:
            return
            
        if self.cognitive_perception_module:
            self.cognitive_perception_module.detach_from_env()
            print(" 认知感知模块已从环境分离")
            
        if self.cognitive_bias_module:
            try:
                self.cognitive_bias_module.detach_from_env()
                print(" 认知偏差模块已从环境分离")
            except Exception as e:
                print(f"⚠️ 认知偏差模块分离失败: {e}")
    
    def _apply_cognitive_parameters(self, new_params: Dict[str, Any]):
        """
        将新采样的认知参数应用到相应的认知模块
        
        Args:
            new_params (Dict[str, Any]): 新的参数字典
        """
 
        # 更新认知偏差模块参数
        if self.cognitive_bias_module and 'bias_inverse_tta_coef' in new_params:
            old_coef = getattr(self.cognitive_bias_module, 'inverse_tta_coef', 'N/A')
            self.cognitive_bias_module.inverse_tta_coef = new_params['bias_inverse_tta_coef']
            
        # 更新认知感知模块参数
        if self.cognitive_perception_module:
            if 'perception_sigma0' in new_params:
                old_sigma0 = getattr(self.cognitive_perception_module, 'sigma0', 'N/A')
                # 注意：感知模块的sigma0需要转换为米制单位
                new_sigma0 = new_params['perception_sigma0'] * 10  # 转换为米制
                self.cognitive_perception_module.sigma0 = new_sigma0
                
            if 'perception_k' in new_params:
                old_k = getattr(self.cognitive_perception_module, 'k', 'N/A')
                self.cognitive_perception_module.k = new_params['perception_k']
                
        # 更新认知延迟模块参数
        if self.cognitive_delay_module and 'delay_steps' in new_params:
            old_delay = getattr(self.cognitive_delay_module, 'delay_steps', 'N/A')
            self.cognitive_delay_module.update_config(delay_steps=new_params['delay_steps'])
              
         
     
    def _concatenate_cognitive_params(self, obs, cognitive_params):
        """
        将认知参数拼接到观测向量中
        
        Args:
            obs: 原始观测 [n_envs, 275] 或 [batch_size, 275]
            cognitive_params: 认知参数字典
        
        Returns:
            扩展后的观测 [n_envs, 279] 或 [batch_size, 279]
        """
        if not self.use_cognitive_modules:
            # 如果未启用认知模块，返回原始观测
            return obs
        
        # 确保obs是numpy数组
        if torch.is_tensor(obs):
            obs_np = obs.cpu().numpy()
        else:
            obs_np = obs
        
        # 提取认知参数值
        bias_coef = cognitive_params.get('bias_inverse_tta_coef', 1.0)
        sigma0 = cognitive_params.get('perception_sigma0', 0.1)
        k = cognitive_params.get('perception_k', 0.02)
        delay = cognitive_params.get('delay_steps', 2)
        
        # 构建认知参数向量
        cognitive_vector = np.array([
            [bias_coef, sigma0, k, delay] for _ in range(obs_np.shape[0])
        ], dtype=np.float32)
        
        # 拼接原始观测和认知参数
        obs_with_cognitive = np.concatenate([obs_np, cognitive_vector], axis=1)
        
        return obs_with_cognitive

    def _collect_speed_control_metrics(self):
        """
        收集速度控制奖励的指标数据
        
        Returns:
            dict: 包含速度控制奖励各项指标的数据字典
        """
        if not self.args.use_speed_control_reward:
            return None
        
        try:
            # 从环境中收集数据
            speed_control_data = {}
            
            # 获取第一个环境实例（用于数据收集）
            env_instance = None
            if hasattr(self.envs, 'envs') and len(self.envs.envs) > 0:
                env_instance = self.envs.envs[0]
            elif hasattr(self.envs, 'venv') and hasattr(self.envs.venv, 'envs'):
                if len(self.envs.venv.envs) > 0:
                    env_instance = self.envs.venv.envs[0]
            
            if env_instance is None:
                return None
            
            # 检查是否是SpeedControlMetaDriveEnv
            if not hasattr(env_instance, '_compute_speed_control_reward'):
                return None
            
            # 获取主车辆
            if hasattr(env_instance, 'agent') and env_instance.agent:
                vehicle = env_instance.agent
            elif hasattr(env_instance, 'agents') and len(env_instance.agents) > 0:
                # 获取第一个智能体
                vehicle_id = list(env_instance.agents.keys())[0]
                vehicle = env_instance.agents[vehicle_id]
            else:
                return None
            
            # 计算速度控制奖励
            try:
                # 模拟一个动作（使用当前状态）
                current_action = [0.0, 0.0]  # 默认动作
                if hasattr(vehicle, 'current_action'):
                    current_action = vehicle.current_action
                
                # 调用速度控制奖励计算
                total_reward = env_instance._compute_speed_control_reward(vehicle, current_action)
                
                # 从step_infos中获取详细数据
                if hasattr(env_instance, 'step_infos') and vehicle.id in env_instance.step_infos:
                    step_info = env_instance.step_infos[vehicle.id]
                    
                    speed_control_data.update({
                        'r_total': step_info.get('sc_r_total', total_reward),
                        'r_track': step_info.get('sc_r_track', 0.0),
                        'r_wall': step_info.get('sc_r_wall', 0.0),
                        'r_act_over': step_info.get('sc_r_act_over', 0.0),
                        'current_speed': step_info.get('sc_v', vehicle.speed),
                        'target_speed': step_info.get('sc_v_ref', env_instance.v_ref),
                        'speed_deviation': step_info.get('sc_dv', vehicle.speed - env_instance.v_ref),
                        'acceleration': step_info.get('sc_a', 0.0),
                        # === 新增：子模块启用状态 ===
                        'speed_control_enable_tracking': step_info.get('sc_enable_tracking', True),
                        'speed_control_enable_soft_wall': step_info.get('sc_enable_soft_wall', True),
                        'speed_control_enable_behavior_guidance': step_info.get('sc_enable_behavior_guidance', True),
                    })
                else:
                    # 如果没有step_infos，使用基本数据
                    current_speed = vehicle.speed
                    target_speed = env_instance.v_ref
                    speed_deviation = current_speed - target_speed
                    
                    speed_control_data.update({
                        'r_total': total_reward,
                        'r_track': 0.0,  # 需要实际计算
                        'r_wall': 0.0,   # 需要实际计算
                        'r_act_over': 0.0, # 需要实际计算
                        'current_speed': current_speed,
                        'target_speed': target_speed,
                        'speed_deviation': speed_deviation,
                        'acceleration': 0.0,
                    })
                
                # 计算额外指标
                current_speed = speed_control_data['current_speed']
                target_speed = speed_control_data['target_speed']
                speed_deviation = speed_control_data['speed_deviation']
                acceleration = speed_control_data['acceleration']
                
                # 速度比率
                speed_control_data['speed_ratio'] = current_speed / max(target_speed, 0.1) if target_speed > 0 else 0.0
                
                # 加速度分析
                speed_control_data['acceleration_positive'] = max(acceleration, 0.0)
                speed_control_data['acceleration_negative'] = max(-acceleration, 0.0)
                
                # 超速标志
                speed_control_data['overspeed_flag'] = 1.0 if speed_deviation > 0 else 0.0
                
                # 奖励统计（使用当前值作为示例）
                speed_control_data['reward_mean'] = speed_control_data['r_total']
                speed_control_data['reward_std'] = 0.0
                speed_control_data['reward_min'] = speed_control_data['r_total']
                speed_control_data['reward_max'] = speed_control_data['r_total']
                
                return speed_control_data
                
            except Exception as e:
                print(f"⚠️ 速度控制奖励计算失败: {e}")
                return None
                
        except Exception as e:
            print(f"⚠️ 收集速度控制指标失败: {e}")
            return None



def add_arguments():
    """添加命令行参数"""
    parser = argparse.ArgumentParser(description="MetaDrive PPO Expert 复现训练")
    
    # ===== 关键超参数 (可调整) =====
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="学习率 (默认: 3e-4)")
    parser.add_argument("--lr_schedule", type=str, default="linear",
                        choices=["constant", "linear", "cosine", "stage"],
                        help="学习率日程 (默认: linear)")
    parser.add_argument("--lr_min", type=float, default=3e-5,
                        help="最小学习率（线性/余弦的下限）")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup占总步数比例 (默认: 0.05)")

    parser.add_argument("--n_steps", type=int, default=2048,
                       help="rollout步数 (默认: 2048)")
    parser.add_argument("--n_envs", type=int, default=4,
                       help="并行环境数量 (默认: 4)")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="SGD批次大小 (默认: 256)")
    parser.add_argument("--n_epochs", type=int, default=10,
                       help="每次更新的训练轮次 (默认: 10)")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="折扣因子 (默认: 0.99)")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                       help="GAE lambda参数 (默认: 0.95)")
    parser.add_argument("--clip_range", type=float, default=0.2,
                       help="PPO裁剪范围 (默认: 0.2)")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                       help="熵系数 (默认: 0.01)")
    
    # ===== 其他超参数 (预留扩展) =====
    parser.add_argument("--vf_coef", type=float, default=0.5,
                       help="值函数损失系数 (默认: 0.5)")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                       help="梯度裁剪阈值 (默认: 0.5)")
    parser.add_argument("--target_kl", type=float, default=None,
                       help="目标KL散度 (早停, 默认: None)")
    
    # ===== 熵系数衰减参数 ( 修复5: 调整熵系数) =====
    parser.add_argument("--entropy_coef_start", type=float, default=0.01,
                       help="初始熵系数 (默认: 0.01, 降低自0.015)")
    parser.add_argument("--entropy_coef_end", type=float, default=0.001,
                       help="最终熵系数 (默认: 0.001, 降低自0.005)")
    parser.add_argument("--entropy_decay_end_ratio", type=float, default=0.5,
                       help="熵系数衰减完成的训练进度比例 (默认: 0.5, 降低自0.8)")
    
    # ===== 奖励配置参数 ( 修复4: 回调到合理量级) =====
    parser.add_argument("--success_reward", type=float, default=20.0,
                       help="成功奖励 (默认: 10.0, 回调自20.0)")
    parser.add_argument("--driving_reward", type=float, default=1,
                       help="前进奖励 (默认: 1.0)")
    parser.add_argument("--speed_reward", type=float, default=0,
                       help="速度奖励 (默认: 0, 回调自0.3)")
    parser.add_argument("--use_lateral_reward", action="store_true", default=True,
                       help="启用车道保持奖励，车辆保持在导航车道上 (默认: True)")
    parser.add_argument("--out_of_road_penalty", type=float, default=8.0,
                       help="冲出道路惩罚 (默认: 5.0, 回调自8.0)")
    parser.add_argument("--crash_penalty", type=float, default=8.0,
                       help="碰撞惩罚 (默认: 5.0, 回调自8.0)")
    parser.add_argument("--crash_sidewalk_penalty", type=float, default=2.0,
                       help="碰撞人行道惩罚 (默认: 8.0, 回调自8.0)")
    
    # ===== 交通密度配置参数 (新增) =====
    parser.add_argument("--traffic_density_min", type=float, default=-1.0,
                       help="最小交通密度（恢复训练时指定则覆盖课程学习，默认: -1表示未指定）")
    parser.add_argument("--traffic_density_max", type=float, default=-1.0,
                       help="最大交通密度（恢复训练时指定则覆盖课程学习，默认: -1表示未指定）")

    # ===== 训练设置 =====
    parser.add_argument("--total_timesteps", type=int, default=1000000,
                       help="总训练步数 (默认: 1,000,000)")
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                       help="检查点保存频率 (默认: 50)")
    parser.add_argument("--eval_freq", type=int, default=50,
                       help="评估频率 (默认: 10)")
    parser.add_argument("--log_freq", type=int, default=1,
                       help="日志记录频率 (默认: 1)")
    
    # ===== 课程学习（Curriculum Learning）开关与参数 =====
    parser.add_argument("--use_curriculum", action="store_true", default=False,
                        help="启用课程学习（默认关闭）")
    parser.add_argument("--curriculum_mode", type=str, default="progress",
                        choices=["progress", "gate"],
                        help="课程推进方式：progress=按训练进度；gate=按评估指标门槛（默认：progress）")
    parser.add_argument("--curriculum_alpha", type=float, default=1.5,
                        help="progress模式下的分段内插值幂指数（默认1.5，越大越保守）")
    parser.add_argument("--gate_succ_threshold", type=float, default=0.70,
                        help="gate模式：晋级所需成功率阈值（默认0.70）")
    parser.add_argument("--gate_coll_threshold", type=float, default=0.20,
                        help="gate模式：晋级所需碰撞率上限（默认0.20）")

    # ===== 认知模块参数 =====
    parser.add_argument("--use_cognitive_modules", action="store_true", default=False,
                        help="启用认知模块（默认关闭）")
    parser.add_argument("--cognitive_visualization", action="store_true", default=False,
                        help="认知模块可视化输出（默认关闭）")
    
    # === 新增：认知参数采样器参数 ===
    parser.add_argument("--use_cognitive_parameter_sampling", action="store_true", default=False,
                        help="启用认知参数采样器（默认关闭）")
    parser.add_argument("--cognitive_sampler_type", type=str, default="discrete",
                        choices=["discrete", "continuous"],
                        help="认知参数采样器类型: discrete=离散采样, continuous=连续采样（默认: discrete）")
    parser.add_argument("--cognitive_param_update_steps", type=int, default=5,
                        help="认知参数更新频率（环境步数，默认5步）")
    parser.add_argument("--bias_inverse_tta_coef_range", type=float, nargs=2, default=[0, 0.5],
                        help="视觉厌恶系数采样范围 [min, max]（默认[0.5, 2.0]）")
    parser.add_argument("--perception_sigma0_range", type=float, nargs=2, default=[0.02, 0.20],
                        help="感知噪声标准差采样范围 [min, max]（默认[0.02, 0.20]米）")
    parser.add_argument("--perception_k_range", type=float, nargs=2, default=[0.002, 0.01],
                        help="距离相关系数采样范围 [min, max]（默认[0.002, 0.01]）")
    parser.add_argument("--delay_steps_range", type=int, nargs=2, default=[0, 3],
                        help="动作延迟步数采样范围 [min, max]（默认[0, 3]，0表示无延迟）")
    
    # === 新增：离散采样器密度参数 ===
    parser.add_argument("--perception_k_density", type=int, default=5,
                        help="感知距离相关系数k的离散采样点数量（默认5）")
    parser.add_argument("--perception_sigma0_density", type=int, default=5,
                        help="感知噪声标准差sigma0的离散采样点数量（默认5）")
    parser.add_argument("--bias_inverse_tta_coef_density", type=int, default=5,
                        help="认知偏差系数的离散采样点数量（默认5）")
    parser.add_argument("--delay_steps_density", type=int, default=3,
                        help="动作延迟步数的离散采样点数量（默认3）")
    
    # 认知偏差模块参数（风险厌恶）
    parser.add_argument("--use_cognitive_bias", action="store_true", default=False,
                        help="启用认知偏差模块（默认开启）")
    parser.add_argument("--bias_inverse_tta_coef", type=float, default=1.0,
                        help="偏差强度系数（默认1.0）")
    parser.add_argument("--bias_tta_threshold", type=float, default=1.0,
                        help="TTA阈值（默认1.0）")
    parser.add_argument("--bias_adaptive", action="store_true", default=False,
                        help="启用自适应偏差（默认关闭）")
    parser.add_argument("--bias_adaptation_rate", type=float, default=0.01,
                        help="自适应速率（默认0.01）")
    parser.add_argument("--bias_visual_distance", type=float, default=50.0,
                        help="视觉检测距离（米）（默认50.0）")
    parser.add_argument("--bias_visual_angle", type=float, default=30.0,
                        help="视觉检测角度（度）（默认30.0）")
    parser.add_argument("--bias_visual_strength", type=float, default=0.5,
                        help="视觉厌恶强度（默认0.5）")
    
    # 认知延迟模块参数（动作延迟）
    parser.add_argument("--use_cognitive_delay", action="store_true", default=False,
                        help="启用认知延迟模块（默认关闭）")
    parser.add_argument("--delay_steps", type=int, default=0,
                        help="延迟步数（默认0，0表示无延迟）")
    parser.add_argument("--delay_smoothing", action="store_true", default=False,
                        help="启用动作平滑（默认开启）")
    parser.add_argument("--delay_smoothing_factor", type=float, default=0.3,
                        help="平滑系数（默认0.3）")
    
    # 认知感知模块参数（观测噪声）
    parser.add_argument("--use_cognitive_perception", action="store_true", default=False,
                        help="启用认知感知模块（默认关闭）")
    parser.add_argument("--perception_sigma0", type=float, default=0.1,
                        help="基础噪声标准差（米）（默认0.1）")
    parser.add_argument("--perception_k", type=float, default=0.02,
                        help="距离相关系数（默认0.02）")
    parser.add_argument("--perception_p_miss0", type=float, default=0.0,
                        help="基础漏检概率（默认0.0，已关闭）")
    parser.add_argument("--perception_p_false", type=float, default=0.0,
                        help="误检概率（默认0.0，已关闭）")
    parser.add_argument("--perception_use_kf", action="store_true", default=True,
                        help="启用卡尔曼滤波（默认开启）")
    parser.add_argument("--perception_kf_dt", type=float, default=0.1,
                        help="卡尔曼滤波步长（默认0.1）")
    parser.add_argument("--perception_kf_q_scale", type=float, default=100.0,
                        help="卡尔曼滤波过程噪声缩放（默认100.0）")
    
    # ===== 认知可视化参数 =====
    parser.add_argument("--enable_radar_beam_viz", action="store_true", default=False,
                        help="启用雷达束可视化（默认关闭）")
    
    # ===== 系统设置 =====
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="计算设备 (默认: auto)")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="保存目录（恢复训练时若未指定则使用原检查点目录）")
    
    # ===== 恢复训练设置 =====
    parser.add_argument("--resume_from", type=str, default=None,
                       help="从指定检查点恢复训练 (默认: None, 从头开始)")
    
    # ===== 速度控制奖励参数 (新增) =====
    parser.add_argument("--use_speed_control_reward", action="store_true", default=False,
                       help="启用速度控制奖励（默认关闭）")
    
    # === 子模块启用控制参数 ===
    parser.add_argument("--speed_control_enable_tracking", action="store_true", default=False,
                       help="启用速度跟踪子模块（默认关闭）")
    parser.add_argument("--speed_control_enable_soft_wall", action="store_true", default=False,
                       help="启用速度软墙子模块（默认关闭）")
    parser.add_argument("--speed_control_enable_behavior_guidance", action="store_true", default=True,
                       help="启用行为导向子模块（默认关闭）")
    
    # === 子模块参数 ===
    parser.add_argument("--speed_control_k", type=float, default=1.0,
                       help="速度跟踪奖励系数 (默认: 1.0)")
    parser.add_argument("--speed_control_kappa", type=float, default=0.5,
                       help="超速软墙惩罚系数 (默认: 0.5)")
    parser.add_argument("--speed_control_mu", type=float, default=0.3,
                       help="超速刹车奖励系数 (默认: 0.3)")
    parser.add_argument("--speed_control_nu", type=float, default=0.2,
                       help="超速加速惩罚系数 (默认: 0.2)")
    parser.add_argument("--speed_control_v_tolerance", type=float, default=1.0,
                       help="速度跟踪容差 (默认: 1.0 m/s)")
    parser.add_argument("--speed_control_v_ref", type=float, default=10.0,
                       help="目标参考速度 (默认: 25.0 m/s)")
    
    # ===== 动作正则化参数 (新增) =====
    parser.add_argument("--action_penalty_coef", type=float, default=0.001,
                       help="动作正则化惩罚系数 (默认: 0.001)")
    
    return parser


def main():
    """主函数"""
    # 解析参数
    parser = add_arguments()
    args = parser.parse_args()
    
    # 自动选择设备
    if args.device == "auto":
        args.device = "cuda"# if torch.cuda.is_available() else "cpu"
    
    print(" MetaDrive PPO Expert 复现训练")
    print("=" * 50)
    
    # 恢复训练信息
    if args.resume_from:
        print(f" 恢复训练模式:")
        print(f"   检查点路径: {args.resume_from}")
        print(f"   检查点存在: {'✅' if os.path.exists(args.resume_from) else '❌'}")
        print("=" * 50)
    
    print(f" 关键超参数:")
    print(f"   学习率: {args.lr}")
    print(f"   rollout步数: {args.n_steps}")
    print(f"   环境数量: {args.n_envs}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   训练轮次: {args.n_epochs}")
    print(f"   裁剪范围: {args.clip_range}")
    
    # === 新增：认知模块配置显示 ===
    if args.use_cognitive_modules:
        print(f" 认知模块配置:")
        print(f"   认知偏差模块: {'启用' if args.use_cognitive_bias else '禁用'}")
        print(f"   认知延迟模块: {'启用' if args.use_cognitive_delay else '禁用'}")
        print(f"   认知感知模块: {'启用' if args.use_cognitive_perception else '禁用'}")
        if args.use_cognitive_parameter_sampling:
            sampler_type_name = "离散" if args.cognitive_sampler_type == "discrete" else "连续"
            print(f"   参数采样器: 启用 ({sampler_type_name})")
            if args.cognitive_sampler_type == "discrete":
                print(f"     采样密度设置: 偏差系数={args.bias_inverse_tta_coef_density}, "
                      f"感知sigma0={args.perception_sigma0_density}, "
                      f"感知k={args.perception_k_density}, "
                      f"延迟步数={args.delay_steps_density}")
        else:
            print(f"   参数采样器: 禁用")
    else:
        print(f" 认知模块: 未启用")
    print(f" 训练设置:")
    print(f"   总步数: {args.total_timesteps:,}")
    print(f"   随机种子: {args.seed}")
    print(f"   计算设备: {args.device}")
    print("=" * 50)
    
    # 创建训练器并开始训练
    trainer = PPOExpertReproduction(args)
    trainer.train()


if __name__ == "__main__":
    main() 

