#!/usr/bin/env python3
"""
PPO检查点仿真控制器
基于训练好的PPO检查点在MetaDrive仿真环境中控制主车行为
支持加载指定检查点、可视化仿真、性能评估等功能
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
import matplotlib.pyplot as plt

# 添加metadrive到路径
current_dir = Path(__file__).parent.absolute()
metadrive_root = current_dir.parent.parent.parent
sys.path.insert(0, str(metadrive_root))

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.state_obs import LidarStateObservation


class PPONetwork(nn.Module):
    """PPO网络结构 - 与训练脚本完全一致"""
    
    def __init__(self, obs_dim: int = 275, action_dim: int = 2, hidden_dim: int = 256):
        super(PPONetwork, self).__init__()
        
        # Actor网络
        self.actor_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim * 2)  # mean + log_std
        
        # Critic网络
        self.critic_fc1 = nn.Linear(obs_dim, hidden_dim)
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
            args: 命令行参数（用于变道惩罚配置）
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.args = args  # 🔧 新增：保存命令行参数引用
        
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
        
        # 创建并加载网络
        self.network = PPONetwork().to(self.device)
        self.network.load_state_dict(self.checkpoint['network_state_dict'])
        self.network.eval()
        
        print(f"📥 检查点加载成功: {os.path.basename(self.checkpoint_path)}")
    
    def _get_default_config(self):
        """获取默认环境配置 - 与训练时保持一致"""
        return {
            # 基础环境配置 - 与训练时一致
            "num_scenarios": 1000,
            "traffic_density": 0.15,
            "random_traffic": False,
            "random_agent_model": False,
            "horizon": 1000,
            "map": "SSSSSSS",
            "start_seed": 8888,
                 # 使用动态数量的直线段


            
            # 奖励配置 - 与训练时完全一致
            "success_reward": 10.0,
            "driving_reward": 1.0,
            "speed_reward": 0.1,
            "use_lateral_reward": True,
            
            # 惩罚配置 - 与训练时一致
            "out_of_road_penalty": 5.0,
            "crash_vehicle_penalty": 5.0,
            "crash_object_penalty": 5.0,
            "crash_sidewalk_penalty": 2.0,
            
            # 终止条件 - 与训练时一致
            "out_of_road_done": True,
            "crash_vehicle_done": True,
            "crash_object_done": True,
            "on_continuous_line_done": False,
            "on_broken_line_done": False,
            
            # 车辆配置 - 与训练时完全一致
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
            
            # 渲染配置
            "use_render": True
        }
    
    def create_environment(self, render: bool = True, scenario_seed: Optional[int] = None) -> MetaDriveEnv:
        """创建MetaDrive仿真环境"""
        env_config = self.config.copy()
        env_config["use_render"] = render
        
        if scenario_seed is not None:
            env_config["start_seed"] = scenario_seed
        
        # 设置观察配置
        env_config["vehicle_config"]["lidar"] = {
            "num_lasers": 240,
            "distance": 50,
            "num_others": 4,
        }
        
        # 创建环境
        env = MetaDriveEnv(env_config)
        
        # 🔧 新增：动态设置变道冷却时间
        if not hasattr(self, '_lane_change_cooldown_steps') or self._lane_change_cooldown_steps is None:
            self._setup_lane_change_cooldown(env)
        
        return env
    
    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        根据观察获取PPO策略动作
        
        Args:
            observation: 环境观察
            deterministic: 是否使用确定性策略
            
        Returns:
            动作数组 [steering, acceleration]
        """
        # 转换为tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # 获取动作
        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(obs_tensor, deterministic=deterministic)
        
        # 转换回numpy
        action = action.cpu().numpy().flatten()
        
        # 限制动作范围
        action = np.clip(action, -1.0, 1.0)
        
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
            "rewards": []
        }
        
        # 🔧 新增：变道惩罚统计变量
        episode_stats["lane_change_penalties"] = []
        episode_stats["lane_change_speed_ratios"] = []
        episode_stats["cooldown_violations"] = []
        episode_stats["lane_changes"] = 0
        
        speeds = []
        step_count = 0
        
        try:
            for step in range(max_steps):
                # 获取PPO动作
                action = self.get_action(obs, deterministic=deterministic)
                episode_stats["actions"].append(action.copy())
                
                # 调试：打印前几步的动作
                if step < 5:
                    print(f"  步骤 {step}: 动作=[{action[0]:.3f}, {action[1]:.3f}]")
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                
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
            
            # 使用不同的场景种子
            scenario_seed = 8888 + episode
            
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
        
        return evaluation


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PPO检查点仿真控制器")
    
    parser.add_argument("--checkpoint", type=str,
                       default="/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/runs/milestone_checkpoint_iter1500_stage2_reward819.2_succ0.90.pt",
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
    
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="计算设备")
    
    args = parser.parse_args()
    
    # 检查检查点文件
    if not os.path.exists(args.checkpoint):
        print(f"❌ 检查点文件不存在: {args.checkpoint}")
        sys.exit(1)
    
    try:
        # 创建仿真控制器
        simulator = PPOCheckpointSimulator(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=args.device,
            args=args  # 🔧 新增：传递命令行参数
        )
        
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
        
    except Exception as e:
        print(f"❌ 仿真运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 