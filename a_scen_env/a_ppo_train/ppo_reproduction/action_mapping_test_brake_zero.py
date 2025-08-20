#!/usr/bin/env python3
"""
动作映射自检 - 实验A：强制brake为0
测试将brake通道强制设为0的效果
"""

import os
import sys
import json
import argparse
import numpy as np
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
from torch.utils.tensorboard import SummaryWriter


class PPONetwork(nn.Module):
    """PPO网络结构 - 严格对齐MetaDrive expert"""
    
    def __init__(self, obs_dim: int = 275, action_dim: int = 2, hidden_dim: int = 256):
        super(PPONetwork, self).__init__()
        
        # Actor网络 (与expert完全对齐)
        self.actor_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim * 2)  # mean + log_std
        
        # Critic网络 (与expert完全对齐)
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
    
    def get_action_and_value(self, obs, action=None):
        """获取动作和价值"""
        action_logits, value = self.forward(obs)
        
        # 分离均值和标准差
        action_mean, action_log_std = torch.chunk(action_logits, 2, dim=-1)
        action_std = torch.exp(action_log_std)
        
        # 创建分布
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)


def make_env(rank: int, config: Dict[str, Any]):
    """环境工厂函数 - 用于创建向量化环境"""
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
        
        # 动态交通密度：每个环境不同（使用基础配置中的值）
        min_density, max_density = config.get("traffic_density_min", 0.1), config.get("traffic_density_max", 0.15)
        base_traffic_density = config.get("traffic_density", min_density)  # 使用配置中的基础值
        
        # 为了保持环境多样性，每个环境在基础密度上增加小的扰动
        density_perturbation = (scenario_index % 100) / 1000.0 * 0.02  # 最大2%的扰动
        traffic_density = base_traffic_density + density_perturbation
        
        # 更新环境配置
        env_config["map"] = map_string
        env_config["traffic_density"] = traffic_density
        
        # 确保子进程环境不使用渲染（避免显示冲突）
        env_config["use_render"] = False
        env_config["debug"] = False
        env_config["image_observation"] = False
        
        # 移除自定义参数（MetaDrive不识别的参数）
        custom_params = ["traffic_density_min", "traffic_density_max"]
        for param in custom_params:
            if param in env_config:
                del env_config[param]
        
        # 调试：打印关键配置参数
        print(f"🔧 环境{rank}配置验证: use_render={env_config.get('use_render')}, image_observation={env_config.get('image_observation')}")
        
        # 创建MetaDrive环境
        env = MetaDriveEnv(env_config)
        
        return env
    
    return _init


class ActionMappingTestBrakeZero:
    """动作映射测试 - 实验A：强制brake为0"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        
        # 设置随机种子
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # 创建实验目录
        self.exp_dir = self._create_experiment_dir()
        
        # 初始化训练统计
        self.global_step = 0
        self.episode_count = 0
        self.train_stats = []
        
        # 创建环境
        self.envs = self._create_environments()
        
        # 创建网络
        self.network = PPONetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.lr)
        
        # 创建TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "tensorboard"))
        
        # 添加episode统计缓冲区
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_speeds = deque(maxlen=100)
        self.episode_path_completions = deque(maxlen=100)
        
        # 创建CSV日志
        self.csv_path = os.path.join(self.exp_dir, "action_mapping_test_brake_zero.csv")
        self._init_csv_log()
        
        print(f"🚀 动作映射测试 - 实验A：强制brake为0 初始化完成")
        print(f"📁 实验目录: {self.exp_dir}")
        print(f"🔧 设备: {self.device}")
        print(f"🌱 随机种子: {args.seed}")
        print(f"🎯 动作修改: 强制brake通道(第1通道)为0")
    
    def _create_experiment_dir(self) -> str:
        """创建实验目录"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"action_mapping_test_brake_zero_{timestamp}"
        exp_dir = os.path.join(self.args.save_dir, "runs", exp_name)
        
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        
        print(f"📁 实验目录已创建: {exp_dir}")
        return exp_dir
    
    def _get_base_env_config(self):
        """获取基础环境配置"""
        return {
            "num_scenarios": 1000,
            "map": "SSSSS",  # 固定5段直线
            "traffic_density": 0.1,
            "random_traffic": True,
            "horizon": 1000,
            "start_seed": self.args.seed,
            "use_render": False,
            "debug": False,
            "image_observation": False,
            "success_reward": 20.0,
            "driving_reward": 2.0,
            "speed_reward": 0.3,
            "use_lateral_reward": True,
            "out_of_road_penalty": 8.0,
            "crash_vehicle_penalty": 8.0,
            "crash_object_penalty": 8.0,
            "crash_sidewalk_penalty": 2.0,
            "out_of_road_done": True,
            "crash_vehicle_done": True,
            "crash_object_done": True,
            "on_continuous_line_done": False,
            "on_broken_line_done": False,
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
    
    def _create_environments(self):
        """创建向量化环境"""
        base_env_config = self._get_base_env_config()
        
        if self.args.n_envs > 1:
            print(f"🚀 创建 {self.args.n_envs} 个并行环境 (SubprocVecEnv)")
            envs = SubprocVecEnv([
                make_env(rank, base_env_config) 
                for rank in range(self.args.n_envs)
            ])
            print(f"✅ 成功创建 {self.args.n_envs} 个并行环境")
            return envs
        else:
            print("📍 创建单个环境 (DummyVecEnv)")
            envs = DummyVecEnv([make_env(0, base_env_config)])
            return envs
    
    def _init_csv_log(self):
        """初始化CSV日志文件"""
        headers = [
            "step", "episode", "ep_reward_mean", "ep_len_mean",
            "policy_loss", "value_loss", "entropy", "approx_kl",
            "learning_rate", "collision_rate", "offroad_rate", 
            "success_rate", "avg_speed", "path_completion"
        ]
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"📄 CSV日志文件已创建: {self.csv_path}")
    
    def collect_rollouts(self) -> Tuple[torch.Tensor, ...]:
        """收集rollout数据 - 实验A：强制brake为0"""
        # 存储rollout数据
        obs_batch = []
        actions_batch = []
        log_probs_batch = []
        rewards_batch = []
        dones_batch = []
        values_batch = []
        
        # 使用向量化环境进行真实并行采样
        obs = self.envs.reset()  # 返回shape: (n_envs, obs_dim)
        
        # Episode统计变量 - 支持多环境
        episode_rewards = np.zeros(self.args.n_envs)
        episode_lengths = np.zeros(self.args.n_envs)
        episode_speeds = [[] for _ in range(self.args.n_envs)]
        
        # 收集n_steps步数据
        for step in range(self.args.n_steps):
            # 将观测转换为tensor
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            with torch.no_grad():
                actions, log_probs, _, values = self.network.get_action_and_value(obs_tensor)
            
            # 执行动作 - 实验A：强制brake通道为0
            actions_np = actions.cpu().numpy()
            
            # 🎯 关键修改：强制brake通道(第1通道)为0
            # 假设动作维度为 [steer, throttle/brake]
            # 将第1通道(索引1)强制设为0
            actions_np[:, 1] = 0.0
            
            print(f"🔧 Step {step}: 原始动作={actions.cpu().numpy()[0]}, 修改后={actions_np[0]}")
            
            next_obs, rewards, dones, infos = self.envs.step(actions_np)
            
            # 收集episode统计信息 - 处理多环境信息
            episode_rewards += rewards
            episode_lengths += 1
            
            # 处理多环境的info信息
            for env_idx, info in enumerate(infos):
                # 速度统计
                if 'velocity' in info:
                    episode_speeds[env_idx].append(info['velocity'])
                elif 'speed' in info:
                    episode_speeds[env_idx].append(info['speed'])
                elif hasattr(info, 'speed'):
                    episode_speeds[env_idx].append(info.speed)
            
            # 处理episode结束 - 检查每个环境
            for env_idx in range(self.args.n_envs):
                if dones[env_idx]:
                    # 记录episode统计
                    self.episode_rewards.append(episode_rewards[env_idx])
                    self.episode_lengths.append(episode_lengths[env_idx])
                    self.episode_speeds.append(np.mean(episode_speeds[env_idx]) if episode_speeds[env_idx] else 0)
                    
                    # 路径完成度计算
                    info = infos[env_idx]
                    path_completion = info.get('route_completion', 0.0)
                    if 'arrive_dest' in info and info['arrive_dest']:
                        path_completion = 1.0
                    self.episode_path_completions.append(path_completion)
                    
                    # 重置该环境的统计
                    episode_rewards[env_idx] = 0
                    episode_lengths[env_idx] = 0
                    episode_speeds[env_idx] = []
            
            # 存储数据
            obs_batch.append(obs.copy())
            actions_batch.append(actions_np)
            log_probs_batch.append(log_probs.cpu().numpy())
            rewards_batch.append(rewards)
            dones_batch.append(dones.astype(np.float32))
            values_batch.append(values.cpu().numpy())
            
            obs = next_obs  # 更新观测
        
        # 更新全局步数
        self.global_step += self.args.n_steps * self.args.n_envs
        
        # 转换为tensor
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        actions_batch = torch.FloatTensor(actions_batch).to(self.device)
        log_probs_batch = torch.FloatTensor(log_probs_batch).to(self.device)
        rewards_batch = torch.FloatTensor(rewards_batch).to(self.device)
        dones_batch = torch.FloatTensor(dones_batch).to(self.device)
        values_batch = torch.FloatTensor(values_batch).to(self.device)
        
        # 计算advantages和returns
        advantages, returns = self.compute_gae(rewards_batch, values_batch, dones_batch)
        
        return (obs_batch, actions_batch, log_probs_batch, 
                advantages, returns)
    
    def compute_gae(self, rewards, values, dones):
        """计算GAE优势函数"""
        advantages = torch.zeros_like(rewards).to(self.device)
        gae = 0
        
        for t in reversed(range(self.args.n_steps)):
            if t == self.args.n_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.args.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.args.gamma * self.args.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update_policy(self, obs, actions, old_log_probs, advantages, returns):
        """更新策略"""
        # 标准化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 准备训练数据
        batch_size = obs.shape[0] * obs.shape[1]  # n_steps * n_envs
        obs = obs.view(batch_size, -1)
        actions = actions.view(batch_size, -1)
        old_log_probs = old_log_probs.view(batch_size)
        advantages = advantages.view(batch_size)
        returns = returns.view(batch_size)
        
        # 多轮更新
        policy_losses = []
        value_losses = []
        entropy_losses = []
        entropies = []
        approx_kls = []
        
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
                
                # PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value损失
                value_loss = nn.MSELoss()(new_values, batch_returns)
                
                # Entropy损失
                entropy_loss = -entropy.mean()
                
                # 总损失
                total_loss = policy_loss + self.args.vf_coef * value_loss + self.args.entropy_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)
                
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
            "approx_kl": np.mean(approx_kls)
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """评估策略 - 实验A：强制brake为0"""
        eval_rewards = []
        eval_lengths = []
        eval_collisions = 0
        eval_offroads = 0
        eval_successes = 0
        eval_speeds = []
        eval_path_completions = []
        
        print(f"🔍 开始策略评估 ({num_episodes} episodes)...")
        print(f"🎯 评估时动作修改: 强制brake通道为0")
        
        for episode in range(num_episodes):
            obs = self.envs.reset()
            episode_reward = 0
            episode_length = 0
            episode_speeds = []
            
            while True:
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                
                with torch.no_grad():
                    action, _, _, _ = self.network.get_action_and_value(obs_tensor)
                
                # 评估时也强制brake为0
                action_np = action.cpu().numpy()
                action_np[1] = 0.0  # 强制brake通道为0
                
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
                
                # 速度统计
                if 'velocity' in info:
                    episode_speeds.append(info['velocity'])
                elif 'speed' in info:
                    episode_speeds.append(info['speed'])
                elif hasattr(info, 'speed'):
                    episode_speeds.append(info.speed)
                
                if done_flag:
                    # 统计终止原因
                    if info.get("crash", False) or info.get("crash_vehicle", False) or info.get("crash_object", False):
                        eval_collisions += 1
                    elif info.get("out_of_road", False):
                        eval_offroads += 1 
                    elif info.get("arrive_dest", False):
                        eval_successes += 1
                    
                    # 计算路径完成度
                    path_completion = info.get('route_completion', 0.0)
                    if info.get('arrive_dest', False):
                        path_completion = 1.0
                    eval_path_completions.append(path_completion)
                    
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_speeds.append(np.mean(episode_speeds) if episode_speeds else 0)
        
        print(f"✅ 评估完成")
        
        return {
            "eval_reward_mean": np.mean(eval_rewards),
            "eval_reward_std": np.std(eval_rewards),
            "eval_length_mean": np.mean(eval_lengths),
            "eval_collision_rate": eval_collisions / num_episodes,
            "eval_offroad_rate": eval_offroads / num_episodes,
            "eval_success_rate": eval_successes / num_episodes,
            "eval_avg_speed": np.mean(eval_speeds),
            "eval_path_completion": np.mean(eval_path_completions)
        }
    
    def log_metrics(self, iteration: int, train_stats: Dict, eval_stats: Dict = None):
        """记录指标"""
        # TensorBoard日志
        for key, value in train_stats.items():
            self.writer.add_scalar(f"train/{key}", value, self.global_step)
        
        # 添加学习率记录
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)
        
        # Episode环境统计
        if len(self.episode_rewards) > 0:
            self.writer.add_scalar("env/ep_rew_mean", np.mean(self.episode_rewards), self.global_step)
            
        if len(self.episode_speeds) > 0:
            self.writer.add_scalar("env/ep_speed_mean", np.mean(self.episode_speeds), self.global_step)
            
        if len(self.episode_path_completions) > 0:
            self.writer.add_scalar("env/ep_path_completion_mean", np.mean(self.episode_path_completions), self.global_step)
        
        # 评估指标
        if eval_stats:
            for key, value in eval_stats.items():
                self.writer.add_scalar(f"eval/{key.replace('eval_', '')}", value, self.global_step)
        
        # CSV日志
        log_data = [
            self.global_step, iteration,
            eval_stats.get("eval_reward_mean", 0) if eval_stats else 0,
            eval_stats.get("eval_length_mean", 0) if eval_stats else 0,
            train_stats.get("policy_loss", 0),
            train_stats.get("value_loss", 0),
            train_stats.get("entropy", 0),
            train_stats.get("approx_kl", 0),
            current_lr,
            eval_stats.get("eval_collision_rate", 0) if eval_stats else 0,
            eval_stats.get("eval_offroad_rate", 0) if eval_stats else 0,
            eval_stats.get("eval_success_rate", 0) if eval_stats else 0,
            eval_stats.get('eval_avg_speed', 0) if eval_stats else 0,
            eval_stats.get('eval_path_completion', 0) if eval_stats else 0
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
            print(f"📊 详细统计 (Step {self.global_step}):")
            print(f"   平均速度: {eval_stats.get('eval_avg_speed', 0):.2f}")
            print(f"   路径完成: {eval_stats.get('eval_path_completion', 0):.3f}")
            print(f"   成功率: {eval_stats.get('eval_success_rate', 0):.3f}")
    
    def train(self):
        """主训练循环 - 实验A：强制brake为0"""
        print(f"🚀 开始动作映射测试 - 实验A：强制brake为0")
        print(f"🎯 目标：测试将brake通道强制设为0的效果")
        print(f"📊 动作处理：强制brake通道(第1通道)为0")
        print(f"🔍 预期效果：如果brake通道配置错误，此修改应能改善性能")
        
        iteration = 0
        start_time = time.time()
        best_reward = float('-inf')
        
        while self.global_step < self.args.total_timesteps:
            iteration += 1
            
            # 收集rollouts
            rollout_start = time.time()
            rollouts = self.collect_rollouts()
            rollout_time = time.time() - rollout_start
            
            # 更新策略
            update_start = time.time()
            train_stats = self.update_policy(*rollouts)
            update_time = time.time() - update_start
            
            # 计算FPS
            fps = (self.args.n_steps * self.args.n_envs) / (rollout_time + update_time)
            train_stats["fps"] = fps
            
            # 定期评估
            eval_stats = None
            if iteration % self.args.eval_freq == 0:
                eval_stats = self.evaluate()
                
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
        print(f"✅ 实验A训练完成！总用时: {time.time() - start_time:.2f}秒")
        
        # 最终评估和保存
        final_eval = self.evaluate(num_episodes=20)
        self.save_checkpoint(iteration, False)
        
        # 生成最终报告
        self.generate_final_report(final_eval)
        
        # 关闭资源
        self.writer.close()
        self.envs.close()
        print(f"🔄 训练环境已安全关闭")
    
    def save_checkpoint(self, iteration: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            "iteration": iteration,
            "global_step": self.global_step,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "args": vars(self.args)
        }
        
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
    
    def generate_final_report(self, final_eval: Dict):
        """生成最终报告"""
        report_path = os.path.join(self.exp_dir, "action_mapping_test_brake_zero_report.md")
        
        report_content = f"""# 动作映射测试 - 实验A：强制brake为0 报告

## 📋 测试目的
测试将brake通道强制设为0的效果，验证动作通道配置的正确性。

## 🔧 测试配置
- **动作处理**: 强制brake通道(第1通道)为0
- **动作维度**: 2 (steer + throttle/brake)
- **网络结构**: 与原始版本完全一致
- **修改逻辑**: `actions_np[:, 1] = 0.0`

## 📊 测试结果

### 最终性能
- **平均奖励**: {final_eval['eval_reward_mean']:.3f} ± {final_eval['eval_reward_std']:.3f}
- **平均长度**: {final_eval['eval_length_mean']:.1f}
- **碰撞率**: {final_eval['eval_collision_rate']:.3f}
- **冲出道路率**: {final_eval['eval_offroad_rate']:.3f}
- **成功率**: {final_eval['eval_success_rate']:.3f}
- **平均速度**: {final_eval['eval_avg_speed']:.2f}
- **路径完成度**: {final_eval['eval_path_completion']:.3f}

## 🎯 判定标准
- **平均速度**: 应 > 5.0 m/s
- **路径完成度**: 应 > 0.3
- **成功率**: 应 > 0.1
- **评估回报**: 应 > 0

## 🔍 预期效果分析
如果brake通道配置错误（如把油门当刹车），此修改应能：
1. **提高平均速度** - 车辆不再被错误"刹车"
2. **改善路径完成度** - 车辆能正常前进
3. **提升成功率** - 减少因速度过低导致的超时

## 📁 产物说明
- **CSV日志**: {os.path.basename(self.csv_path)}
- **TensorBoard**: {os.path.join(self.exp_dir, "tensorboard")}
- **检查点**: {os.path.join(self.exp_dir, "checkpoints")}

---
**测试完成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**实验目录**: `{self.exp_dir}`
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 最终报告已生成: {report_path}")
    
    def _validate_and_log_scenarios(self):
        """验证和记录场景配置"""
        print(f"\n🛣️ 场景配置验证:")
        print(f"   场景类型: 固定5段直线道路")
        print(f"   交通密度: 0.1")
        print(f"   交通随机化: 启用")
        print(f"   随机种子: {self.args.seed}")
        print("=" * 50)


def add_arguments():
    """添加命令行参数"""
    parser = argparse.ArgumentParser(description="动作映射测试 - 实验A：强制brake为0")
    
    # 关键超参数
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="学习率 (默认: 3e-4)")
    parser.add_argument("--n_steps", type=int, default=1024,
                       help="rollout步数 (默认: 1024)")
    parser.add_argument("--n_envs", type=int, default=2,
                       help="并行环境数量 (默认: 2)")
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
    parser.add_argument("--vf_coef", type=float, default=0.5,
                       help="值函数损失系数 (默认: 0.5)")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                       help="梯度裁剪阈值 (默认: 0.5)")
    parser.add_argument("--target_kl", type=float, default=None,
                       help="目标KL散度 (早停, 默认: None)")
    
    # 训练设置
    parser.add_argument("--total_timesteps", type=int, default=50000,
                       help="总训练步数 (默认: 50,000)")
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                       help="检查点保存频率 (默认: 10)")
    parser.add_argument("--eval_freq", type=int, default=5,
                       help="评估频率 (默认: 5)")
    parser.add_argument("--log_freq", type=int, default=1,
                       help="日志记录频率 (默认: 1)")
    
    # 系统设置
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="计算设备 (默认: auto)")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    parser.add_argument("--save_dir", type=str, 
                       default="/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction",
                       help="保存目录")
    
    return parser


def main():
    """主函数"""
    # 解析参数
    parser = add_arguments()
    args = parser.parse_args()
    
    # 自动选择设备
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🎯 动作映射测试 - 实验A：强制brake为0")
    print("=" * 50)
    print(f"📊 关键超参数:")
    print(f"   学习率: {args.lr}")
    print(f"   rollout步数: {args.n_steps}")
    print(f"   环境数量: {args.n_envs}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   训练轮次: {args.n_epochs}")
    print(f"   裁剪范围: {args.clip_range}")
    print(f"🔧 训练设置:")
    print(f"   总步数: {args.total_timesteps:,}")
    print(f"   随机种子: {args.seed}")
    print(f"   计算设备: {args.device}")
    print("=" * 50)
    
    # 创建训练器并开始训练
    trainer = ActionMappingTestBrakeZero(args)
    trainer.train()


if __name__ == "__main__":
    main() 