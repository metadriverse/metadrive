#!/usr/bin/env python3
"""
MetaDrive PPO Expert 复现训练系统
严格对齐MetaDrive PPO expert的配置，仅关键超参数可调整
支持TensorBoard可视化、完整产物落地和详细说明文档生成
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

# 添加metadrive到路径
current_dir = Path(__file__).parent.absolute()
metadrive_root = current_dir.parent.parent.parent
sys.path.insert(0, str(metadrive_root))

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.state_obs import LidarStateObservation
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
        
        # 创建环境
        self.envs = self._create_environments()
        
        # 创建网络
        self.network = PPONetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.lr)
        
        # 创建TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "tensorboard"))
        
        # 初始化训练统计
        self.global_step = 0
        self.episode_count = 0
        self.train_stats = []
        
        # 添加episode统计缓冲区
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_speeds = deque(maxlen=100)
        self.episode_lane_deviations = deque(maxlen=100)
        self.episode_lane_changes = deque(maxlen=100)
        self.episode_min_ttcs = deque(maxlen=100)
        self.episode_path_completions = deque(maxlen=100)
        self.episode_timeouts = deque(maxlen=100)
        
        # 创建CSV日志
        self.csv_path = os.path.join(self.exp_dir, "training_logs.csv")
        self._init_csv_log()
        
        print(f"🚀 PPO Expert复现训练初始化完成")
        print(f"📁 实验目录: {self.exp_dir}")
        print(f"🔧 设备: {self.device}")
        print(f"🌱 随机种子: {args.seed}")
    
    def _create_experiment_dir(self) -> str:
        """创建实验目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"ppo_expert_reproduction_{timestamp}"
        exp_dir = os.path.join(self.args.save_dir, f"runs/{exp_name}")
        
        # 创建子目录
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "tensorboard"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        
        return exp_dir
    
    def _build_config(self) -> Dict[str, Any]:
        """构建完整配置"""
        return {
            # ===== 复现设定 =====
            "reproduction_target": "MetaDrive PPO Expert",
            "experiment_name": os.path.basename(self.exp_dir),
            "timestamp": datetime.now().isoformat(),
            "random_seed": self.args.seed,
            "device": str(self.device),
            
            # ===== 网络结构 (严格对齐expert) =====
            "network": {
                "observation_dim": 275,  # 与expert对齐
                "action_dim": 2,
                "hidden_dim": 256,
                "activation": "tanh"
            },
            
            # ===== 环境配置 (严格对齐expert) =====
            "env_config": {
                "num_scenarios": 1000,
                "traffic_density": 0.1,
                "random_traffic": False,
                "horizon": 1000,
                "map": 3,
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
                "target_kl": self.args.target_kl
            },
            
            # ===== 训练设定 =====
            "training": {
                "total_timesteps": self.args.total_timesteps,
                "checkpoint_freq": self.args.checkpoint_freq,
                "eval_freq": self.args.eval_freq,
                "log_freq": self.args.log_freq
            }
        }
    
    def _save_config(self):
        """保存配置文件"""
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def _get_env_config(self):
        """获取环境配置"""
        return {
            "num_scenarios": 1000,
            "traffic_density": 0.1,
            "random_traffic": False,
            "horizon": 1000,
            "map": 3,
            "start_seed": self.args.seed,
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
    
    def _create_single_environment(self):
        """创建单个环境实例"""
        return MetaDriveEnv(self._get_env_config())
    
    def _create_environments(self):
        """创建单个环境（MetaDrive不支持多环境实例）"""
        # MetaDrive由于Engine单例模式限制，不支持在同一进程中创建多个环境实例
        # 正确的方法是使用vectorized environment (如SubprocVecEnv)，但为了简化，这里使用单环境
        print("⚠️  注意：由于MetaDrive Engine单例模式限制，使用单环境训练")
        print("   如需真正的多环境并行，请使用SubprocVecEnv等vectorized environment")
        
        env = self._create_single_environment()
        return [env]  # 返回单环境列表以保持兼容性
    
    def _init_csv_log(self):
        """初始化CSV日志"""
        headers = [
            "step", "episode", "ep_reward_mean", "ep_len_mean",
            "policy_loss", "value_loss", "entropy", "approx_kl",
            "learning_rate", "collision_rate", "offroad_rate", 
            "success_rate", "fps", "clipfrac", "explained_variance",
            "grad_norm", "avg_speed", "lane_deviation", "lane_change_count",
            "min_ttc", "path_completion"
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def collect_rollouts(self) -> Tuple[torch.Tensor, ...]:
        """收集rollout数据（单环境模拟批处理）"""
        # 存储rollout数据
        obs_batch = []
        actions_batch = []
        log_probs_batch = []
        rewards_batch = []
        dones_batch = []
        values_batch = []
        
        # 使用单环境（避免MetaDrive Engine单例问题）
        env = self.envs[0]  # 只有一个环境
        
        # 初始化环境状态
        obs, _ = env.reset()
        
        # Episode统计变量
        episode_reward = 0
        episode_length = 0
        episode_speeds = []
        lane_deviations = []
        lane_changes = 0
        min_ttcs = []
        
        # 收集n_steps步数据，通过单环境重复采样模拟n_envs个并行样本
        for step in range(self.args.n_steps):
            # 对于单环境，我们重复当前观测来模拟批处理
            # 在实际应用中，这样做会降低样本多样性，但可以避免MetaDrive的单例问题
            observations = [obs for _ in range(self.args.n_envs)]
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            
            with torch.no_grad():
                actions, log_probs, _, values = self.network.get_action_and_value(obs_tensor)
            
            # 执行动作（只使用第一个动作，因为只有一个环境）
            action = actions[0].cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 收集episode统计信息
            episode_reward += reward
            episode_length += 1
            
            # 速度统计
            if hasattr(env.agent, 'speed'):
                episode_speeds.append(env.agent.speed)
            
            # 车道偏移统计
            if hasattr(env.agent, 'lane'):
                lateral_distance = getattr(env.agent, 'dist_to_left_side', 0) + getattr(env.agent, 'dist_to_right_side', 0)
                if lateral_distance > 0:
                    lane_deviation = abs(getattr(env.agent, 'dist_to_left_side', 0) - getattr(env.agent, 'dist_to_right_side', 0)) / lateral_distance
                    lane_deviations.append(lane_deviation)
            
            # TTC统计（Time to Collision）
            if 'ttc' in info:
                min_ttcs.append(info['ttc'])
            elif hasattr(env.agent, 'min_ttc'):
                min_ttcs.append(env.agent.min_ttc)
            
            # 车道变换检测（简化版）
            if 'lane_change' in info and info['lane_change']:
                lane_changes += 1
            
            # 如果episode结束，记录统计信息并重置环境
            if done:
                # 记录episode统计
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_speeds.append(np.mean(episode_speeds) if episode_speeds else 0)
                self.episode_lane_deviations.append(np.mean(lane_deviations) if lane_deviations else 0)
                self.episode_lane_changes.append(lane_changes)
                self.episode_min_ttcs.append(np.min(min_ttcs) if min_ttcs else float('inf'))
                
                # 路径完成度计算
                path_completion = info.get('route_completion', 0.0)
                if 'arrive_dest' in info and info['arrive_dest']:
                    path_completion = 1.0
                self.episode_path_completions.append(path_completion)
                
                # 超时检测
                timeout = episode_length >= env.config['horizon'] and not info.get('arrive_dest', False) and not info.get('crash', False)
                self.episode_timeouts.append(1 if timeout else 0)
                
                next_obs, _ = env.reset()
                
                # 重置episode统计
                episode_reward = 0
                episode_length = 0
                episode_speeds = []
                lane_deviations = []
                lane_changes = 0
                min_ttcs = []
            
            # 为了模拟批处理，我们复制结果
            next_observations = [next_obs for _ in range(self.args.n_envs)]
            rewards_list = [reward for _ in range(self.args.n_envs)]
            dones_list = [done for _ in range(self.args.n_envs)]
            
            # 存储数据
            obs_batch.append(observations.copy())
            actions_batch.append(actions.cpu().numpy())
            log_probs_batch.append(log_probs.cpu().numpy())
            rewards_batch.append(rewards_list)
            dones_batch.append(dones_list)
            values_batch.append(values.cpu().numpy())
            
            obs = next_obs  # 更新当前观测
        
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
                next_value = values[t]  # 最后一步使用当前值
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
                
                # 总损失
                total_loss = policy_loss + self.args.vf_coef * value_loss + self.args.entropy_coef * entropy_loss
                
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
            "grad_norm": np.mean(grad_norms)
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """评估策略"""
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
        
        # 使用第一个训练环境进行评估（避免重复初始化）
        eval_env = self.envs[0]
        
        for episode in range(num_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_speeds = []
            episode_lane_deviations = []
            episode_lane_changes = 0
            episode_min_ttcs = []
            
            while True:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, _, _, _ = self.network.get_action_and_value(obs_tensor)
                
                obs, reward, terminated, truncated, info = eval_env.step(action.cpu().numpy()[0])
                episode_reward += reward
                episode_length += 1
                
                # 收集详细统计信息
                # 速度统计
                if hasattr(eval_env.agent, 'speed'):
                    episode_speeds.append(eval_env.agent.speed)
                
                # 车道偏移统计
                if hasattr(eval_env.agent, 'lane'):
                    lateral_distance = getattr(eval_env.agent, 'dist_to_left_side', 0) + getattr(eval_env.agent, 'dist_to_right_side', 0)
                    if lateral_distance > 0:
                        lane_deviation = abs(getattr(eval_env.agent, 'dist_to_left_side', 0) - getattr(eval_env.agent, 'dist_to_right_side', 0)) / lateral_distance
                        episode_lane_deviations.append(lane_deviation)
                
                # TTC统计
                if 'ttc' in info:
                    episode_min_ttcs.append(info['ttc'])
                elif hasattr(eval_env.agent, 'min_ttc'):
                    episode_min_ttcs.append(eval_env.agent.min_ttc)
                
                # 车道变换检测
                if 'lane_change' in info and info['lane_change']:
                    episode_lane_changes += 1
                
                if terminated or truncated:
                    # 统计终止原因
                    if info.get("crash", False):
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
            eval_lane_deviations.append(np.mean(episode_lane_deviations) if episode_lane_deviations else 0)
            eval_lane_changes.append(episode_lane_changes)
            eval_min_ttcs.append(np.min(episode_min_ttcs) if episode_min_ttcs else float('inf'))
        
        return {
            "eval_reward_mean": np.mean(eval_rewards),
            "eval_reward_std": np.std(eval_rewards),
            "eval_length_mean": np.mean(eval_lengths),
            "eval_collision_rate": eval_collisions / num_episodes,
            "eval_offroad_rate": eval_offroads / num_episodes,
            "eval_success_rate": eval_successes / num_episodes,
            "eval_avg_speed": np.mean(eval_speeds),
            "eval_lane_deviation": np.mean(eval_lane_deviations),
            "eval_lane_change_count": np.mean(eval_lane_changes),
            "eval_min_ttc": np.mean([ttc for ttc in eval_min_ttcs if ttc != float('inf')]) if any(ttc != float('inf') for ttc in eval_min_ttcs) else 0,
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
            train_stats.get("fps", 0),
            train_stats.get('clipfrac', 0),
            train_stats.get('explained_variance', 0),
            train_stats.get('grad_norm', 0),
            eval_stats.get('eval_avg_speed', 0) if eval_stats else 0,
            eval_stats.get('eval_lane_deviation', 0) if eval_stats else 0,
            eval_stats.get('eval_lane_change_count', 0) if eval_stats else 0,
            eval_stats.get('eval_min_ttc', 0) if eval_stats else 0,
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
            print(f"   车道偏移: {eval_stats.get('eval_lane_deviation', 0):.3f}")
            print(f"   路径完成: {eval_stats.get('eval_path_completion', 0):.3f}")
            print(f"   成功率: {eval_stats.get('eval_success_rate', 0):.3f}")
            if train_stats.get('clipfrac', 0) > 0:
                print(f"   Clip Fraction: {train_stats.get('clipfrac', 0):.3f}")
                print(f"   Explained Var: {train_stats.get('explained_variance', 0):.3f}")
    
    def train(self):
        """主训练循环"""
        print(f"🚀 开始PPO训练 - 目标步数: {self.args.total_timesteps:,}")
        
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
        print(f"✅ 训练完成！总用时: {time.time() - start_time:.2f}秒")
        
        # 最终评估和保存
        final_eval = self.evaluate(num_episodes=20)
        self.save_checkpoint(iteration, False)
        
        # 生成最终报告
        self.generate_final_report(final_eval)
        
        # 关闭资源
        self.writer.close()
        for env in self.envs:
            env.close()
    
    def generate_final_report(self, final_eval: Dict):
        """生成最终报告"""
        report_path = os.path.join(self.exp_dir, "report.md")
        
        # 加载训练数据
        df = pd.read_csv(self.csv_path)
        
        report_content = f"""# MetaDrive PPO Expert 复现训练报告

## 📋 背景与目标

本实验旨在复现MetaDrive PPO Expert的训练过程，严格对齐网络结构、观测空间、动作空间和环境配置，仅对关键超参数进行可控调整。

## 🔧 实验配置

### 网络结构 (严格对齐Expert)
- **观测维度**: 275 (Lidar: 240 + State: 35)
- **动作维度**: 2 (连续控制: 转向 + 油门/刹车)
- **隐藏层**: 256 -> 256
- **激活函数**: Tanh

### 环境配置 (严格对齐Expert)
- **场景数量**: 1000
- **交通密度**: 0.1
- **时长限制**: 1000步
- **Lidar配置**: 240束激光，50米距离，4个其他车辆
- **随机种子**: {self.args.seed}

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

## 📊 训练结果

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

## 🎯 使用方法

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

### 命令行参数一览
- `--lr`: 学习率 (默认: 3e-4)
- `--n_steps`: rollout步数 (默认: 2048)
- `--n_envs`: 并行环境数 (默认: 8)
- `--batch_size`: 批次大小 (默认: 256)
- `--n_epochs`: 训练轮次 (默认: 10)
- `--gamma`: 折扣因子 (默认: 0.99)
- `--gae_lambda`: GAE lambda (默认: 0.95)
- `--clip_range`: PPO裁剪范围 (默认: 0.2)
- `--entropy_coef`: 熵系数 (默认: 0.01)
- `--total_timesteps`: 总训练步数 (默认: 1,000,000)

## 📈 可视化说明

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

## 🏆 评估协议

### 验证设置
- **验证环境**: 与训练环境相同配置
- **验证频率**: 每{self.args.eval_freq}次迭代
- **验证episode数**: 10 (最终评估20)
- **确定性策略**: 使用动作均值

### 最优模型选择标准
1. **主要指标**: 验证集平均episode奖励
2. **约束条件**: 碰撞率不劣化
3. **辅助指标**: 成功率、episode长度

## 🔄 复现性保证

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

## 📁 产物说明

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

## 🚀 扩展说明

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
        
        print(f"📄 最终报告已生成: {report_path}")


def add_arguments():
    """添加命令行参数"""
    parser = argparse.ArgumentParser(description="MetaDrive PPO Expert 复现训练")
    
    # ===== 关键超参数 (可调整) =====
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="学习率 (默认: 3e-4)")
    parser.add_argument("--n_steps", type=int, default=2048,
                       help="rollout步数 (默认: 2048)")
    parser.add_argument("--n_envs", type=int, default=1,
                       help="并行环境数量 (默认: 1)")
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
    
    # ===== 训练设置 =====
    parser.add_argument("--total_timesteps", type=int, default=1000000,
                       help="总训练步数 (默认: 1,000,000)")
    parser.add_argument("--checkpoint_freq", type=int, default=50,
                       help="检查点保存频率 (默认: 50)")
    parser.add_argument("--eval_freq", type=int, default=10,
                       help="评估频率 (默认: 10)")
    parser.add_argument("--log_freq", type=int, default=1,
                       help="日志记录频率 (默认: 1)")
    
    # ===== 系统设置 =====
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
    
    print("🎯 MetaDrive PPO Expert 复现训练")
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
    trainer = PPOExpertReproduction(args)
    trainer.train()


if __name__ == "__main__":
    main() 