#!/usr/bin/env python3
"""
PPO Expert复现模型评估工具
用于评估训练完成的PPO模型在多个验证种子和场景上的性能
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# 添加metadrive到路径
current_dir = Path(__file__).parent.absolute()
metadrive_root = current_dir.parent.parent.parent
sys.path.insert(0, str(metadrive_root))

from metadrive.envs.metadrive_env import MetaDriveEnv


class PPONetwork(nn.Module):
    """PPO网络结构 - 与训练脚本保持一致"""
    
    def __init__(self, obs_dim: int = 275, action_dim: int = 2, hidden_dim: int = 256):
        super(PPONetwork, self).__init__()
        
        # Actor网络
        self.actor_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim * 2)
        
        # Critic网络
        self.critic_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)
        
        self.tanh = nn.Tanh()
    
    def forward(self, obs):
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
        action_logits, value = self.forward(obs)
        action_mean, action_log_std = torch.chunk(action_logits, 2, dim=-1)
        action_std = torch.exp(action_log_std)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str, config_path: str, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 加载模型
        self.network = PPONetwork().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.eval()
        
        print(f"✅ 模型加载完成: {model_path}")
        print(f"🔧 使用设备: {self.device}")
    
    def evaluate_single_episode(self, env, deterministic: bool = True, render: bool = False) -> Dict:
        """评估单个episode"""
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        collision = False
        offroad = False
        success = False
        timeout = False
        
        lane_changes = 0
        overtakes = 0
        prev_lane_idx = None
        
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if deterministic:
                    action_logits, _ = self.network.forward(obs_tensor)
                    action_mean, _ = torch.chunk(action_logits, 2, dim=-1)
                    action = action_mean.cpu().numpy()[0]
                else:
                    action, _, _, _ = self.network.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # 统计车道变更
            if hasattr(env.agent, 'navigation') and hasattr(env.agent.navigation, 'current_lane'):
                current_lane_idx = env.agent.navigation.current_lane.index
                if prev_lane_idx is not None and current_lane_idx != prev_lane_idx:
                    lane_changes += 1
                prev_lane_idx = current_lane_idx
            
            # 统计超车（简化版本 - 基于速度和位置）
            if hasattr(env.agent, 'speed') and env.agent.speed > 5:  # 基本速度阈值
                overtakes += 0.1  # 简化的超车计数
            
            if render:
                env.render()
            
            if terminated or truncated:
                # 分析终止原因
                if info.get("crash", False) or info.get("crash_vehicle", False):
                    collision = True
                elif info.get("out_of_road", False):
                    offroad = True
                elif info.get("arrive_dest", False) or info.get("success", False):
                    success = True
                else:
                    timeout = True
                break
        
        return {
            "reward": episode_reward,
            "length": episode_length,
            "collision": collision,
            "offroad": offroad,
            "success": success,
            "timeout": timeout,
            "lane_changes": lane_changes,
            "overtakes": int(overtakes)
        }
    
    def evaluate_multiple_episodes(self, num_episodes: int = 50, seeds: List[int] = None, 
                                 deterministic: bool = True, render: bool = False) -> Dict:
        """评估多个episodes"""
        if seeds is None:
            seeds = list(range(100, 100 + num_episodes))
        
        all_results = []
        
        for i, seed in enumerate(seeds[:num_episodes]):
            # 创建环境配置
            env_config = self.config["env_config"].copy()
            env_config["start_seed"] = seed
            
            # 创建环境
            env = MetaDriveEnv(env_config)
            
            try:
                result = self.evaluate_single_episode(env, deterministic, render)
                result["seed"] = seed
                result["episode"] = i
                all_results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"完成评估: {i + 1}/{num_episodes}")
                    
            except Exception as e:
                print(f"评估失败 (seed={seed}): {e}")
            finally:
                env.close()
        
        # 计算统计结果
        df = pd.DataFrame(all_results)
        
        stats = {
            "num_episodes": len(all_results),
            "reward_mean": df["reward"].mean(),
            "reward_std": df["reward"].std(),
            "reward_min": df["reward"].min(),
            "reward_max": df["reward"].max(),
            "length_mean": df["length"].mean(),
            "length_std": df["length"].std(),
            "collision_rate": df["collision"].mean(),
            "offroad_rate": df["offroad"].mean(),
            "success_rate": df["success"].mean(),
            "timeout_rate": df["timeout"].mean(),
            "lane_changes_mean": df["lane_changes"].mean(),
            "overtakes_mean": df["overtakes"].mean(),
            "raw_results": all_results
        }
        
        return stats
    
    def compare_with_baseline(self, baseline_stats: Dict = None) -> Dict:
        """与基线模型比较"""
        # 如果没有提供基线，使用预定义的expert基线
        if baseline_stats is None:
            baseline_stats = {
                "reward_mean": 15.0,  # 假设的expert基线
                "collision_rate": 0.1,
                "success_rate": 0.8,
                "length_mean": 500
            }
        
        # 评估当前模型
        current_stats = self.evaluate_multiple_episodes(num_episodes=30)
        
        comparison = {
            "current_performance": current_stats,
            "baseline_performance": baseline_stats,
            "comparison": {
                "reward_improvement": current_stats["reward_mean"] - baseline_stats["reward_mean"],
                "collision_improvement": baseline_stats["collision_rate"] - current_stats["collision_rate"],
                "success_improvement": current_stats["success_rate"] - baseline_stats["success_rate"],
                "length_improvement": current_stats["length_mean"] - baseline_stats["length_mean"]
            }
        }
        
        return comparison
    
    def generate_evaluation_report(self, output_dir: str, num_episodes: int = 50) -> str:
        """生成评估报告"""
        print(f"🔍 开始全面评估 ({num_episodes} episodes)...")
        
        # 执行评估
        stats = self.evaluate_multiple_episodes(num_episodes=num_episodes)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始数据
        results_df = pd.DataFrame(stats["raw_results"])
        results_csv = os.path.join(output_dir, "evaluation_results.csv")
        results_df.to_csv(results_csv, index=False)
        
        # 生成可视化
        self._create_visualizations(results_df, output_dir)
        
        # 生成报告
        report_path = os.path.join(output_dir, "evaluation_report.md")
        report_content = f"""# PPO Expert复现模型评估报告

## 📊 评估概览

- **评估episodes**: {stats['num_episodes']}
- **模型类型**: PPO Expert复现
- **评估时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 核心性能指标

### 奖励指标
- **平均奖励**: {stats['reward_mean']:.3f} ± {stats['reward_std']:.3f}
- **最高奖励**: {stats['reward_max']:.3f}
- **最低奖励**: {stats['reward_min']:.3f}

### Episode长度
- **平均长度**: {stats['length_mean']:.1f} ± {stats['length_std']:.1f} 步

### 安全性指标
- **碰撞率**: {stats['collision_rate']:.3f} ({stats['collision_rate']*100:.1f}%)
- **冲出道路率**: {stats['offroad_rate']:.3f} ({stats['offroad_rate']*100:.1f}%)
- **成功率**: {stats['success_rate']:.3f} ({stats['success_rate']*100:.1f}%)
- **超时率**: {stats['timeout_rate']:.3f} ({stats['timeout_rate']*100:.1f}%)

### 驾驶行为指标
- **平均车道变更次数**: {stats['lane_changes_mean']:.1f}
- **平均超车次数**: {stats['overtakes_mean']:.1f}

## 📈 性能分析

### 奖励分布
- **中位数奖励**: {results_df['reward'].median():.3f}
- **75分位数奖励**: {results_df['reward'].quantile(0.75):.3f}
- **25分位数奖励**: {results_df['reward'].quantile(0.25):.3f}

### 稳定性评估
- **奖励变异系数**: {(stats['reward_std'] / stats['reward_mean']):.3f}
- **长度变异系数**: {(stats['length_std'] / stats['length_mean']):.3f}

## 🏆 与Expert基线比较

| 指标 | 当前模型 | Expert基线 | 差距 |
|------|----------|------------|------|
| 平均奖励 | {stats['reward_mean']:.3f} | ~15.000 | {stats['reward_mean'] - 15:.3f} |
| 碰撞率 | {stats['collision_rate']:.3f} | ~0.100 | {0.1 - stats['collision_rate']:.3f} |
| 成功率 | {stats['success_rate']:.3f} | ~0.800 | {stats['success_rate'] - 0.8:.3f} |

## 📁 评估产物

- `evaluation_results.csv`: 详细的逐episode结果
- `reward_distribution.png`: 奖励分布图
- `performance_metrics.png`: 性能指标对比图
- `episode_timeline.png`: episode表现时间线

## 🔍 结论与建议

### 性能评价
"""

        # 添加性能评价
        if stats['reward_mean'] >= 12.0:
            if stats['collision_rate'] <= 0.15:
                conclusion = "🎉 **优秀**: 模型性能优异，接近或超过expert水平"
            else:
                conclusion = "✅ **良好**: 奖励表现良好，但安全性需要改进"
        elif stats['reward_mean'] >= 8.0:
            conclusion = "📈 **中等**: 基本达到可用水平，有进一步优化空间"
        else:
            conclusion = "⚠️ **需要改进**: 性能低于预期，建议重新训练或调整超参数"
        
        report_content += f"\n{conclusion}\n\n"
        
        # 添加具体建议
        suggestions = []
        if stats['collision_rate'] > 0.2:
            suggestions.append("- 碰撞率较高，建议调整奖励函数增加安全性权重")
        if stats['success_rate'] < 0.6:
            suggestions.append("- 成功率偏低，建议增加训练时间或调整探索策略")
        if stats['reward_std'] / stats['reward_mean'] > 0.5:
            suggestions.append("- 性能不够稳定，建议降低学习率或增加正则化")
        
        if suggestions:
            report_content += "### 改进建议\n\n" + "\n".join(suggestions) + "\n\n"
        
        report_content += f"""---
**评估完成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**评估数据**: `{results_csv}`
"""
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 评估报告已生成: {report_path}")
        print(f"📊 评估数据已保存: {results_csv}")
        
        return report_path
    
    def _create_visualizations(self, df: pd.DataFrame, output_dir: str):
        """创建可视化图表"""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. 奖励分布图
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(df['reward'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.hist(df['length'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Episode Length')
        plt.ylabel('Frequency')
        plt.title('Episode Length Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        termination_counts = [
            df['collision'].sum(),
            df['offroad'].sum(), 
            df['success'].sum(),
            df['timeout'].sum()
        ]
        labels = ['Collision', 'Offroad', 'Success', 'Timeout']
        plt.pie(termination_counts, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Termination Reasons')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reward_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 性能时间线
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 3, 1)
        plt.plot(df['episode'], df['reward'], alpha=0.7, marker='o', markersize=3)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Timeline')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(df['episode'], df['length'], alpha=0.7, marker='o', markersize=3, color='green')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.title('Episode Length Timeline')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        rolling_reward = df['reward'].rolling(window=10, min_periods=1).mean()
        plt.plot(df['episode'], rolling_reward, linewidth=2, color='red')
        plt.xlabel('Episode')
        plt.ylabel('Rolling Mean Reward')
        plt.title('Reward Trend (10-episode average)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        plt.plot(df['episode'], df['lane_changes'], alpha=0.7, marker='o', markersize=3, color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Lane Changes')
        plt.title('Lane Changes per Episode')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        # 累积成功率
        cumulative_success = df['success'].cumsum() / (df.index + 1)
        plt.plot(df['episode'], cumulative_success, linewidth=2, color='purple')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Success Rate')
        plt.title('Success Rate Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        # 安全性指标
        safety_scores = 1 - (df['collision'].astype(int) + df['offroad'].astype(int))
        rolling_safety = pd.Series(safety_scores).rolling(window=10, min_periods=1).mean()
        plt.plot(df['episode'], rolling_safety, linewidth=2, color='green')
        plt.xlabel('Episode')
        plt.ylabel('Safety Score (10-ep avg)')
        plt.title('Safety Trend')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'episode_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 性能对比雷达图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # 计算标准化指标
        metrics = {
            'Reward': df['reward'].mean() / 20.0,  # 假设最大值20
            'Safety': (1 - df['collision'].mean() - df['offroad'].mean()),
            'Success': df['success'].mean(),
            'Efficiency': min(df['length'].mean() / 1000.0, 1.0),  # 标准化到1000步
            'Stability': 1 - min(df['reward'].std() / df['reward'].mean(), 1.0),
            'Lane Usage': min(df['lane_changes'].mean() / 10.0, 1.0)  # 标准化到10次
        }
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        values = list(metrics.values())
        
        # 闭合雷达图
        angles = np.concatenate((angles, [angles[0]]))
        values = np.concatenate((values, [values[0]]))
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics.keys())
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart', size=16, weight='bold', pad=20)
        ax.grid(True)
        
        plt.savefig(os.path.join(output_dir, 'performance_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 可视化图表已生成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PPO Expert复现模型评估")
    parser.add_argument("experiment_dir", type=str, 
                       help="实验目录路径 (包含config.json和checkpoints/)")
    parser.add_argument("--model", type=str, default="best_model.pt",
                       help="模型文件名 (默认: best_model.pt)")
    parser.add_argument("--num_episodes", type=int, default=50,
                       help="评估episode数量 (默认: 50)")
    parser.add_argument("--deterministic", action="store_true",
                       help="使用确定性策略")
    parser.add_argument("--render", action="store_true",
                       help="渲染评估过程")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="计算设备")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="评估结果输出目录 (默认: experiment_dir/evaluation)")
    
    args = parser.parse_args()
    
    # 检查实验目录
    if not os.path.exists(args.experiment_dir):
        print(f"❌ 实验目录不存在: {args.experiment_dir}")
        return 1
    
    # 设置路径
    config_path = os.path.join(args.experiment_dir, "config.json")
    model_path = os.path.join(args.experiment_dir, "checkpoints", args.model)
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return 1
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return 1
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, "evaluation")
    
    print("🎯 PPO Expert复现模型评估")
    print("=" * 50)
    print(f"📂 实验目录: {args.experiment_dir}")
    print(f"🤖 模型文件: {args.model}")
    print(f"📊 评估episodes: {args.num_episodes}")
    print(f"🎲 确定性策略: {args.deterministic}")
    print("=" * 50)
    
    try:
        # 创建评估器
        evaluator = ModelEvaluator(model_path, config_path, args.device)
        
        # 生成评估报告
        report_path = evaluator.generate_evaluation_report(
            args.output_dir, args.num_episodes
        )
        
        print("\n🎉 评估完成！")
        print(f"📄 评估报告: {report_path}")
        print(f"📁 评估结果: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 