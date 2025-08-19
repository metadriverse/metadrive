#!/usr/bin/env python3
"""
基于已有检查点继续PPO训练脚本
演示如何使用--resume_from参数从检查点恢复训练
"""

import os
import sys
import subprocess
from pathlib import Path


def find_latest_checkpoint():
    """查找最新的检查点文件"""
    current_dir = Path(__file__).parent.absolute()
    runs_dir = current_dir / "runs"
    
    if not runs_dir.exists():
        print("❌ 没有找到训练结果目录")
        return None
    
    # 查找所有检查点文件
    checkpoint_files = []
    for exp_dir in runs_dir.iterdir():
        if exp_dir.is_dir():
            checkpoints_dir = exp_dir / "checkpoints"
            if checkpoints_dir.exists():
                for ckpt_file in checkpoints_dir.glob("*.pt"):
                    checkpoint_files.append(ckpt_file)
    
    if not checkpoint_files:
        print("❌ 没有找到任何检查点文件")
        return None
    
    # 按修改时间排序，获取最新的
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    return latest_checkpoint


def continue_training_with_checkpoint(checkpoint_path: str):
    """基于检查点继续训练"""
    
    print(f"🚀 基于检查点继续训练")
    print(f"📁 检查点文件: {checkpoint_path}")
    print("=" * 80)
    
    # 构建训练命令 - 使用您提供的超参数配置
    cmd = [
        "python3", "ppo_expert_reproduction.py",
        "--resume_from", checkpoint_path,
        "--total_timesteps", "50000000",
        "--eval_freq", "10", 
        "--seed", "46",
        "--lr", "1e-4",
        "--n_steps", "2048",
        "--batch_size", "512",
        "--gamma", "0.99",
        "--gae_lambda", "0.95",
        "--clip_range", "0.10",
        "--vf_coef", "1.0",
        "--max_grad_norm", "0.5",
        "--entropy_coef_start", "0.02",
        "--entropy_coef_end", "0.01",
        "--entropy_decay_end_ratio", "0.95",
        "--device", "cpu",
        "--n_envs", "8",
        "--n_epochs", "8",
        "--target_kl", "0.03"
    ]
    
    print("🔧 训练命令:")
    print(" ".join(cmd))
    print()
    print("🎯 关键配置:")
    print(f"   总训练步数: 50,000,000")
    print(f"   学习率: 1e-4 (降低学习率进行精细调优)")
    print(f"   并行环境: 8 (真正的多进程并行)")
    print(f"   批次大小: 512 (大批次训练)")
    print(f"   裁剪范围: 0.10 (更保守的策略更新)")
    print(f"   值函数系数: 1.0 (增强值函数学习)")
    print(f"   熵系数衰减: 0.02 → 0.01 (探索到利用的渐进过渡)")
    print(f"   目标KL散度: 0.03 (早停控制)")
    print()
    
    # 询问用户是否继续
    response = input("🤔 是否立即开始继续训练？[y/N]: ").strip().lower()
    
    if response in ['y', 'yes']:
        print("🚀 开始继续训练...")
        try:
            # 执行训练命令
            subprocess.run(cmd, check=True)
            print("✅ 训练完成！")
        except subprocess.CalledProcessError as e:
            print(f"❌ 训练过程中出现错误: {e}")
        except KeyboardInterrupt:
            print("\n⚠️ 训练被用户中断")
    else:
        print("📝 训练命令已准备就绪，您可以稍后手动执行:")
        print()
        print("cd " + str(Path(__file__).parent.absolute()))
        print(" ".join(cmd))


def main():
    """主函数"""
    print("🔄 PPO训练检查点恢复工具")
    print("=" * 50)
    
    # 查找最新检查点
    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint is None:
        print("💡 建议先运行一次正常训练生成检查点:")
        print("python3 ppo_expert_reproduction.py --total_timesteps 100000")
        return
    
    print(f"📍 找到最新检查点: {latest_checkpoint.name}")
    print(f"📁 完整路径: {latest_checkpoint}")
    print(f"📅 修改时间: {latest_checkpoint.stat().st_mtime}")
    print()
    
    # 检查检查点文件有效性
    if not latest_checkpoint.exists():
        print(f"❌ 检查点文件不存在: {latest_checkpoint}")
        return
        
    file_size = latest_checkpoint.stat().st_size / (1024 * 1024)  # MB
    print(f"📦 文件大小: {file_size:.1f} MB")
    
    if file_size < 1:  # 小于1MB可能是损坏的文件
        print("⚠️ 检查点文件似乎太小，可能已损坏")
        return
    
    print("✅ 检查点文件验证通过")
    print()
    
    # 继续训练
    continue_training_with_checkpoint(str(latest_checkpoint))


if __name__ == "__main__":
    main() 