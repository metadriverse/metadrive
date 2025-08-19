#!/bin/bash

# PPO检查点继续训练脚本
# 使用您提供的具体超参数配置

echo "🚀 开始基于检查点的PPO继续训练"
echo "=================================================="

# 检查点路径 - 使用最新的检查点
CHECKPOINT_PATH="runs/ppo_expert_reproduction_20250818_112439/checkpoints/latest_model.pt"

# 检查检查点文件是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ 检查点文件不存在: $CHECKPOINT_PATH"
    echo "💡 请先查看可用的检查点文件："
    echo "   ls -la runs/*/checkpoints/"
    exit 1
fi

echo "✅ 检查点文件验证成功: $CHECKPOINT_PATH"
echo "📦 文件大小: $(du -h $CHECKPOINT_PATH | cut -f1)"
echo ""

echo "🎯 训练配置："
echo "   总训练步数: 50,000,000"
echo "   学习率: 1e-4 (精细调优)"
echo "   并行环境: 8 (真正多进程)"
echo "   批次大小: 512 (大批次稳定训练)"
echo "   裁剪范围: 0.10 (保守策略更新)"
echo "   值函数系数: 1.0 (增强值函数学习)"
echo "   熵系数衰减: 0.02 → 0.01"
echo "   目标KL散度: 0.03 (早停控制)"
echo ""

echo "🔧 执行命令："
echo "python3 ppo_expert_reproduction.py \\"
echo "    --resume_from $CHECKPOINT_PATH \\"
echo "    --total_timesteps 50000000 \\"
echo "    --eval_freq 10 --seed 46 \\"
echo "    --lr 1e-4 --n_steps 2048 --batch_size 512 \\"
echo "    --gamma 0.99 --gae_lambda 0.95 --clip_range 0.10 \\"
echo "    --vf_coef 1.0 --max_grad_norm 0.5 \\"
echo "    --entropy_coef_start 0.02 --entropy_coef_end 0.01 \\"
echo "    --entropy_decay_end_ratio 0.95 --device cpu \\"
echo "    --n_envs 8 --n_epochs 8 --target_kl 0.03"
echo ""

# 询问用户确认
read -p "🤔 是否立即开始训练？[y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 开始训练..."
    echo ""
    
    # 执行训练命令
    python3 ppo_expert_reproduction.py \
        --resume_from "$CHECKPOINT_PATH" \
        --total_timesteps 50000000 \
        --eval_freq 10 --seed 46 \
        --lr 1e-4 --n_steps 2048 --batch_size 512 \
        --gamma 0.99 --gae_lambda 0.95 --clip_range 0.10 \
        --vf_coef 1.0 --max_grad_norm 0.5 \
        --entropy_coef_start 0.02 --entropy_coef_end 0.01 \
        --entropy_decay_end_ratio 0.95 --device cpu \
        --n_envs 8 --n_epochs 8 --target_kl 0.03
    
    echo ""
    echo "✅ 训练完成！"
else
    echo "📝 训练已取消。您可以稍后手动执行上述命令。"
fi 