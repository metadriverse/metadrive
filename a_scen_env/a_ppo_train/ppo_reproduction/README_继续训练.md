# PPO 检查点继续训练使用说明

## 🎯 功能概述

`ppo_expert_reproduction.py` 已内置完整的检查点恢复功能，支持从任何已保存的检查点文件继续训练。

## 📁 检查点文件位置

训练过程中会自动保存检查点到：
```
runs/ppo_expert_reproduction_YYYYMMDD_HHMMSS/checkpoints/
├── best_model.pt              # 最佳模型（基于评估奖励）
├── latest_model.pt            # 最新模型（总是最新状态）
├── checkpoint_50.pt           # 定期检查点（每50次迭代）
├── checkpoint_100.pt
└── ...
```

## 🚀 基本使用方法

### 1. 从最新检查点继续训练

```bash
python3 ppo_expert_reproduction.py \
    --resume_from runs/ppo_expert_reproduction_20250818_112439/checkpoints/latest_model.pt \
    --total_timesteps 10000000
```

### 2. 从最佳模型检查点继续训练

```bash
python3 ppo_expert_reproduction.py \
    --resume_from runs/ppo_expert_reproduction_20250818_112439/checkpoints/best_model.pt \
    --total_timesteps 10000000
```

### 3. 从特定迭代检查点继续训练

```bash
python3 ppo_expert_reproduction.py \
    --resume_from runs/ppo_expert_reproduction_20250818_112439/checkpoints/checkpoint_300.pt \
    --total_timesteps 10000000
```

## 🎛️ 您的高级配置示例

使用您提供的具体超参数配置：

```bash
python3 ppo_expert_reproduction.py \
    --resume_from runs/ppo_expert_reproduction_20250818_112439/checkpoints/latest_model.pt \
    --total_timesteps 50000000 \
    --eval_freq 10 \
    --seed 46 \
    --lr 1e-4 \
    --n_steps 2048 \
    --batch_size 512 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_range 0.10 \
    --vf_coef 1.0 \
    --max_grad_norm 0.5 \
    --entropy_coef_start 0.02 \
    --entropy_coef_end 0.01 \
    --entropy_decay_end_ratio 0.95 \
    --device cpu \
    --n_envs 8 \
    --n_epochs 8 \
    --target_kl 0.03
```

## 🔧 超参数说明

| 参数 | 值 | 说明 |
|------|----|----|
| `--total_timesteps` | 50000000 | 总训练步数（5千万步长时间训练）|
| `--lr` | 1e-4 | 学习率（较低，适合精细调优）|
| `--n_envs` | 8 | 真正的8个并行环境 |
| `--batch_size` | 512 | 大批次训练（提高稳定性）|
| `--clip_range` | 0.10 | 更保守的策略更新 |
| `--vf_coef` | 1.0 | 增强值函数学习 |
| `--entropy_coef_start/end` | 0.02→0.01 | 探索到利用的渐进过渡 |
| `--target_kl` | 0.03 | KL散度早停控制 |

## 📊 训练恢复效果

恢复训练时会自动：
- ✅ 恢复网络权重和优化器状态
- ✅ 继续使用原实验目录（日志连续性）
- ✅ 保持TensorBoard日志连续
- ✅ 维持CSV训练记录连续性
- ✅ 从正确的迭代步数继续

## 🔍 快速检查检查点

查看可用的检查点：
```bash
ls -la runs/*/checkpoints/*.pt
```

查看检查点信息：
```bash
python3 -c "
import torch
ckpt = torch.load('runs/ppo_expert_reproduction_20250818_112439/checkpoints/latest_model.pt')
print(f'迭代: {ckpt[\"iteration\"]}')
print(f'全局步数: {ckpt[\"global_step\"]:,}')
print(f'训练时间: {ckpt.get(\"timestamp\", \"N/A\")}')
"
```

## ⚠️ 注意事项

1. **检查点兼容性**：确保新的超参数与原检查点兼容
2. **随机种子**：使用 `--seed` 参数控制可重现性
3. **设备兼容**：如果原来用GPU训练，现在用CPU，系统会自动适配
4. **路径检查**：确保检查点文件路径正确且文件存在

## 🛠️ 故障排除

**问题1**：检查点文件不存在
```bash
# 解决：检查路径是否正确
ls -la runs/*/checkpoints/
```

**问题2**：配置不兼容警告
```
# 这通常是安全警告，可以继续训练
# 如需确保完全兼容，使用原始训练配置
```

**问题3**：内存不足
```bash
# 减少并行环境数量
--n_envs 4

# 或减少批次大小
--batch_size 256
```

## 🎉 开始您的继续训练

根据您的配置，推荐的完整命令：

```bash
# 替换为您实际的检查点路径
CHECKPOINT_PATH="runs/ppo_expert_reproduction_20250818_112439/checkpoints/latest_model.pt"

python3 ppo_expert_reproduction.py \
    --resume_from $CHECKPOINT_PATH \
    --total_timesteps 50000000 \
    --eval_freq 10 --seed 46 \
    --lr 1e-4 --n_steps 2048 --batch_size 512 \
    --gamma 0.99 --gae_lambda 0.95 --clip_range 0.10 \
    --vf_coef 1.0 --max_grad_norm 0.5 \
    --entropy_coef_start 0.02 --entropy_coef_end 0.01 \
    --entropy_decay_end_ratio 0.95 --device cpu \
    --n_envs 8 --n_epochs 8 --target_kl 0.03
```

现在就可以开始您的高级配置继续训练了！🚀 