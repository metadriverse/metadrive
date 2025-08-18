# MetaDrive PPO Expert 复现训练报告

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
- **随机种子**: 42

### 关键超参数 (可调整)
- **学习率**: 0.0001
- **rollout步数**: 4096
- **环境数量**: 1
- **批次大小**: 512
- **训练轮次**: 10
- **折扣因子**: 0.99
- **GAE Lambda**: 0.95
- **裁剪范围**: 0.15
- **熵系数**: 0.0015

## 📊 训练结果

### 最终性能
- **平均奖励**: 331.673 ± 186.332
- **平均长度**: 341.9
- **碰撞率**: 0.550
- **冲出道路率**: 0.000
- **成功率**: 0.450

### 训练统计
- **总训练步数**: 2,002,944
- **最终策略损失**: 0.038313
- **最终值函数损失**: 83.117551
- **最终熵值**: 2.508522

## 🎯 使用方法

### 基础训练
```bash
python ppo_expert_reproduction.py
```

### 自定义超参数
```bash
python ppo_expert_reproduction.py \
    --lr 3e-4 \
    --n_steps 2048 \
    --n_envs 8 \
    --batch_size 256 \
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
tensorboard --logdir /home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/runs/ppo_expert_reproduction_20250818_112439/tensorboard
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
- **验证频率**: 每2次迭代
- **验证episode数**: 10 (最终评估20)
- **确定性策略**: 使用动作均值

### 最优模型选择标准
1. **主要指标**: 验证集平均episode奖励
2. **约束条件**: 碰撞率不劣化
3. **辅助指标**: 成功率、episode长度

## 🔄 复现性保证

### 依赖版本
- Python: 3.10.18
- PyTorch: 2.8.0+cu128
- NumPy: 2.2.6

### 随机性控制
- **全局种子**: 42
- **PyTorch种子**: 已设置
- **NumPy种子**: 已设置

### 硬件环境
- **计算设备**: cuda
- **训练时长**: 预计2-4小时 (取决于硬件)

## 📁 产物说明

```
ppo_expert_reproduction_20250818_112439/
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
env_config.update({
    "num_scenarios": 5000,  # 更多场景
    "map": "SSSSSSSS"      # 自定义地图
})
```

### 自定义奖励函数
继承`MetaDriveEnv`并重写`reward_function`方法。

---
**报告生成时间**: 2025-08-18 15:00:49  
**实验目录**: `/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/runs/ppo_expert_reproduction_20250818_112439`
