# PPO ckpt 仿真控制器使用说明

## 🎯 功能概述

PPO检查点仿真控制器（`ppo_checkpoint_simulation.py`）是一个专门用于加载训练好的PPO检查点并在MetaDrive仿真环境中控制主车行为的工具。它提供了以下核心功能：

- ✅ **检查点加载**: 加载指定的PPO训练检查点
- ✅ **仿真控制**: 使用训练好的PPO策略控制MetaDrive中的主车
- ✅ **可视化展示**: 实时渲染仿真过程
- ✅ **性能评估**: 评估模型在多个场景下的表现
- ✅ **统计分析**: 详细的性能指标统计和分析

## 🚀 快速开始

### 1. 基础使用 - 加载默认检查点

```bash
# 使用默认检查点运行5个episodes的仿真
cd /home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction

python3 ppo_checkpoint_simulation.py
```

### 2. 指定检查点文件

```bash
# 使用特定检查点文件
python3 ppo_checkpoint_simulation.py \
    --checkpoint ./runs/ppo_expert_reproduction_20250818_112527/checkpoints/checkpoint_350.pt \
    --episodes 3
```

### 3. 无渲染模式（快速评估）

```bash
# 无渲染模式，适合批量评估
python3 ppo_checkpoint_simulation.py \
    --no_render \
    --episodes 10
```

### 4. 模型性能评估

```bash
# 运行全面的模型评估
python3 ppo_checkpoint_simulation.py \
    --evaluate \
    --eval_episodes 20 \
    --no_render
```

## 📖 详细使用指南

### 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | str | `checkpoint_350.pt` | 检查点文件路径 |
| `--config` | str | None | 配置文件路径（可选）|
| `--episodes` | int | 5 | 仿真episodes数量 |
| `--max_steps` | int | 1000 | 每个episode最大步数 |
| `--no_render` | flag | False | 禁用渲染显示 |
| `--stochastic` | flag | False | 使用随机策略 |
| `--evaluate` | flag | False | 运行模型评估 |
| `--eval_episodes` | int | 20 | 评估episodes数量 |
| `--device` | str | auto | 计算设备 (auto/cpu/cuda) |

### 使用场景示例

#### 场景1: 快速验证模型效果
```bash
# 快速查看模型在3个不同场景下的表现
python3 ppo_checkpoint_simulation.py \
    --episodes 3 \
    --max_steps 500
```

#### 场景2: 详细性能评估
```bash
# 在20个场景下评估模型性能，不显示渲染
python3 ppo_checkpoint_simulation.py \
    --evaluate \
    --eval_episodes 20 \
    --no_render
```

#### 场景3: 比较不同检查点
```bash
# 评估checkpoint_300.pt
python3 ppo_checkpoint_simulation.py \
    --checkpoint ./runs/ppo_expert_reproduction_20250818_112527/checkpoints/checkpoint_300.pt \
    --evaluate \
    --eval_episodes 10

# 评估checkpoint_350.pt
python3 ppo_checkpoint_simulation.py \
    --checkpoint ./runs/ppo_expert_reproduction_20250818_112527/checkpoints/checkpoint_350.pt \
    --evaluate \
    --eval_episodes 10
```

#### 场景4: 随机策略测试
```bash
# 测试模型的随机策略表现
python3 ppo_checkpoint_simulation.py \
    --stochastic \
    --episodes 5
```

## 📊 输出说明

### 实时输出示例
```
✅ PPO检查点仿真控制器初始化完成
📁 检查点: checkpoint_350.pt
🔧 设备: cuda
🎯 训练迭代: 350
🚀 全局步数: 179200

🚗 开始PPO仿真 (5 episodes)
============================================================

🎮 Episode 1/5
📊 Episode 1 结果:
   💰 总奖励: 125.34
   📏 Episode长度: 342
   🏁 成功到达: ✅
   💥 发生碰撞: ✅
   🛣️  冲出道路: ✅
   🚀 最高速度: 12.45 m/s
   📈 平均速度: 8.23 m/s
   🎯 路径完成度: 85.2%
```

### 性能评估输出
```
📈 总体统计 (5 episodes)
============================================================
🏆 成功率: 60.0%
💥 碰撞率: 20.0%
🛣️  冲出道路率: 40.0%
💰 平均奖励: 98.45
📏 平均Episode长度: 287.4
🚀 平均速度: 7.89 m/s
🎯 平均路径完成度: 72.3%
```

## 🔧 高级配置

### 自定义环境配置

如果需要修改仿真环境配置，可以编辑脚本中的`_get_default_config`方法：

```python
def _get_default_config(self):
    return {
        "num_scenarios": 1,           # 场景数量
        "traffic_density": 0.3,       # 交通密度
        "start_seed": 8888,           # 起始种子
        "random_traffic": False,      # 随机交通
        "random_agent_model": False,  # 随机智能体模型
        "use_render": True,           # 渲染显示
        "horizon": 1000,              # 时域长度
        "accident_prob": 0.0,         # 事故概率
        # 奖励配置
        "crash_vehicle_penalty": 8.0,
        "crash_object_penalty": 8.0,
        "out_of_road_penalty": 8.0,
        "success_reward": 20.0,
        "driving_reward": 2.0,
        "speed_reward": 0.3,
        "use_lateral_reward": True,
        # 车辆配置
        "vehicle_config": {
            "lidar": {
                "num_lasers": 240,
                "distance": 50,
                "num_others": 4,
            }
        }
    }
```

### 批量评估脚本

创建批量评估脚本来比较多个检查点：

```bash
#!/bin/bash
# 批量评估多个检查点

CHECKPOINTS=(
    "checkpoint_300.pt"
    "checkpoint_350.pt" 
    "checkpoint_400.pt"
    "best_model.pt"
)

for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "评估检查点: $checkpoint"
    python3 ppo_checkpoint_simulation.py \
        --checkpoint "./runs/ppo_expert_reproduction_20250818_112527/checkpoints/$checkpoint" \
        --evaluate \
        --eval_episodes 15 \
        --no_render
    echo "--------------------------------"
done
```

## 📈 性能指标说明

### 核心指标

| 指标 | 说明 | 理想值 |
|------|------|--------|
| **成功率** | 成功到达目标的episode比例 | > 50% |
| **碰撞率** | 发生碰撞的episode比例 | < 20% |
| **冲出道路率** | 冲出道路的episode比例 | < 30% |
| **平均奖励** | 每个episode的平均奖励 | > 80 |
| **平均速度** | 车辆的平均行驶速度 | 6-12 m/s |
| **路径完成度** | 完成预定路径的比例 | > 70% |

### 性能评估标准

- **优秀**: 成功率 > 70%, 碰撞率 < 10%
- **良好**: 成功率 > 50%, 碰撞率 < 20%  
- **一般**: 成功率 > 30%, 碰撞率 < 40%
- **需要改进**: 成功率 < 30% 或 碰撞率 > 40%

## 🔍 故障排除

### 常见问题

#### 1. 检查点加载失败
```
❌ 检查点文件不存在: xxx.pt
```
**解决方案**: 检查文件路径是否正确，确保检查点文件存在。

#### 2. GPU内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**: 使用CPU模式 `--device cpu` 或减少batch size。

#### 3. MetaDrive环境初始化失败
```
❌ 环境创建失败: xxx
```
**解决方案**: 确保MetaDrive正确安装，检查依赖项。

#### 4. 渲染窗口无法显示
**解决方案**: 
- 确保有图形界面支持
- 使用 `--no_render` 跳过渲染
- 检查X11转发设置

### 调试模式

如需调试，可以修改脚本添加详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 扩展开发

### 添加自定义指标

可以在`run_single_episode`方法中添加自定义统计指标：

```python
# 在episode_stats中添加新指标
episode_stats["custom_metric"] = custom_value

# 在_print_summary_stats中添加显示逻辑
custom_avg = np.mean([s['custom_metric'] for s in all_stats])
print(f"🔧 自定义指标: {custom_avg:.2f}")
```

### 集成到现有训练流程

可以将仿真控制器集成到训练脚本中，用于实时评估：

```python
from ppo_checkpoint_simulation import PPOCheckpointSimulator

# 在训练过程中
if iteration % eval_freq == 0:
    simulator = PPOCheckpointSimulator(checkpoint_path)
    evaluation = simulator.evaluate_model(num_episodes=10, render=False)
    print(f"Eval Success Rate: {evaluation['success_rate']:.1%}")
```

## 📚 相关文档

- [MetaDrive 官方文档](https://metadrive-simulator.readthedocs.io/)
- [PPO算法原理](https://arxiv.org/abs/1707.06347)
- [训练配置优化指南](./extra/奖励配置优化指南.md)

## ⚡ 总结

PPO检查点仿真控制器提供了一个完整的解决方案，用于：

1. ✅ **加载训练好的PPO检查点**
2. ✅ **在MetaDrive仿真中控制主车行为**  
3. ✅ **可视化展示仿真过程**
4. ✅ **评估模型性能指标**
5. ✅ **支持批量测试和比较**

通过这个工具，您可以方便地验证训练模型的效果，并在不同场景下测试其驾驶行为的表现。 