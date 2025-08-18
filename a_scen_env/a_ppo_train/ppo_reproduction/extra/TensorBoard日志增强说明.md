# TensorBoard日志增强功能说明

## 📊 新增TensorBoard曲线列表

### 🔧 训练相关指标 (train/)
- **train/clipfrac** - PPO裁剪比例，显示有多少比率被裁剪
- **train/explained_variance** - 解释方差，衡量值函数拟合质量
- **train/grad_norm** - 梯度范数，监控梯度大小
- **train/learning_rate** - 学习率变化曲线
- **train/policy_loss** - 策略损失
- **train/value_loss** - 值函数损失  
- **train/entropy** - 策略熵值
- **train/approx_kl** - 近似KL散度

### 💥 损失分解 (loss/)
- **loss/actor_loss** - Actor损失（等同于policy_loss）
- **loss/entropy_loss** - 熵损失项

### 🌍 环境统计 (env/)
- **env/ep_rew_mean** - Episode平均奖励
- **env/ep_rew_max** - Episode最大奖励
- **env/ep_rew_min** - Episode最小奖励
- **env/ep_len_mean** - Episode平均长度
- **env/ep_len_max** - Episode最大长度
- **env/ep_len_min** - Episode最小长度
- **env/time_outs** - 超时率

### 🎯 评估指标 (eval/)
- **eval/reward_mean** - 评估平均奖励
- **eval/reward_std** - 评估奖励标准差
- **eval/length_mean** - 评估平均长度
- **eval/collision_rate** - 碰撞率
- **eval/offroad_rate** - 冲出道路率
- **eval/success_rate** - 成功率
- **eval/avg_speed** - 平均速度
- **eval/lane_deviation** - 车道偏移度
- **eval/lane_change_count** - 车道变换次数
- **eval/min_ttc** - 最小碰撞时间
- **eval/path_completion** - 路径完成度

## 🚀 访问TensorBoard

### 启动命令
```bash
tensorboard --logdir runs/[实验目录]/tensorboard --port 6008 --host 0.0.0.0
```

### 访问地址
- http://localhost:6008
- http://127.0.0.1:6008

## 📈 关键监控指标说明

### 训练稳定性指标
1. **clipfrac**: 应该在0.1-0.3之间，过高说明策略更新过激进
2. **explained_variance**: 接近1表示值函数拟合良好
3. **grad_norm**: 监控梯度爆炸，应该相对稳定
4. **approx_kl**: KL散度应该保持在较小值

### 驾驶性能指标
1. **success_rate**: 任务完成率，目标是逐步提升
2. **collision_rate**: 碰撞率，应该逐步降低
3. **avg_speed**: 平均速度，反映驾驶效率
4. **lane_deviation**: 车道保持能力
5. **path_completion**: 路径完成度

### 训练效果指标
1. **reward_mean**: 平均奖励应该逐步增长
2. **entropy**: 策略探索性，训练初期较高，后期降低
3. **policy_loss**: 策略损失应该逐步降低并稳定

## 📝 CSV日志增强

### 新增CSV列
- clipfrac: 裁剪比例
- explained_variance: 解释方差
- grad_norm: 梯度范数
- avg_speed: 平均速度
- lane_deviation: 车道偏移
- lane_change_count: 车道变换次数
- min_ttc: 最小碰撞时间
- path_completion: 路径完成度

## 🔍 使用建议

### 监控优先级
1. **主要指标**: reward_mean, success_rate, collision_rate
2. **训练稳定性**: clipfrac, explained_variance, grad_norm
3. **驾驶质量**: avg_speed, lane_deviation, path_completion
4. **算法性能**: policy_loss, value_loss, entropy

### 异常检测
- **clipfrac > 0.5**: 学习率可能过高
- **explained_variance < 0**: 值函数拟合失败
- **grad_norm 急剧增长**: 梯度爆炸，需要调整学习率
- **entropy 快速下降**: 策略过早收敛，可能需要增加熵系数

## 🛠️ 代码实现要点

### Episode统计缓冲区
- 使用`deque(maxlen=100)`存储最近100个episode的统计
- 自动计算滑动平均值

### 评估数据收集
- 在rollout和evaluation阶段收集详细驾驶数据
- 兼容MetaDrive的agent接口（替代vehicle接口）

### TensorBoard组织
- 按功能分组：train/, loss/, env/, eval/
- 统一的数据记录时机和格式 