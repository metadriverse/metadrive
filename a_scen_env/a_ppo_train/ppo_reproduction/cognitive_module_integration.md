# 认知模块集成文档

## 概述
本文档记录了将`ppo_checkpoint_simulation_with_cog.py`中的认知模块功能完整集成到`ppo_expert_reproduction_with_cog.py`训练脚本中的详细内容。

## 时间
2025-01-20

## 集成的认知模块架构

### 1. 认知偏差模块 (Cognitive Bias Module)
**功能**: 基于时间碰撞(TTA)和视觉检测实现风险厌恶行为
- **主要组件**:
  - TTA逆向偏差计算
  - 视觉厌恶区域检测
  - 自适应偏差调整
  - 环境附加与分离机制

### 2. 认知延迟模块 (Cognitive Delay Module)
**功能**: 模拟人类反应延迟
- **主要组件**:
  - 动作缓冲队列
  - 可配置延迟步数
  - 动作平滑机制
  - PPO训练模式支持

### 3. 认知感知模块 (Cognitive Perception Module)
**功能**: 模拟感知噪声和不确定性
- **主要组件**:
  - 雷达噪声注入
  - 卡尔曼滤波
  - 环境传感器替换
  - 实时噪声可视化

## 从仿真脚本迁移的关键功能

### 1. 环境附加机制 (attach_to_env/detach_from_env)

#### 仿真脚本中的实现
```python
# 认知感知模块附加到环境
if self.cognitive_perception_module:
    self.cognitive_perception_module.reset()
    self.cognitive_perception_module.attach_to_env(env)
    print("🔗 认知感知模块已附加到环境 - 噪声将在传感器层自动注入")
    
# 认知偏差模块附加到环境
if self.cognitive_bias_module:
    success = self.cognitive_bias_module.attach_to_env(env)
    if success:
        print("🔗 认知偏差模块已附加到环境 - 将基于TTA动态调整奖励")
```

#### 需要添加到训练脚本的功能
- 在环境创建后自动附加认知模块
- 在训练结束时安全分离认知模块
- 支持多环境并行时的附加机制

### 2. 认知感知模块与偏差模块的联动

#### 仿真脚本中的实现
```python
# 认知偏差模块引用感知模块
self.cognitive_bias_module = CognitiveBiasModule(
    bias_config=bias_config,
    cognitive_perception_module=self.cognitive_perception_module
)
```

#### 训练脚本需要的改进
- 在初始化认知偏差模块时传入感知模块引用
- 实现两个模块之间的数据共享机制

### 3. 可视化数据收集

#### 仿真脚本中的高级功能
```python
self.cognitive_viz_data = {
    'timestamps': [],
    'bias_strength': [],
    'bias_applied': [],
    'delay_steps': [],
    'delay_applied': [],
    'perception_noise': [],
    'perception_applied': [],
    'original_rewards': [],
    'modified_rewards': [],
    'original_actions': [],
    'delayed_actions': [],
    'original_observations': [],
    'noisy_observations': [],
    'step_count': []
}
```

#### 训练脚本的增强需求
- 添加认知模块效果的实时可视化
- 在TensorBoard中记录认知模块指标
- 生成认知影响分析报告

### 4. 前方雷达束信息获取

#### 仿真脚本独有功能
```python
front_beam_data = self.cognitive_perception_module.get_front_beam_info()
noise_level = front_beam_data.get('noise_level', 0.0)
```

#### 训练脚本集成建议
- 在训练过程中监控前方雷达噪声水平
- 用于分析感知噪声对决策的影响

## 训练脚本需要添加的具体内容

### 1. 认知模块完整初始化
```python
# 在PPOExpertReproduction.__init__中添加
if self.use_cognitive_modules:
    # 先初始化感知模块（其他模块可能依赖它）
    if args.use_cognitive_perception:
        perception_config = {
            'sigma0': args.perception_sigma0,
            'k': args.perception_k,
            'p_miss0': args.perception_p_miss0,
            'p_false': args.perception_p_false,
            'use_ar1': True,  # 新增AR(1)过程
            'rho': 0.8,        # AR(1)相关系数
            'use_kf': args.perception_use_kf,
            'kf_dt': args.perception_kf_dt,
            'kf_q_scale': args.perception_kf_q_scale
        }
        self.cognitive_perception_module = CognitivePerceptionModule(perception_config)
        
    # 初始化偏差模块（传入感知模块引用）
    if args.use_cognitive_bias:
        bias_config = {
            'inverse_tta_coef': args.bias_inverse_tta_coef,
            'tta_threshold': args.bias_tta_threshold,
            'adaptive_bias': args.bias_adaptive,
            'adaptation_rate': args.bias_adaptation_rate,
            'visual_detection_distance': args.bias_visual_distance,
            'visual_detection_angle': args.bias_visual_angle,
            'visual_aversion_strength': args.bias_visual_strength,
            'verbose': args.cognitive_verbose
        }
        self.cognitive_bias_module = CognitiveBiasModule(
            bias_config,
            cognitive_perception_module=self.cognitive_perception_module  # 关键：传入感知模块
        )
```

### 2. 环境附加机制
```python
# 在create_environments方法后添加附加逻辑
def _attach_cognitive_modules_to_env(self, env):
    """将认知模块附加到环境"""
    if not self.use_cognitive_modules:
        return
        
    # 附加感知模块（必须先于偏差模块）
    if self.cognitive_perception_module:
        self.cognitive_perception_module.attach_to_env(env)
        print("🔗 认知感知模块已附加到环境")
        
    # 附加偏差模块
    if self.cognitive_bias_module:
        success = self.cognitive_bias_module.attach_to_env(env)
        if success:
            print("🔗 认知偏差模块已附加到环境")
```

### 3. 认知可视化增强
```python
# 在log_metrics方法中添加
if self.use_cognitive_modules:
    # 记录认知偏差强度
    if self.cognitive_bias_module:
        bias_info = self.cognitive_bias_module.get_bias_info()
        if isinstance(bias_info, dict):
            self.writer.add_scalar("cognitive/bias_strength", 
                                  bias_info.get('bias_strength', 0.0), 
                                  self.global_step)
            self.writer.add_scalar("cognitive/bias_active_ratio", 
                                  bias_info.get('active_ratio', 0.0), 
                                  self.global_step)
    
    # 记录认知延迟信息
    if self.cognitive_delay_module:
        delay_info = self.cognitive_delay_module.get_delay_info()
        self.writer.add_scalar("cognitive/current_delay", 
                              delay_info.get('current_delay', 0), 
                              self.global_step)
    
    # 记录感知噪声水平
    if self.cognitive_perception_module:
        front_beam = self.cognitive_perception_module.get_front_beam_info()
        self.writer.add_scalar("cognitive/perception_noise", 
                              front_beam.get('noise_level', 0.0), 
                              self.global_step)
```

### 4. 奖励处理改进
```python
# 在collect_rollouts方法中改进奖励处理
if self.use_cognitive_modules and self.cognitive_bias_module:
    # 处理奖励偏差
    adjusted_reward, bias_info = self.cognitive_bias_module.process_reward(
        original_reward=rewards[env_idx],
        env=current_env,
        info=infos[env_idx],
        is_ppo_mode=True
    )
    
    # 记录偏差信息用于分析
    if isinstance(bias_info, dict):
        bias_amount = bias_info.get('bias_applied', 0.0)
        if abs(bias_amount) > 1e-6:
            print(f"🧠 认知偏差应用: {rewards[env_idx]:.3f} → {adjusted_reward:.3f}")
    
    rewards[env_idx] = adjusted_reward
```

### 5. 检查点保存增强
```python
# 在save_checkpoint方法中添加
if self.use_cognitive_modules:
    cognitive_states = {}
    
    if self.cognitive_bias_module:
        cognitive_states['bias_module'] = {
            'adaptive_factor': self.cognitive_bias_module.adaptive_factor,
            'step_count': self.cognitive_bias_module._step_count,
            'total_bias': self.cognitive_bias_module._total_bias,
            'active_steps': self.cognitive_bias_module._active_steps
        }
    
    if self.cognitive_delay_module:
        cognitive_states['delay_module'] = {
            'delay_steps': self.cognitive_delay_module.delay_steps,
            'buffer_state': self.cognitive_delay_module.get_status()
        }
    
    if self.cognitive_perception_module:
        cognitive_states['perception_module'] = {
            'noise_config': self.cognitive_perception_module.noise_config,
            'initialized': True
        }
    
    checkpoint['cognitive_states'] = cognitive_states
```

### 6. 多环境并行支持
```python
# 改进collect_rollouts中的认知模块处理
for env_idx in range(self.args.n_envs):
    # 获取每个环境实例
    if hasattr(self.envs, 'envs'):
        current_env = self.envs.envs[env_idx]
    elif hasattr(self.envs, 'venv'):
        current_env = self.envs.venv.envs[env_idx]
    else:
        current_env = None
    
    if current_env and self.use_cognitive_modules:
        # 每个环境单独处理认知模块
        # ...
```

## 测试验证清单

- [ ] 认知模块正确初始化
- [ ] 感知模块与偏差模块联动工作
- [ ] 环境附加/分离机制正常
- [ ] 多环境并行时认知模块独立工作
- [ ] TensorBoard正确记录认知指标
- [ ] 检查点保存/恢复认知状态
- [ ] 认知可视化生成正常
- [ ] 训练稳定性不受影响

## 性能影响评估

### 预期影响
1. **计算开销**: 约增加5-10%的计算时间
2. **内存占用**: 每个环境增加约10MB内存
3. **训练收敛**: 可能需要更多训练步数达到同等性能
4. **探索能力**: 认知噪声可能增强探索

### 优化建议
1. 使用批处理减少认知模块调用开销
2. 在评估时可选择性关闭认知模块
3. 调整认知模块参数平衡真实性和训练效率

## 使用示例

### 基础训练（无认知模块）
```bash
python ppo_expert_reproduction_with_cog.py
```

### 启用所有认知模块
```bash
python ppo_expert_reproduction_with_cog.py \
    --use_cognitive_modules \
    --use_cognitive_bias \
    --use_cognitive_delay \
    --use_cognitive_perception \
    --cognitive_verbose
```

### 仅启用部分认知模块
```bash
python ppo_expert_reproduction_with_cog.py \
    --use_cognitive_modules \
    --use_cognitive_delay \
    --delay_steps 3
```

## 未来改进方向

1. **认知模块自适应**: 根据训练进度自动调整认知参数
2. **认知课程学习**: 逐步增加认知难度
3. **认知对抗训练**: 使用认知模块提升鲁棒性
4. **认知迁移学习**: 将认知特性迁移到新环境

## 参考文献
- 原始仿真脚本: `ppo_checkpoint_simulation_with_cog.py`
- 认知模块实现: `cognitive_module/`目录
- MetaDrive文档: [MetaDrive官方文档](https://metadrive-simulator.readthedocs.io/) 