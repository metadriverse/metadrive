# 认知模块Mask机制说明文档

## 概述

认知模块Mask机制是对原有认知参数集成方案的重要改进，解决了子模块开关控制不精确的问题。通过引入mask向量，实现了对认知参数在网络中作用的精确控制。

## 问题背景

### 原有方案的问题
在之前的实现中，存在以下问题：
- 只要启用 `--use_cognitive_modules`，所有4个认知参数都会被添加到网络输入中
- 即使某个子模块被禁用（如 `--use_cognitive_bias=False`），其对应的参数仍然会影响网络训练
- 导致单模块实验不纯净，包含了其他模块的参数噪声

### 解决方案
引入**参数级Mask机制**：
- 输入维度：275（原始） + 4（认知参数） + 4（mask） = **283维**
- mask为0时，对应参数被网络忽略；mask为1时，参数有效
- 通过mask向量精确控制每个认知参数的作用

## 技术实现

### 1. 维度结构
```
原始观测: [275维] - Lidar(240) + State(35)
认知参数: [4维]   - [bias_coef, sigma0, k, delay]
认知mask: [4维]   - [bias_mask, sigma0_mask, k_mask, delay_mask]
最终输入: [283维] - 原始观测 + 认知参数 + 认知mask
```

### 2. Mask生成规则
```python
# 根据模块开关状态生成mask
bias_mask = 1.0 if args.use_cognitive_bias else 0.0
perception_sigma0_mask = 1.0 if args.use_cognitive_perception else 0.0
perception_k_mask = 1.0 if args.use_cognitive_perception else 0.0
delay_mask = 1.0 if args.use_cognitive_delay else 0.0
```

### 3. 核心函数：`_concatenate_cognitive_params`
```python
def _concatenate_cognitive_params(self, obs, cognitive_params):
    """
    将认知参数和对应的mask拼接到观测向量中
    
    Args:
        obs: 原始观测 [n_envs, 275]
        cognitive_params: 认知参数字典
    
    Returns:
        扩展后的观测 [n_envs, 283]
    """
    # 提取认知参数值
    bias_coef = cognitive_params.get('bias_inverse_tta_coef', 1.0)
    sigma0 = cognitive_params.get('perception_sigma0', 0.1)
    k = cognitive_params.get('perception_k', 0.02)
    delay = cognitive_params.get('delay_steps', 2)
    
    # 生成认知参数mask
    bias_mask = 1.0 if self.args.use_cognitive_bias else 0.0
    perception_sigma0_mask = 1.0 if self.args.use_cognitive_perception else 0.0
    perception_k_mask = 1.0 if self.args.use_cognitive_perception else 0.0
    delay_mask = 1.0 if self.args.use_cognitive_delay else 0.0
    
    # 构建向量并拼接
    cognitive_vector = np.array([[bias_coef, sigma0, k, delay] for _ in range(obs_np.shape[0])])
    cognitive_mask = np.array([[bias_mask, perception_sigma0_mask, perception_k_mask, delay_mask] for _ in range(obs_np.shape[0])])
    obs_with_cognitive = np.concatenate([obs_np, cognitive_vector, cognitive_mask], axis=1)
    
    return obs_with_cognitive
```

## 使用示例

### 1. 启用所有认知模块
```bash
python ppo_expert_reproduction_with_cog.py \
    --use_cognitive_modules \
    --use_cognitive_bias \
    --use_cognitive_delay \
    --use_cognitive_perception
```
**结果：** mask = [1.0, 1.0, 1.0, 1.0] - 所有参数有效

### 2. 仅启用偏差模块
```bash
python ppo_expert_reproduction_with_cog.py \
    --use_cognitive_modules \
    --use_cognitive_bias
```
**结果：** mask = [1.0, 0.0, 0.0, 0.0] - 仅偏差参数有效

### 3. 启用感知+延迟模块
```bash
python ppo_expert_reproduction_with_cog.py \
    --use_cognitive_modules \
    --use_cognitive_perception \
    --use_cognitive_delay
```
**结果：** mask = [0.0, 1.0, 1.0, 1.0] - 感知和延迟参数有效

## 验证测试

运行测试脚本验证mask机制：
```bash
python test_cognitive_mask.py
```

测试涵盖8种模块组合，验证：
- ✅ 维度正确性（283维）
- ✅ Mask值正确性
- ✅ 网络兼容性

## 监控与可视化

### TensorBoard监控
mask状态会被记录到TensorBoard：
```
cognitive_mask/bias_module: 1.0 或 0.0
cognitive_mask/perception_module: 1.0 或 0.0  
cognitive_mask/delay_module: 1.0 或 0.0
cognitive_mask/active_modules_count: 0-3
```

### 控制台输出
训练开始时会显示mask状态：
```
🎭 认知模块Mask状态:
   偏差模块: ✅启用 (mask=1.0)
   感知模块: ❌禁用 (mask=0.0)
   延迟模块: ✅启用 (mask=1.0)
   活跃模块数: 2/3
   输入维度: 275(原始) + 4(认知参数) + 4(mask) = 283
```

## 兼容性

### 检查点加载
支持多种维度的检查点迁移：
- 275 → 283：扩展权重（新增认知参数+mask）
- 279 → 283：扩展权重（增加mask维度）
- 283 → 275：截取权重（移除认知参数+mask）

### 渐进式训练
mask机制与渐进式训练兼容，支持从275维检查点恢复到283维网络。

## 技术优势

### 1. 精确控制
- 每个认知参数都有独立的mask控制
- 禁用模块的参数不会影响网络训练
- 保证单模块实验的纯净性

### 2. 统一架构
- 所有配置都使用相同的283维网络结构
- 避免了动态维度带来的复杂性
- 便于不同配置间的比较

### 3. 可解释性
- Mask值明确显示哪些参数在起作用
- 便于分析参数重要性
- 支持消融实验

### 4. 向后兼容
- 支持从旧版本检查点迁移
- 不破坏现有训练流程
- 可选择性启用

## 最佳实践

### 1. 单模块实验
进行单一认知模块的效果研究时，确保只启用目标模块：
```bash
# 研究偏差模块效果
python ppo_expert_reproduction_with_cog.py \
    --use_cognitive_modules \
    --use_cognitive_bias
```

### 2. 消融实验
通过不同mask组合进行消融实验：
```bash
# 基线：无认知模块
python ppo_expert_reproduction_with_cog.py

# 对照1：仅偏差
python ppo_expert_reproduction_with_cog.py --use_cognitive_modules --use_cognitive_bias

# 对照2：偏差+感知
python ppo_expert_reproduction_with_cog.py --use_cognitive_modules --use_cognitive_bias --use_cognitive_perception

# 完整：所有模块
python ppo_expert_reproduction_with_cog.py --use_cognitive_modules --use_cognitive_bias --use_cognitive_perception --use_cognitive_delay
```

### 3. 监控建议
- 使用TensorBoard监控mask状态变化
- 记录每次实验的mask配置
- 对比不同mask组合的性能差异

## 总结

认知模块Mask机制成功解决了原有方案中子模块控制不精确的问题，通过引入283维输入（275+4+4），实现了对认知参数的精确控制。该机制具有以下特点：

1. **精确控制**：每个认知参数都有独立的mask
2. **架构统一**：所有配置使用相同的网络结构  
3. **实验纯净**：禁用模块不会影响网络训练
4. **易于监控**：mask状态可视化和日志记录
5. **向后兼容**：支持旧版本检查点迁移

这为进行精确的认知模块研究和消融实验提供了可靠的技术基础。 