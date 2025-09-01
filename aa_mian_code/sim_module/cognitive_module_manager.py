"""
认知模块管理器

该模块专门负责认知模块的初始化、配置、生命周期管理和协调。
从原始PPOCheckpointSimulator中提取认知模块相关功能。
"""

import argparse
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path
import sys

# 添加认知模块路径
current_dir = Path(__file__).parent.absolute()
cognitive_module_path = current_dir.parent.parent / "cognitive_module"
sys.path.insert(0, str(cognitive_module_path))

# 导入认知模块
from cognitive_module.cognitive_bias_module import CognitiveBiasModule
from cognitive_module.cognitive_delay_module import CognitiveDelayModule
from cognitive_module.cognitive_perception_module import CognitivePerceptionModule


class CognitiveModuleManager:
    """认知模块管理器
    
    专门负责：
    1. 认知模块的初始化和配置
    2. 认知模块的生命周期管理（附加/分离环境）
    3. 观测、动作、奖励的认知处理流水线
    4. 认知参数的拼接和管理
    """
    
    def __init__(self, args: Optional[argparse.Namespace] = None):
        """
        初始化认知模块管理器
        
        Args:
            args: 命令行参数，包含认知模块配置
        """
        self.args = args
        self.use_cognitive_modules = args and getattr(args, 'use_cognitive_modules', False)
        
        # 认知模块实例
        self.cognitive_bias_module = None
        self.cognitive_delay_module = None
        self.cognitive_perception_module = None
        
        # 认知参数相关
        self._cognitive_params_logged = False
        
        if self.use_cognitive_modules:
            print("🧠 认知模块管理器: 开始初始化认知模块...")
            self._initialize_modules()
        else:
            print("🧠 认知模块管理器: 认知模块未启用")
    
    def _initialize_modules(self):
        """初始化所有启用的认知模块"""
        # 先初始化认知感知模块（因为认知偏差模块需要引用它）
        if self.args and getattr(self.args, 'use_cognitive_perception', False):
            self._initialize_perception_module()
        
        # 初始化认知偏差模块（传入认知感知模块引用）
        if self.args and getattr(self.args, 'use_cognitive_bias', False):
            self._initialize_bias_module()
        
        # 初始化认知延迟模块
        if self.args and getattr(self.args, 'use_cognitive_delay', False):
            self._initialize_delay_module()
    
    def _initialize_perception_module(self):
        """初始化认知感知模块"""
        perception_config = {
            'sigma0': getattr(self.args, 'perception_sigma0', 0.1),  # 基础噪声（米）
            'sigma_max': getattr(self.args, 'perception_sigma_max', 0.8),  # 最大噪声（米）
            'p_miss0': getattr(self.args, 'perception_p_miss0', 0.0),
            'far_distance': 50.0,
            'p_false': getattr(self.args, 'perception_p_false', 0.0),
            'use_ar1': True,  # 启用AR(1)过程
            'rho': 0.8,       # AR(1)相关系数
            'use_kf': getattr(self.args, 'perception_use_kf', True),
            'kf_dt': getattr(self.args, 'perception_kf_dt', 0.1),
            'kf_q_scale': getattr(self.args, 'perception_kf_q_scale', 100.0)
        }
        self.cognitive_perception_module = CognitivePerceptionModule(noise_config=perception_config)
        
        # 启用雷达束可视化（如果指定）
        if getattr(self.args, 'enable_radar_beam_viz', False):
            self.cognitive_perception_module.enable_radar_visualization(True)
        
        print(f"   ✅ 认知感知模块已启用")
        print(f"      基准噪声σ0: {perception_config['sigma0']:.3f}")
        print(f"      最大噪声σ_max: {perception_config['sigma_max']:.3f}")
        print(f"      卡尔曼滤波: {'启用' if perception_config['use_kf'] else '禁用'}")
    
    def _initialize_bias_module(self):
        """初始化认知偏差模块"""
        bias_config = {
            'inverse_tta_coef': getattr(self.args, 'bias_inverse_tta_coef', 1.5),
            'tta_threshold': getattr(self.args, 'bias_tta_threshold', 0.1),
            'visual_detection_distance': getattr(self.args, 'bias_visual_distance', 50.0),
            'verbose': True
        }
        # 传入认知感知模块引用
        self.cognitive_bias_module = CognitiveBiasModule(
            bias_config=bias_config,
            cognitive_perception_module=self.cognitive_perception_module
        )
        print(f"   ✅ 认知偏差模块已启用")
        print(f"      TTA逆系数: {bias_config['inverse_tta_coef']:.2f}")
        print(f"      TTA阈值: {bias_config['tta_threshold']:.3f}")
        if self.cognitive_perception_module:
            print(f"      🔗 已连接到认知感知模块")
    
    def _initialize_delay_module(self):
        """初始化认知延迟模块"""
        delay_steps = int(getattr(self.args, 'delay_steps', 2))  # 确保是整数类型
        self.cognitive_delay_module = CognitiveDelayModule(
            delay_steps=delay_steps,
            enable_smoothing=False,
            smoothing_factor=0.3,
            enable_visualization=True
        )
        print(f"   ✅ 认知延迟模块已启用 (延迟{delay_steps}步)")
    
    def attach_to_env(self, env) -> bool:
        """
        将认知模块附加到环境
        
        Args:
            env: MetaDrive环境实例
            
        Returns:
            是否成功附加
        """
        if not self.use_cognitive_modules:
            return False
        
        success_count = 0
        total_modules = 0
        
        # 附加认知感知模块
        if self.cognitive_perception_module:
            total_modules += 1
            self.cognitive_perception_module.attach_to_env(env)
            print("🔗 认知感知模块已附加到环境 - 噪声将在传感器层自动注入")
            
            # 验证环境配置，确保避免双重噪声
            lidar_config = env.config.get("vehicle_config", {}).get("lidar", {})
            gaussian_noise = lidar_config.get("gaussian_noise", 0.0)
            dropout_prob = lidar_config.get("dropout_prob", 0.0)
            
            if gaussian_noise > 0.0 or dropout_prob > 0.0:
                print(f"⚠️ 警告：检测到环境lidar配置中存在额外噪声！")
                print(f"   gaussian_noise: {gaussian_noise}")
                print(f"   dropout_prob: {dropout_prob}")
                print(f"   这可能导致双重噪声问题，建议设置为0.0")
            else:
                print(f"✅ 环境lidar噪声配置正确 (gaussian_noise=0.0, dropout_prob=0.0)")
            
            success_count += 1
        
        # 附加认知偏差模块
        if self.cognitive_bias_module:
            total_modules += 1
            success = self.cognitive_bias_module.attach_to_env(env)
            if success:
                print("🔗 认知偏差模块已附加到环境 - 将基于TTA动态调整奖励")
                success_count += 1
            else:
                print("⚠️ 认知偏差模块附加失败")
        
        return success_count == total_modules
    
    def detach_from_env(self):
        """从环境分离认知模块"""
        if not self.use_cognitive_modules:
            return
        
        if self.cognitive_perception_module:
            try:
                self.cognitive_perception_module.detach_from_env()
                print("🔗 认知感知模块已从环境分离")
            except Exception as e:
                print(f"⚠️ 认知感知模块分离失败: {e}")
        
        if self.cognitive_bias_module:
            try:
                self.cognitive_bias_module.detach_from_env()
                print("🔗 认知偏差模块已从环境分离")
            except Exception as e:
                print(f"⚠️ 认知偏差模块分离失败: {e}")
    
    def reset_modules(self):
        """重置所有认知模块状态"""
        if not self.use_cognitive_modules:
            return
        
        if self.cognitive_bias_module:
            self.cognitive_bias_module.reset()
        if self.cognitive_delay_module:
            self.cognitive_delay_module.reset()
        if self.cognitive_perception_module:
            self.cognitive_perception_module.reset()
        
        print("🔄 所有认知模块状态已重置")
    
    def process_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        处理观测（认知感知模块在传感器层已自动处理噪声）
        主要负责拼接认知参数到观测向量
        
        Args:
            obs: 原始观测 [275维]
            
        Returns:
            处理后的观测 [275维或283维]
        """
        if not self.use_cognitive_modules:
            return obs
        
        # 认知感知模块的噪声已在传感器层自动注入
        # 这里主要负责拼接认知参数（如果网络需要283维）
        return self._concatenate_cognitive_params(obs)
    
    def process_action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        处理动作（认知延迟模块）
        
        Args:
            action: 原始动作
            
        Returns:
            tuple: (处理后的动作, 是否应用了延迟)
        """
        if not self.use_cognitive_modules or not self.cognitive_delay_module:
            return action, False
        
        try:
            delayed_action = self.cognitive_delay_module.process_action(action, is_ppo_mode=True)
            return delayed_action, True
        except Exception as e:
            print(f"⚠️ 认知延迟模块处理失败: {e}")
            return action, False
    
    def process_reward(self, reward: float, env, info: dict) -> Tuple[float, bool, Dict]:
        """
        处理奖励（认知偏差模块）
        
        Args:
            reward: 原始奖励
            env: 环境实例
            info: 环境信息
            
        Returns:
            tuple: (处理后的奖励, 是否应用了偏差, 偏差信息)
        """
        if not self.use_cognitive_modules or not self.cognitive_bias_module:
            return reward, False, {}
        
        try:
            # 根据文档使用正确的参数调用 process_reward
            if hasattr(self.cognitive_bias_module, 'process_reward'):
                reward_result = self.cognitive_bias_module.process_reward(
                    original_reward=reward,
                    env=env,
                    info=info,
                    is_ppo_mode=True
                )
                
                # 处理返回值 - 根据文档是 (adjusted_reward, bias_info)
                if isinstance(reward_result, (tuple, list)) and len(reward_result) >= 2:
                    adjusted_reward, bias_info = reward_result[0], reward_result[1]
                    adjusted_reward = float(adjusted_reward)
                    
                    # 检查偏差是否真正活跃
                    if isinstance(bias_info, dict):
                        bias_amount = bias_info.get('bias_applied', 0.0)
                        bias_active = bias_info.get('bias_active', False)
                        
                        if bias_active and abs(bias_amount) > 1e-6:
                            return adjusted_reward, True, bias_info
                        else:
                            return reward, False, bias_info
                    else:
                        return adjusted_reward, True, {'bias_info': bias_info}
                else:
                    # 兼容性处理 - 单一返回值
                    adjusted_reward = float(reward_result) if reward_result is not None else reward
                    bias_applied = abs(adjusted_reward - reward) > 1e-6
                    return adjusted_reward, bias_applied, {}
            else:
                # 如果没有process_reward方法，应用简单偏差
                if reward > 0:
                    adjusted_reward = reward * 0.9  # 轻微减少正奖励
                else:
                    adjusted_reward = reward * 1.1  # 增加负奖励影响
                
                return adjusted_reward, True, {'simple_bias': True}
        except Exception as e:
            print(f"⚠️ 认知偏差模块处理失败: {e}")
            return reward, False, {}
    
    def _concatenate_cognitive_params(self, obs: np.ndarray) -> np.ndarray:
        """
        将认知参数和对应的mask拼接到观测向量中
        
        Args:
            obs: 原始观测 [275维]
            
        Returns:
            扩展后的观测 [283维] - 275(原始) + 4(认知参数) + 4(mask)
        """
        # 确保obs是275维度
        if obs.shape[-1] != 275:
            # 如果已经是283维度，直接返回
            if obs.shape[-1] == 283:
                return obs
            raise ValueError(f"原始观测维度应为275，实际为{obs.shape[-1]}")
        
        # 获取认知参数值（使用默认值或从命令行参数获取）
        bias_coef = getattr(self.args, 'bias_inverse_tta_coef', 1.5) if self.args else 1.5
        sigma0 = getattr(self.args, 'perception_sigma0', 0.1) if self.args else 0.1
        sigma_max = getattr(self.args, 'perception_sigma_max', 0.8) if self.args else 0.8
        delay = getattr(self.args, 'delay_steps', 2) if self.args else 2
        
        # 生成认知参数mask（根据命令行开关状态）
        bias_mask = 1.0 if (self.args and getattr(self.args, 'use_cognitive_bias', False)) else 0.0
        perception_sigma0_mask = 1.0 if (self.args and getattr(self.args, 'use_cognitive_perception', False)) else 0.0
        perception_sigma_max_mask = 1.0 if (self.args and getattr(self.args, 'use_cognitive_perception', False)) else 0.0
        delay_mask = 1.0 if (self.args and getattr(self.args, 'use_cognitive_delay', False)) else 0.0
        
        # 构建认知参数向量 [bias_coef, sigma0, sigma_max, delay]
        cognitive_params = np.array([bias_coef, sigma0, sigma_max, delay], dtype=np.float32)
        
        # 构建认知参数mask向量 [bias_mask, sigma0_mask, sigma_max_mask, delay_mask]
        cognitive_mask = np.array([bias_mask, perception_sigma0_mask, perception_sigma_max_mask, delay_mask], dtype=np.float32)
        
        # 拼接：原始观测 + 认知参数 + 认知mask
        obs_with_cognitive = np.concatenate([obs, cognitive_params, cognitive_mask], axis=-1)
        
        # 只在第一次调用时打印详细信息
        if not self._cognitive_params_logged:
            print(f"🧠 认知参数拼接: {obs.shape} → {obs_with_cognitive.shape}")
            print(f"   参数: [bias={bias_coef:.2f}, σ0={sigma0:.3f}, σ_max={sigma_max:.3f}, delay={delay}]")
            print(f"   mask: [bias={bias_mask}, σ0={perception_sigma0_mask}, σ_max={perception_sigma_max_mask}, delay={delay_mask}]")
            self._cognitive_params_logged = True
        
        return obs_with_cognitive
    
    def generate_visualizations(self, save_dir: str, env=None):
        """生成认知模块可视化"""
        if not self.use_cognitive_modules:
            return
        
        # 生成认知感知模块可视化
        if self.cognitive_perception_module:
            try:
                self.cognitive_perception_module.generate_visualization(save_dir=save_dir, env=env)
                print("📊 认知感知模块可视化已生成")
            except Exception as e:
                print(f"⚠️ 认知感知模块可视化生成失败: {e}")
        
        # 生成认知偏差模块可视化
        if self.cognitive_bias_module:
            try:
                self.cognitive_bias_module.generate_visualization(env=env, save_dir=save_dir)
                print("📊 认知偏差模块可视化已生成")
            except Exception as e:
                print(f"⚠️ 认知偏差模块可视化生成失败: {e}")
    
    def get_statistics(self) -> Dict:
        """获取认知模块统计信息"""
        stats = {
            'use_cognitive_modules': self.use_cognitive_modules,
            'modules_enabled': {}
        }
        
        if not self.use_cognitive_modules:
            return stats
        
        # 认知偏差统计
        if self.cognitive_bias_module:
            try:
                if hasattr(self.cognitive_bias_module, 'get_statistics'):
                    stats['modules_enabled']['bias'] = self.cognitive_bias_module.get_statistics()
                else:
                    stats['modules_enabled']['bias'] = {'status': 'active'}
            except Exception as e:
                stats['modules_enabled']['bias'] = {'error': str(e)}
        
        # 认知延迟统计
        if self.cognitive_delay_module:
            try:
                if hasattr(self.cognitive_delay_module, 'get_statistics'):
                    stats['modules_enabled']['delay'] = self.cognitive_delay_module.get_statistics()
                else:
                    delay_steps = getattr(self.cognitive_delay_module, 'delay_steps', 0)
                    stats['modules_enabled']['delay'] = {'delay_steps': delay_steps, 'status': 'active'}
            except Exception as e:
                stats['modules_enabled']['delay'] = {'error': str(e)}
        
        # 认知感知统计
        if self.cognitive_perception_module:
            try:
                if hasattr(self.cognitive_perception_module, 'get_statistics'):
                    stats['modules_enabled']['perception'] = self.cognitive_perception_module.get_statistics()
                else:
                    noise_config = getattr(self.cognitive_perception_module, 'noise_config', {})
                    stats['modules_enabled']['perception'] = {'noise_config': noise_config, 'status': 'active'}
            except Exception as e:
                stats['modules_enabled']['perception'] = {'error': str(e)}
        
        return stats
    
    def get_closest_beam_info(self) -> Dict:
        """获取最近雷达束信息（用于可视化）"""
        if self.cognitive_perception_module and hasattr(self.cognitive_perception_module, 'get_closest_beam_info'):
            return self.cognitive_perception_module.get_closest_beam_info()
        
        return {'original_distance': 0.0, 'noisy_distance': 0.0, 'noise_level': 0.0, 'beam_index': 0}
    
    def get_bias_info(self) -> Dict:
        """获取偏差信息（用于可视化）"""
        if self.cognitive_bias_module and hasattr(self.cognitive_bias_module, 'get_bias_info'):
            return self.cognitive_bias_module.get_bias_info()
        
        return {'bias_strength': 0.0, 'bias_active': False}
    
    def get_delay_info(self) -> Dict:
        """获取延迟信息（用于可视化）"""
        if self.cognitive_delay_module:
            if hasattr(self.cognitive_delay_module, 'get_delay_info'):
                return self.cognitive_delay_module.get_delay_info()
            else:
                return {'current_delay': getattr(self.cognitive_delay_module, 'delay_steps', 0)}
        
        return {'current_delay': 0} 