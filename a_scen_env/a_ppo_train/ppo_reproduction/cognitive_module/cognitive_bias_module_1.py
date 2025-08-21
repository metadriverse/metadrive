#!/usr/bin/env python3
"""
认知偏差模块 - 独立实现，不依赖gym环境
基于TTA（Time-To-Arrival）的视觉厌恶风险偏差调整

新增功能：
- 视觉厌恶：基于主车车头方向30°范围内50m距离内的物体检测
- 详细的检测历史记录和可视化
- 支持可重现性控制
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class CognitiveBiasModule:
    """
    认知偏差模块 - 基于TTA的风险厌恶偏差调整
    
    支持视觉厌恶功能和可重现性控制
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化认知偏差模块
        
        Args:
            config: 配置参数字典
        """
        self.config = self._get_default_config()
        self.config.update(config)
        
        # 基础偏差参数
        self.inverse_tta_coef = self.config['inverse_tta_coef']
        self.tta_threshold = self.config['tta_threshold']
        self.adaptive_bias = self.config['adaptive_bias']
        self.adaptation_rate = self.config['adaptation_rate']
        self.min_adaptive_factor = self.config['min_adaptive_factor']
        self.max_adaptive_factor = self.config['max_adaptive_factor']
        self.verbose = self.config['verbose']
        
        # 视觉厌恶参数
        self.visual_detection_distance = self.config['visual_detection_distance']
        self.visual_detection_angle = self.config['visual_detection_angle']
        self.visual_aversion_strength = self.config['visual_aversion_strength']
        
        # 可重现性参数（新增）
        self.random_seed = self.config.get('random_seed', None)
        self.deterministic_mode = self.config.get('deterministic_mode', False)
        self._random_state = None  # 独立的随机状态
        
        # 初始化随机状态
        self._initialize_random_state()
        
        # 历史记录
        regular_history_length = self.config['history_length']
        extended_history_length = self.config['extended_history_length']
        
        self.bias_history = deque(maxlen=regular_history_length)
        self.reward_history = deque(maxlen=regular_history_length)
        self.tta_history = deque(maxlen=regular_history_length)
        self.adaptive_history = deque(maxlen=regular_history_length)
        
        # 视觉厌恶检测历史记录
        # 修复：增加历史记录长度以确保保留完整的数据，包括reset瞬间的27.8 m/s记录
        self.detection_history = deque(maxlen=extended_history_length)  # 检测到的物体信息
        self.distance_history = deque(maxlen=extended_history_length)   # 最近威胁距离历史
        self.threat_count_history = deque(maxlen=extended_history_length)  # 威胁物体数量历史
        
        # 状态变量
        self.adaptive_factor = 1.0
        self._step_count = 0
        self._total_bias = 0.0
        self._active_steps = 0
        self._attached_env = None
        
        logger.info(f"CognitiveBiasModule初始化完成，视觉厌恶启用")
        logger.info(f"检测参数: 距离={self.visual_detection_distance}m, 角度=±{self.visual_detection_angle}°, 强度={self.visual_aversion_strength}")
        
        if self.deterministic_mode:
            logger.info(f"✅ 确定性模式已启用，随机种子: {self.random_seed}")
    
    def _initialize_random_state(self):
        """初始化随机状态以确保可重现性"""
        if self.random_seed is not None:
            # 创建独立的随机状态
            self._random_state = np.random.RandomState(self.random_seed)
            logger.info(f"认知偏差模块随机种子设置: {self.random_seed}")
        else:
            self._random_state = np.random.RandomState()
    
    def set_random_seed(self, seed: int):
        """
        设置随机种子（可重现性接口）
        
        Args:
            seed: 随机种子
        """
        self.random_seed = seed
        self._initialize_random_state()
        logger.info(f"认知偏差模块随机种子更新: {seed}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'inverse_tta_coef': 1.0,      # 偏差强度系数
            'tta_threshold': 1.0,          # TTA阈值（降低以便更容易触发视觉厌恶）
            'adaptive_bias': True,         # 启用自适应偏差
            'adaptation_rate': 0.01,       # 自适应速率
            'min_adaptive_factor': 0.5,    # 最小自适应因子
            'max_adaptive_factor': 2.0,    # 最大自适应因子
            'history_length': 100,         # 历史记录长度（用于自适应计算）
            'extended_history_length': 1000,  # 扩展历史记录长度（用于可视化数据存储）
            'verbose': False,              # 详细日志输出
            
            # 视觉厌恶参数
            'visual_detection_distance': 50.0,  # 视觉检测距离（米）
            'visual_detection_angle': 30.0,     # 视觉检测角度（度）
            'visual_aversion_strength': 0.5,    # 视觉厌恶增强强度
            
            # 可重现性参数（新增）
            'random_seed': None,           # 随机种子
            'deterministic_mode': False    # 确定性模式
        }
    
    def attach_to_env(self, env) -> bool:
        """
        附加到环境
        
        Args:
            env: 环境实例
            
        Returns:
            bool: 是否成功附加
        """
        try:
            self._attached_env = env
            logger.info("认知偏差模块已附加到环境")
            return True
        except Exception as e:
            logger.error(f"认知偏差模块附加失败: {e}")
            return False
    
    def detach_from_env(self):
        """从环境分离"""
        self._attached_env = None
        logger.info("认知偏差模块已从环境分离")
    
    def reset(self):
        """重置模块状态"""
        self.bias_history.clear()
        self.reward_history.clear()
        self.tta_history.clear()
        self.adaptive_history.clear()
        
        # 清空新的历史记录
        self.detection_history.clear()
        self.distance_history.clear()
        self.threat_count_history.clear()
        
        self.adaptive_factor = 1.0
        self._step_count = 0
        self._total_bias = 0.0
        self._active_steps = 0
        
        # 重置随机状态以确保可重现性
        if self.deterministic_mode and self.random_seed is not None:
            self._initialize_random_state()
        
        if self.verbose:
            logger.info("认知偏差模块已重置")
    
    def get_inverse_tta(self, env, info: Optional[Dict] = None) -> Optional[float]:
        """
        获取inverse_tta值
        
        Args:
            env: 环境实例
            info: 信息字典
            
        Returns:
            inverse_tta值，如果无法获取则返回None
        """
        # 尝试多种方式获取inverse_tta
        if hasattr(env, 'current_inverse_tta'):
            return env.current_inverse_tta
        elif hasattr(env, 'agent') and hasattr(env.agent, 'inverse_tta'):
            return env.agent.inverse_tta
        elif info and 'inverse_tta' in info:
            return info['inverse_tta']
        else:
            # 手动计算inverse_tta（基于视觉厌恶）
            return self._compute_inverse_tta_manually(env)
    
    def _compute_inverse_tta_manually(self, env) -> Optional[float]:
        """
        手动计算inverse_tta（基于视觉厌恶）
        检测主车车头方向30°范围内的50m距离内的物体，应用视觉厌恶
        """
        try:
            agent = env.agent
            agent_pos = agent.position
            agent_heading = agent.heading_theta  # 主车朝向角度（弧度）
            agent_speed = agent.speed
            
            # 视觉厌恶参数
            max_detection_distance = self.visual_detection_distance  # 最大检测距离
            detection_angle = np.radians(self.visual_detection_angle)  # 车头方向检测角度范围（转换为弧度）
            
            min_threat_distance = float('inf')
            threat_detected = False
            threat_count = 0
            detected_objects = []  # 记录检测到的物体信息
            
            # 方法1：从雷达数据中检测威胁物体
            # MetaDrive的雷达返回距离数组，需要转换为物体检测
            if hasattr(agent, 'lidar'):
                try:
                    # 获取雷达观测数据（从观测向量中提取）
                    from metadrive.obs.observation_base import ObservationBase
                    if hasattr(env, 'engine') and hasattr(env.engine, 'get_sensor'):
                        lidar_sensor = env.engine.get_sensor('lidar')
                        if lidar_sensor and hasattr(agent, 'vehicle'):
                            # 直接从传感器获取距离数据
                            lidar_result, _ = lidar_sensor.perceive(
                                agent, env.engine.physics_world, 
                                num_lasers=240, distance=50.0
                            )
                            
                            # 转换雷达数据为物体检测
                            if lidar_result and len(lidar_result) >= 240:
                                # 240个激光射线，每1.5度一个 (360度/240 = 1.5度)
                                angle_step = 2 * np.pi / 240  # 每个射线的角度间隔
                                
                                for i, normalized_distance in enumerate(lidar_result):
                                    if normalized_distance < 1.0:  # 检测到物体 (归一化距离 < 1.0)
                                        actual_distance = normalized_distance * 50.0  # 转换为实际距离
                                        
                                        if actual_distance <= max_detection_distance and actual_distance > 0.5:
                                            # 计算射线角度（相对于主车坐标系）
                                            ray_angle = i * angle_step  # 射线在雷达坐标系中的角度
                                            # 转换为世界坐标系角度
                                            world_angle = agent_heading + ray_angle
                                            
                                            # 计算与车头方向的角度差
                                            forward_angle_diff = min(ray_angle, 2*np.pi - ray_angle)
                                            
                                            if forward_angle_diff <= detection_angle:
                                                min_threat_distance = min(min_threat_distance, actual_distance)
                                                threat_detected = True
                                                threat_count += 1
                                                detected_objects.append({
                                                    'source': 'lidar',
                                                    'distance': actual_distance,
                                                    'ray_index': i,
                                                    'ray_angle': np.degrees(ray_angle),
                                                    'world_angle': np.degrees(world_angle),
                                                    'angle_diff': np.degrees(forward_angle_diff)
                                                })
                except Exception as e:
                    if self.verbose:
                        logger.debug(f"雷达检测失败: {e}")
            
            # 方法2：检查环境中的背景车辆
            if hasattr(env, 'ghost_vehicles') and env.ghost_vehicles:
                for vehicle_id, vehicle in env.ghost_vehicles.items():
                    if hasattr(vehicle, 'position'):
                        v_pos = vehicle.position
                        
                        # 计算相对位置
                        dx = v_pos[0] - agent_pos[0]
                        dy = v_pos[1] - agent_pos[1]
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        # 检查距离是否在50m内
                        if distance <= max_detection_distance and distance > 0:
                            # 计算相对角度
                            relative_angle = np.arctan2(dy, dx)
                            
                            # 计算与主车朝向的角度差
                            angle_diff = abs(relative_angle - agent_heading)
                            # 处理角度环绕问题
                            if angle_diff > np.pi:
                                angle_diff = 2 * np.pi - angle_diff
                            
                            # 检查是否在车头方向±30度内
                            if angle_diff <= detection_angle:
                                min_threat_distance = min(min_threat_distance, distance)
                                threat_detected = True
                                threat_count += 1
                                detected_objects.append({
                                    'source': 'ghost_vehicle',
                                    'vehicle_id': vehicle_id,
                                    'distance': distance,
                                    'angle': relative_angle,
                                    'angle_diff': np.degrees(angle_diff),
                                    'position': v_pos
                                })
            
            # 计算视觉厌恶强度（inverse_tta）
            if threat_detected and min_threat_distance < float('inf'):
                if agent_speed > 0.1:  # 避免除零错误
                    # 基于距离和速度计算威胁度
                    basic_inverse_tta = agent_speed / min_threat_distance
                    
                    # 应用视觉厌恶增强因子
                    # 距离越近，视觉厌恶越强
                    distance_factor = max_detection_distance / min_threat_distance
                    visual_aversion_factor = 1.0 + (distance_factor - 1.0) * self.visual_aversion_strength  # 视觉厌恶增强
                    
                    inverse_tta = basic_inverse_tta * visual_aversion_factor
                    
                    # 在确定性模式下，添加可控的"随机"因子以模拟认知不确定性
                    if self.deterministic_mode and self._random_state is not None:
                        # 使用确定性的伪随机因子，基于当前状态
                        uncertainty_factor = 1.0 + 0.05 * np.sin(self._step_count * 0.1)  # 周期性变化
                        inverse_tta *= uncertainty_factor
                    
                    # 记录检测信息
                    detection_info = {
                        'threat_count': threat_count,
                        'min_distance': min_threat_distance,
                        'detected_objects': detected_objects,
                        'agent_speed': agent_speed,
                        'visual_aversion_factor': visual_aversion_factor,
                        'basic_inverse_tta': basic_inverse_tta,
                        'final_inverse_tta': inverse_tta
                    }
                    self.detection_history.append(detection_info)
                    self.distance_history.append(min_threat_distance)
                    self.threat_count_history.append(threat_count)
                    
                    if self.verbose:
                        logger.debug(f"视觉威胁检测: 威胁数量={threat_count}, 最近距离={min_threat_distance:.1f}m, "
                                    f"速度={agent_speed:.1f}m/s, 视觉厌恶因子={visual_aversion_factor:.2f}, "
                                    f"基础inverse_tta={basic_inverse_tta:.3f}, 最终inverse_tta={inverse_tta:.3f}")
                        for obj in detected_objects:
                            logger.debug(f"  - {obj['source']}: 距离={obj['distance']:.1f}m, 角度差={obj['angle_diff']:.1f}°")
                    
                    return inverse_tta
            
            # 如果没有检测到威胁，记录空检测信息
            detection_info = {
                'threat_count': 0,
                'min_distance': float('inf'),
                'detected_objects': [],
                'agent_speed': agent_speed,
                'visual_aversion_factor': 1.0,
                'basic_inverse_tta': 0.0,
                'final_inverse_tta': 0.1 if agent_speed > 0 else 0.0
            }
            self.detection_history.append(detection_info)
            self.distance_history.append(float('inf'))
            self.threat_count_history.append(0)
            
            if self.verbose:
                # 提供更详细的调试信息
                total_objects = 0
                if hasattr(env, 'ghost_vehicles'):
                    total_objects = len(env.ghost_vehicles)
                
                logger.debug(f"视觉威胁检测: 未检测到威胁, 速度={agent_speed:.1f}m/s, "
                            f"总背景车数量={total_objects}, 检测距离={max_detection_distance}m, "
                            f"检测角度=±{np.degrees(detection_angle):.1f}°")
            
            # 如果没有检测到威胁，返回低威胁值
            return 0.1 if agent_speed > 0 else 0.0
            
        except Exception as e:
            if self.verbose:
                logger.debug(f"视觉厌恶计算失败: {e}")
        
        return None
    
    def process_reward(self, original_reward: float, env=None, info: Optional[Dict] = None, 
                      is_ppo_mode: bool = False) -> Tuple[float, Dict[str, Any]]:
        """
        处理奖励：基于认知偏差调整
        
        Args:
            original_reward: 原始奖励
            env: 环境实例
            info: 环境info字典
            is_ppo_mode: 是否为PPO模式
            
        Returns:
            (adjusted_reward, bias_info): 调整后的奖励和偏差信息
        """
        # 初始化偏差信息
        bias_info = {
            'original_reward': original_reward,
            'bias_applied': 0.0,
            'inverse_tta': None,
            'adaptive_factor': self.adaptive_factor,
            'bias_active': False
        }
        
        # 仅在PPO模式下应用偏差
        if not is_ppo_mode:
            return original_reward, bias_info
        
        # 获取inverse_tta值
        inverse_tta = self.get_inverse_tta(env, info)
        bias_info['inverse_tta'] = inverse_tta
        
        if inverse_tta is None:
            # 无法获取inverse_tta，不应用偏差
            self.tta_history.append(0.0)
            self.bias_history.append(0.0)
            self.reward_history.append(original_reward)
            return original_reward, bias_info
        
        # 记录TTA历史
        self.tta_history.append(inverse_tta)
        
        # 判断是否应用偏差
        if inverse_tta > self.tta_threshold:
            # 计算偏差量
            bias_factor = self.inverse_tta_coef * inverse_tta
            
            # 应用自适应因子
            if self.adaptive_bias:
                bias_factor *= self.adaptive_factor
            
            # 应用偏差（惩罚高风险行为）
            bias_amount = bias_factor
            adjusted_reward = original_reward - bias_amount
            
            # 记录偏差
            self.bias_history.append(bias_amount)
            bias_info['bias_applied'] = bias_amount
            bias_info['bias_active'] = True
            
            # 更新统计
            self._total_bias += bias_amount
            self._active_steps += 1
            
        else:
            # 不应用偏差
            adjusted_reward = original_reward
            self.bias_history.append(0.0)
        
        # 记录奖励历史
        self.reward_history.append(original_reward)
        
        # 更新自适应因子
        if self.adaptive_bias:
            self._update_adaptive_factor()
        
        # 更新步数
        self._step_count += 1
        
        # 定期日志输出
        if self.verbose and self._step_count % 50 == 0:
            avg_bias = np.mean(list(self.bias_history)) if self.bias_history else 0.0
            avg_reward = np.mean(list(self.reward_history)) if self.reward_history else 0.0
            avg_tta = np.mean(list(self.tta_history)) if self.tta_history else 0.0
            logger.info(f"[CognitiveBias] Step {self._step_count}: "
                       f"avg_bias={avg_bias:.4f}, avg_reward={avg_reward:.4f}, "
                       f"avg_tta={avg_tta:.4f}, active_rate={self._active_steps/self._step_count:.2%}")
        
        bias_info['adjusted_reward'] = adjusted_reward
        return adjusted_reward, bias_info
    
    def _update_adaptive_factor(self):
        """更新自适应偏差因子"""
        if len(self.reward_history) >= 10:
            # 基于最近的奖励变化调整偏差强度
            recent_rewards = list(self.reward_history)[-10:]
            reward_variance = np.var(recent_rewards)
            
            # 基于TTA历史调整
            if self.tta_history:
                recent_tta = list(self.tta_history)[-10:]
                avg_tta = np.mean(recent_tta)
                
                # 如果平均TTA很高，增加偏差；如果很低，减少偏差
                if avg_tta > self.tta_threshold * 1.5:
                    self.adaptive_factor = min(
                        self.max_adaptive_factor,
                        self.adaptive_factor + self.adaptation_rate
                    )
                elif avg_tta < self.tta_threshold * 0.5:
                    self.adaptive_factor = max(
                        self.min_adaptive_factor,
                        self.adaptive_factor - self.adaptation_rate
                    )
            
            # 基于奖励方差调整
            if reward_variance > 1.0:
                self.adaptive_factor = min(
                    self.max_adaptive_factor,
                    self.adaptive_factor + self.adaptation_rate * 0.5
                )
            else:
                self.adaptive_factor = max(
                    self.min_adaptive_factor,
                    self.adaptive_factor - self.adaptation_rate * 0.5
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            包含各种统计数据的字典
        """
        stats = {
            'total_steps': self._step_count,
            'active_steps': self._active_steps,
            'activation_rate': self._active_steps / max(1, self._step_count),
            'total_bias': self._total_bias,
            'average_bias': self._total_bias / max(1, self._step_count),
            'adaptive_factor': self.adaptive_factor
        }
        
        if self.bias_history:
            stats['recent_avg_bias'] = np.mean(list(self.bias_history))
            stats['recent_max_bias'] = np.max(list(self.bias_history))
        
        if self.reward_history:
            stats['recent_avg_reward'] = np.mean(list(self.reward_history))
        
        if self.tta_history:
            stats['recent_avg_tta'] = np.mean(list(self.tta_history))
            stats['recent_max_tta'] = np.max(list(self.tta_history))
        
        return stats
    
    # === 可视化功能 ===
    
    def generate_visualization(self, env=None, save_dir: Optional[str] = None):
        """
        生成认知偏差模块的可视化图表
        
        Args:
            env: 环境实例（用于获取输出目录）
            save_dir: 保存目录，如果为None则自动确定
        """
        # 确定保存目录
        if save_dir is None:
            save_dir = self._get_visualization_dir(env)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成各种图表
        # self._plot_bias_history(save_dir)
        # self._plot_tta_distribution(save_dir)
        # self._plot_adaptive_factor(save_dir)
        self._plot_reward_comparison(save_dir)
        self._plot_detection_analysis(save_dir)  # 新增：检测分析图表
        self._plot_visual_aversion_details(save_dir)  # 新增：视觉厌恶详细信息
        
        logger.info(f"认知偏差可视化完成，图片保存在: {save_dir}")
        return save_dir
    
    def _get_visualization_dir(self, env=None) -> str:
        """获取可视化输出目录"""
        if env and hasattr(env, 'visualization_output_dir'):
            base_dir = env.visualization_output_dir
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = f"/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/fig_cog/cognitive_analysis_{timestamp}"
        
        bias_dir = os.path.join(base_dir, "cognitive_bias")
        os.makedirs(bias_dir, exist_ok=True)
        return bias_dir
    
    def _plot_bias_history(self, save_dir: str):
        """绘制偏差历史"""
        if not self.bias_history:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 偏差时间序列
        bias_data = list(self.bias_history)
        ax1.plot(bias_data, 'b-', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Bias Amount')
        ax1.set_title('Cognitive Bias Over Time')
        ax1.grid(True, alpha=0.3)
        
        # 偏差分布直方图
        ax2.hist(bias_data, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Bias Amount')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Cognitive Bias')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_bias = np.mean(bias_data)
        std_bias = np.std(bias_data)
        ax2.axvline(mean_bias, color='red', linestyle='--', 
                   label=f'Mean: {mean_bias:.3f}')
        ax2.axvline(mean_bias + std_bias, color='red', linestyle=':', 
                   label=f'±1σ: {std_bias:.3f}')
        ax2.axvline(mean_bias - std_bias, color='red', linestyle=':')
        ax2.legend()
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'bias_history.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_tta_distribution(self, save_dir: str):
        """绘制TTA分布"""
        if not self.tta_history:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # TTA时间序列
        tta_data = list(self.tta_history)
        ax1.plot(tta_data, 'g-', alpha=0.7)
        ax1.axhline(self.tta_threshold, color='red', linestyle='--', 
                   label=f'Threshold: {self.tta_threshold}')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Inverse TTA')
        ax1.set_title('Inverse TTA Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # TTA分布
        ax2.hist(tta_data, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(self.tta_threshold, color='red', linestyle='--', 
                   label=f'Threshold: {self.tta_threshold}')
        ax2.set_xlabel('Inverse TTA')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Inverse TTA')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'tta_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_adaptive_factor(self, save_dir: str):
        """绘制自适应因子变化"""
        if not self.adaptive_bias or self._step_count == 0:
            return
        
        # 这里简化处理，只显示当前的自适应因子
        # 实际使用中可以记录自适应因子的历史
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 创建一个简单的条形图显示当前状态
        categories = ['Min', 'Current', 'Max']
        values = [self.min_adaptive_factor, self.adaptive_factor, self.max_adaptive_factor]
        colors = ['blue', 'green', 'red']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylabel('Adaptive Factor')
        ax.set_title('Adaptive Bias Factor Status')
        ax.set_ylim(0, self.max_adaptive_factor * 1.2)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3, axis='y')
        
        save_path = os.path.join(save_dir, 'adaptive_factor.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_reward_comparison(self, save_dir: str):
        """绘制奖励对比（原始vs调整后）"""
        if not self.reward_history or not self.bias_history:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 计算调整后的奖励
        original_rewards = list(self.reward_history)
        biases = list(self.bias_history)
        adjusted_rewards = [r - b for r, b in zip(original_rewards, biases)]
        
        steps = range(len(original_rewards))
        
        # 绘制原始和调整后的奖励
        ax.plot(steps, original_rewards, 'b-', alpha=0.7, label='Original Reward')
        ax.plot(steps, adjusted_rewards, 'r-', alpha=0.7, label='Adjusted Reward')
        
        # 填充偏差区域
        ax.fill_between(steps, original_rewards, adjusted_rewards, 
                        alpha=0.3, color='yellow', label='Bias Effect')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Comparison: Original vs Cognitive Bias Adjusted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        textstr = f'Total Bias: {self._total_bias:.2f}\n'
        textstr += f'Activation Rate: {self._active_steps/max(1, self._step_count):.2%}\n'
        textstr += f'Avg Original: {np.mean(original_rewards):.3f}\n'
        textstr += f'Avg Adjusted: {np.mean(adjusted_rewards):.3f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'reward_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_detection_analysis(self, save_dir: str):
        """绘制检测分析图表"""
        if not self.detection_history:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 提取数据
        steps = range(len(self.detection_history))
        threat_counts = list(self.threat_count_history)
        distances = [d if d != float('inf') else None for d in self.distance_history]
        detection_data = list(self.detection_history)
        
        # 1. 威胁物体数量时间序列
        ax1.plot(steps, threat_counts, 'r-', alpha=0.7, linewidth=1.5)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Threat Count')
        ax1.set_title('Detected Threat Objects Over Time')
        ax1.grid(True, alpha=0.3)
        
        # 2. 最近威胁距离时间序列
        valid_distances = [d for d in distances if d is not None]
        valid_steps = [s for s, d in zip(steps, distances) if d is not None]
        if valid_distances:
            ax2.plot(valid_steps, valid_distances, 'b-', alpha=0.7, linewidth=1.5)
            ax2.axhline(self.visual_detection_distance, color='red', linestyle='--', 
                       label=f'Detection Range: {self.visual_detection_distance}m')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Distance (m)')
            ax2.set_title('Closest Threat Distance Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 威胁数量分布
        if threat_counts:
            unique_counts = sorted(set(threat_counts))
            count_freq = [threat_counts.count(c) for c in unique_counts]
            ax3.bar(unique_counts, count_freq, alpha=0.7, color='orange')
            ax3.set_xlabel('Number of Threats')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Threat Counts')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 距离分布（只包含有效距离）
        if valid_distances:
            ax4.hist(valid_distances, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax4.axvline(self.visual_detection_distance, color='red', linestyle='--', 
                       label=f'Detection Range: {self.visual_detection_distance}m')
            ax4.set_xlabel('Distance (m)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Threat Distances')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'detection_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_visual_aversion_details(self, save_dir: str):
        """绘制视觉厌恶详细信息"""
        if not self.detection_history:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 提取数据
        steps = range(len(self.detection_history))
        detection_data = list(self.detection_history)
        
        basic_ttas = [d['basic_inverse_tta'] for d in detection_data]
        final_ttas = [d['final_inverse_tta'] for d in detection_data]
        aversion_factors = [d['visual_aversion_factor'] for d in detection_data]
        agent_speeds = [d['agent_speed'] for d in detection_data]
        
        # 1. 基础vs最终inverse_tta对比
        ax1.plot(steps, basic_ttas, 'b-', alpha=0.7, label='Basic Inverse TTA', linewidth=1.5)
        ax1.plot(steps, final_ttas, 'r-', alpha=0.7, label='Final Inverse TTA (with aversion)', linewidth=1.5)
        ax1.axhline(self.tta_threshold, color='orange', linestyle='--', 
                   label=f'Threshold: {self.tta_threshold}')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Inverse TTA')
        ax1.set_title('Basic vs Visual Aversion Enhanced Inverse TTA')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 视觉厌恶因子时间序列
        ax2.plot(steps, aversion_factors, 'g-', alpha=0.7, linewidth=1.5)
        ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='No Aversion (1.0)')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Visual Aversion Factor')
        ax2.set_title('Visual Aversion Enhancement Factor Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 主车速度时间序列
        ax3.plot(steps, agent_speeds, 'm-', alpha=0.7, linewidth=1.5)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Agent Speed Over Time')
        ax3.grid(True, alpha=0.3)
        
        # 4. 视觉厌恶增强效果（最终TTA与基础TTA的比值）
        enhancement_ratios = []
        for basic, final in zip(basic_ttas, final_ttas):
            if basic > 0:
                enhancement_ratios.append(final / basic)
            else:
                enhancement_ratios.append(1.0)
        
        ax4.plot(steps, enhancement_ratios, 'orange', alpha=0.7, linewidth=1.5)
        ax4.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='No Enhancement')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Enhancement Ratio (Final/Basic)')
        ax4.set_title('Visual Aversion Enhancement Effect')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'visual_aversion_details.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def create_cognitive_bias_module(config: Optional[Dict[str, Any]] = None) -> CognitiveBiasModule:
    """
    工厂函数：创建认知偏差模块实例
    
    Args:
        config: 可选的偏差配置
        
    Returns:
        CognitiveBiasModule实例
    """
    return CognitiveBiasModule(config)


# 测试代码
if __name__ == "__main__":
    # 创建测试配置
    test_config = {
        'inverse_tta_coef': 1.5,
        'tta_threshold': 2.0,
        'adaptive_bias': True,
        'verbose': True
    }
    
    # 创建模块
    bias_module = create_cognitive_bias_module(test_config)
    
    # 模拟测试
    print("认知偏差模块测试")
    print(f"配置: {test_config}")
    
    # 模拟一些奖励处理
    for i in range(100):
        original_reward = np.random.uniform(-1, 1)
        # 模拟inverse_tta（随机生成）
        mock_info = {'inverse_tta': np.random.uniform(0, 5)}
        
        adjusted_reward, bias_info = bias_module.process_reward(
            original_reward, 
            env=None, 
            info=mock_info,
            is_ppo_mode=True
        )
        
        if i % 20 == 0:
            print(f"Step {i}: original={original_reward:.3f}, "
                  f"adjusted={adjusted_reward:.3f}, "
                  f"bias={bias_info['bias_applied']:.3f}, "
                  f"tta={bias_info['inverse_tta']:.3f}")
    
    # 获取统计信息
    stats = bias_module.get_statistics()
    print("\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n测试完成") 