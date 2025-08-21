#!/usr/bin/env python3
"""
认知偏差模块 - TTA（Time-To-Arrival）风险厌恶
基于inverse_tta值动态调整奖励信号，模拟人类对风险的认知偏差

核心功能：
1. 根据环境中的inverse_tta值动态调整奖励
2. 模拟人类对风险的认知偏差（风险厌恶）
3. 支持自适应偏差调整
4. 记录偏差历史用于分析

该模块独立于gym环境，直接在MetaDrive环境中工作
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)


class CognitiveBiasModule:
    """
    认知偏差模块 - 基于TTA的风险厌恶
    
    功能：
    1. 根据环境中的inverse_tta值动态调整奖励
    2. 模拟人类对风险的认知偏差
    3. 支持自适应偏差调整
    """
    
    def __init__(self, bias_config: Optional[Dict[str, Any]] = None, 
                 cognitive_perception_module: Optional[Any] = None):
        """
        初始化认知偏差模块
        
        Args:
            bias_config: 偏差配置参数
            cognitive_perception_module: 认知感知模块实例，用于获取处理后的雷达数据
        """
        # 获取配置
        config = bias_config or self._get_default_config()
        
        # 核心参数 - 新的looming penalty模式
        self.inverse_tta_coef = config.get('inverse_tta_coef', 1.0)  # looming penalty系数 c
        self.tta_threshold = config.get('tta_threshold', 1.0)  # TTA阈值
        
        # 检测参数（保留用于雷达数据获取）
        self.visual_detection_distance = config.get('visual_detection_distance', 50.0)  # 雷达检测距离
        
        # 历史记录（用于分析）
        self.bias_history = deque(maxlen=config.get('history_length', 100))
        self.reward_history = deque(maxlen=config.get('history_length', 100))
        self.tta_history = deque(maxlen=config.get('history_length', 100))
        
        # 检测历史记录
        extended_history_length = config.get('extended_history_length', 1000)  # 扩展到1000步
        self.detection_history = deque(maxlen=extended_history_length)  # 检测到的物体信息
        self.distance_history = deque(maxlen=extended_history_length)   # 最近威胁距离历史
        self.threat_count_history = deque(maxlen=extended_history_length)  # 威胁物体数量历史
        
        # 统计变量
        self._step_count = 0
        self._total_bias = 0.0
        self._active_steps = 0  # 偏差激活的步数
        
        # 附加的环境实例
        self.attached_env = None
        
        # === 新增：认知感知模块引用 ===
        self.cognitive_perception_module = cognitive_perception_module
        
        # 日志级别
        self.verbose = config.get('verbose', False)
        
        logger.info(f"CognitiveBiasModule初始化 (Looming Penalty模式): c={self.inverse_tta_coef}, "
                   f"tta_threshold={self.tta_threshold}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'inverse_tta_coef': 1.0,      # looming penalty系数 c
            'tta_threshold': 1.0,          # TTA阈值
            'history_length': 100,         # 常规历史记录长度
            'extended_history_length': 1000,  # 扩展历史记录长度（用于可视化分析）
            'verbose': False,              # 详细日志输出
            
            # 检测参数（保留用于雷达数据获取）
            'visual_detection_distance': 50.0,  # 雷达检测距离（米）
        }
    
    def attach_to_env(self, env):
        """
        将偏差模块附加到MetaDrive环境
        
        Args:
            env: MetaDrive环境实例
            
        Returns:
            bool: 是否成功附加
        """
        try:
            self.attached_env = env
            
            # 检查环境是否支持获取inverse_tta
            if not self._check_tta_support(env):
                logger.warning("环境可能不支持inverse_tta功能")
                return False
            
            logger.info("✅ 认知偏差模块已成功附加到环境")
            return True
            
        except Exception as e:
            logger.error(f"❌ 附加认知偏差模块失败: {e}")
            return False
    
    def set_cognitive_perception_module(self, cognitive_perception_module):
        """
        设置认知感知模块引用
        
        Args:
            cognitive_perception_module: 认知感知模块实例
        """
        self.cognitive_perception_module = cognitive_perception_module
        logger.info("✅ 认知感知模块引用已设置到认知偏差模块")
    
    def _check_tta_support(self, env) -> bool:
        """
        检查环境是否支持inverse_tta
        
        Args:
            env: 环境实例
            
        Returns:
            bool: 是否支持
        """
        # 检查多种可能的TTA获取方式
        has_tta = (
            hasattr(env, 'current_inverse_tta') or
            (hasattr(env, 'agent') and hasattr(env.agent, 'inverse_tta')) or
            (hasattr(env, 'get_inverse_tta'))
        )
        
        if not has_tta:
            # 尝试通过info字典获取
            try:
                # 模拟一步来检查info是否包含inverse_tta
                # 这里只是检查，不实际执行
                logger.info("环境未直接提供inverse_tta属性，将尝试从info字典获取")
            except:
                pass
        
        return True  # 默认假设支持，实际运行时再处理
    
    def get_inverse_tta(self, env, info: Optional[Dict] = None) -> Optional[float]:
        """
        从环境中获取inverse_tta值
        
        Args:
            env: 环境实例
            info: 可选的info字典
            
        Returns:
            inverse_tta值，如果不存在则返回None
        """
        # 方式1：直接从环境属性获取
        if hasattr(env, 'current_inverse_tta'):
            return env.current_inverse_tta
        
        # 方式2：从agent获取
        if hasattr(env, 'agent'):
            agent = env.agent
            if hasattr(agent, 'inverse_tta'):
                return agent.inverse_tta
            
            # 尝试计算inverse_tta
            if hasattr(agent, 'compute_inverse_tta'):
                return agent.compute_inverse_tta()
        
        # 方式3：从环境方法获取
        if hasattr(env, 'get_inverse_tta'):
            return env.get_inverse_tta()
        
        # 方式4：从info字典获取
        if info and 'inverse_tta' in info:
            return info['inverse_tta']
        
        # 方式5：尝试手动计算（如果有必要的信息）
        if hasattr(env, 'agent'):
            tta_value = self._compute_inverse_tta_manually(env)
            if tta_value is not None:
                return tta_value
        
        return None
    
    def _calculate_relative_speed_from_radar(self, current_min_distance: float, data_source: str) -> Optional[float]:
        """
        基于雷达距离变化率计算相对速度
        
        Args:
            current_min_distance: 当前检测到的最小距离
            data_source: 数据来源标识
            
        Returns:
            相对速度（m/s），正值表示接近，如果无法计算则返回None
        """
        try:
            # 方法1：从认知感知模块获取距离历史（如果可用）
            if self.cognitive_perception_module is not None and data_source == "认知感知模块":
                # 获取正前方雷达束的历史数据
                history_data = self.cognitive_perception_module.get_processed_radar_distances()
                if hasattr(self.cognitive_perception_module, 'noise_lidar') and self.cognitive_perception_module.noise_lidar:
                    front_beam_history = self.cognitive_perception_module.noise_lidar.get_front_beam_history()
                    
                    if front_beam_history['length'] >= 2:
                        distances = front_beam_history['noisy_distances']  # 使用加噪后的距离
                        timestamps = front_beam_history['timestamps']
   
                        if len(distances) >= 2 and len(timestamps) >= 2:
                            # 计算最近两个时间点的距离变化率
                            dt = timestamps[-1] - timestamps[-2]
                            dr = distances[-2] - distances[-1]  # 距离减少为正（接近）
                            
                            if dt > 1e-6:  # 避免除零
                                relative_speed = dr / dt
                                logger.info(f"📡 从认知感知模块计算相对速度: dr={dr:.3f}m, dt={dt:.3f}s, v_rel={relative_speed:.3f}m/s")
                                return max(0.0, relative_speed)  # 确保非负
            
            # 方法2：基于认知偏差模块自身的距离历史
            if len(self.distance_history) >= 2:
                # 使用最近两次的距离测量
                recent_distance = self.distance_history[-1]
                prev_distance = self.distance_history[-2]
                
                # 估算时间间隔（假设固定的环境步长）
                # MetaDrive环境的典型频率是10Hz，即0.1s间隔
                estimated_dt = 0.1  # 秒
                
                # 计算距离变化
                dr = prev_distance - recent_distance  # 距离减少为正（接近）
                relative_speed = dr / estimated_dt
                
                logger.info(f"📊 从历史距离计算相对速度: dr={dr:.3f}m, dt={estimated_dt:.3f}s, v_rel={relative_speed:.3f}m/s")
                return max(0.0, relative_speed)  # 确保非负
            
            # 方法3：如果历史数据不足，无法计算
            logger.info(f"⚠️ 历史数据不足，无法计算相对速度")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ 相对速度计算异常: {e}")
            return None
    
    def _compute_inverse_tta_manually(self, env) -> Optional[float]:
        """
        手动计算inverse_tta（基于认知感知模块处理后的雷达射线数据）
        从正前方±15度范围内（总30度）的所有雷达射线中找到最小距离，使用TTA = min_distance / agent_speed计算
        
        Args:
            env: 环境实例
            
        Returns:
            计算得到的inverse_tta值，如果无法计算则返回None
        """
        try:
            agent = env.agent
            agent_speed = agent.speed
            
            # 检测参数
            detection_angle_degrees = 30  # ±15度范围，总30度
            max_detection_distance = self.visual_detection_distance  # 最大检测距离
            
            min_distance = float('inf')
            radar_data_found = False
            beam_count = 0
            valid_beams = 0
            data_source = "未知"
            
            # 强制输出调试信息，无论verbose设置如何
            logger.info(f"🚀 开始雷达TTA计算: 检测角度=±{detection_angle_degrees//2}°(总{detection_angle_degrees}°), 最大距离={max_detection_distance}m, 当前速度={agent_speed:.2f}m/s")
            
            # 方法1：优先从认知感知模块获取处理后的雷达数据
            if self.cognitive_perception_module is not None:
                try:
                    processed_distances = self.cognitive_perception_module.get_processed_radar_distances()
                    if processed_distances is not None and len(processed_distances) > 0:
                        radar_data_found = True
                        beam_count = len(processed_distances)
                        data_source = "认知感知模块"
                        
                        # 使用相同的角度计算逻辑
                        total_beams = len(processed_distances)
                        angle_per_beam = 360.0 / total_beams
                        front_beam_index = 0  # 正前方雷达束索引
                        beam_range = int(detection_angle_degrees / angle_per_beam)  # ±15度对应的射线数量
                        
                        start_index = (front_beam_index - beam_range) % total_beams
                        end_index = (front_beam_index + beam_range + 1) % total_beams
                        
                        logger.info(f"📡 认知感知模块雷达配置: 总射线={total_beams}, 每度射线数={total_beams/360:.1f}, "
                                   f"前方射线范围=[{start_index}:{end_index}], 射线数量={beam_range*2+1}")
                        
                        # 提取前方±15度范围内的雷达数据
                        if start_index < end_index:
                            front_radar_data = processed_distances[start_index:end_index]
                        else:
                            front_radar_data = np.concatenate([
                                processed_distances[start_index:],
                                processed_distances[:end_index]
                            ])
                        
                        # 筛选有效距离（排除无效值）
                        valid_distances = front_radar_data[
                            (front_radar_data > 0) & 
                            (front_radar_data < max_detection_distance) &
                            (front_radar_data != float('inf')) &
                            (~np.isnan(front_radar_data))
                        ]
                        
                        valid_beams = len(valid_distances)
                        
                        if len(valid_distances) > 0:
                            min_distance = float(np.min(valid_distances))
                            
                            logger.info(f"✅ 认知感知模块雷达数据成功: 总射线={total_beams}, 有效距离数={valid_beams}, "
                                       f"最小距离={min_distance:.2f}m")
                        else:
                            logger.info(f"⚠️ 认知感知模块雷达数据无有效距离: 总射线={total_beams}, 检查距离数={len(front_radar_data)}")
                    else:
                        logger.warning(f"⚠️ 认知感知模块未提供有效雷达数据")
                        
                except Exception as e:
                    logger.warning(f"⚠️ 从认知感知模块获取雷达数据失败: {e}")
            
            # 方法2：回退到从观测空间获取雷达数据（如果认知感知模块不可用）
            if not radar_data_found:
                try:
                    obs = env.get_single_observation()
                    logger.info(f"🔍 回退方案-观测数据类型: {type(obs)}, 键: {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")
                    
                    # 处理字典格式的观测数据
                    if isinstance(obs, dict) and 'lidar' in obs:
                        lidar_obs = obs['lidar']
                        logger.info(f"🔍 回退方案-雷达数据类型: {type(lidar_obs)}, 形状: {lidar_obs.shape if hasattr(lidar_obs, 'shape') else 'N/A'}")
                        if isinstance(lidar_obs, np.ndarray) and len(lidar_obs) > 0:
                            radar_data_found = True
                            beam_count = len(lidar_obs)
                            data_source = "环境观测-字典"
                            front_radar_data = lidar_obs  # 直接使用观测数据
                            
                    # 处理LidarStateObservation对象
                    elif hasattr(obs, 'cloud_points'):
                        logger.info(f"🔍 LidarStateObservation属性检查: cloud_points={obs.cloud_points}, type={type(obs.cloud_points) if obs.cloud_points is not None else 'None'}")
                        # 尝试触发雷达观测
                        if hasattr(obs, 'lidar_observe') and hasattr(env, 'agent'):
                            try:
                                lidar_data = obs.lidar_observe(env.agent)
                                logger.info(f"🔍 尝试lidar_observe: 数据类型={type(lidar_data)}, 长度={len(lidar_data) if lidar_data else 0}")
                                if lidar_data and len(lidar_data) > 0:
                                    lidar_obs = np.array(lidar_data)
                                    logger.info(f"🔍 从lidar_observe获取数据: 形状={lidar_obs.shape}")
                                    radar_data_found = True
                                    beam_count = len(lidar_obs)
                                    data_source = "环境观测-LidarObserve"
                                    front_radar_data = lidar_obs
                            except Exception as e:
                                logger.warning(f"⚠️ lidar_observe调用失败: {e}")
                        
                        # 检查cloud_points
                        if not radar_data_found and obs.cloud_points is not None:
                            cloud_points = obs.cloud_points
                            if isinstance(cloud_points, (list, np.ndarray)) and len(cloud_points) > 0:
                                lidar_obs = np.array(cloud_points)
                                logger.info(f"🔍 回退方案-LidarStateObservation雷达数据: 形状={lidar_obs.shape}, 类型={type(lidar_obs)}")
                                radar_data_found = True
                                beam_count = len(lidar_obs)
                                data_source = "环境观测-LidarState"
                                front_radar_data = lidar_obs
                            else:
                                logger.warning(f"⚠️ LidarStateObservation的cloud_points为空或无效: {cloud_points}")
                        
                        # 如果都没获取到，打印更多调试信息
                        if not radar_data_found:
                            logger.info(f"🔍 LidarStateObservation调试信息:")
                            logger.info(f"    hasattr cloud_points: {hasattr(obs, 'cloud_points')}")
                            logger.info(f"    hasattr lidar_observe: {hasattr(obs, 'lidar_observe')}")
                            logger.info(f"    hasattr state_obs: {hasattr(obs, 'state_obs')}")
                            logger.info(f"    dir(obs): {[attr for attr in dir(obs) if not attr.startswith('_')]}")
                    
                    # 如果获取到雷达数据，进行处理
                    if radar_data_found:
                        # 使用相同的角度计算逻辑
                        total_beams = len(front_radar_data)
                        angle_per_beam = 360.0 / total_beams
                        front_beam_index = 0
                        beam_range = int(detection_angle_degrees / angle_per_beam)
                        
                        start_index = (front_beam_index - beam_range) % total_beams
                        end_index = (front_beam_index + beam_range + 1) % total_beams
                        
                        logger.info(f"📡 环境观测雷达配置: 总射线={total_beams}, 每度射线数={total_beams/360:.1f}, "
                                   f"前方射线范围=[{start_index}:{end_index}]")
                        
                        # 提取前方范围的雷达数据
                        if start_index < end_index:
                            front_radar_data = front_radar_data[start_index:end_index]
                        else:
                            front_radar_data = np.concatenate([
                                front_radar_data[start_index:],
                                front_radar_data[:end_index]
                            ])
                        
                        # 注意：观测数据可能是归一化的(0-1)，需要转换为距离
                        # 假设最大检测距离为50米（与配置保持一致）
                        if np.max(front_radar_data) <= 1.0:  # 可能是归一化数据
                            front_radar_data = front_radar_data * 50.0  # 转换为米
                            logger.info(f"🔄 检测到归一化数据，已转换为距离（米）")
                        
                        valid_distances = front_radar_data[
                            (front_radar_data > 0) & 
                            (front_radar_data < max_detection_distance) &
                            (front_radar_data != float('inf')) &
                            (~np.isnan(front_radar_data))
                        ]
                        
                        valid_beams = len(valid_distances)
                        
                        if len(valid_distances) > 0:
                            min_distance = float(np.min(valid_distances))
                            
                            logger.info(f"✅ 环境观测雷达数据成功: 总射线={total_beams}, 有效距离数={valid_beams}, "
                                       f"最小距离={min_distance:.2f}m, 数据源={data_source}")
                    else:
                        logger.warning(f"⚠️ 无法从观测数据中提取雷达信息")
                    
                except Exception as e:
                    logger.warning(f"⚠️ 回退方案-从环境观测获取雷达数据失败: {e}")
            
            # 计算TTA和inverse_TTA
            if radar_data_found and min_distance < float('inf'):
                # 尝试基于雷达距离变化率计算相对速度
                relative_speed = self._calculate_relative_speed_from_radar(min_distance, data_source)
                

                speed_source = "雷达距离变化率"
                logger.info(f"✅ 使用相对速度: {relative_speed:.2f}m/s (来源: {speed_source})")
                
                # 计算TTA = distance / relative_speed  
                tta = min_distance / relative_speed
                # 计算inverse_TTA = relative_speed / distance
                basic_inverse_tta = relative_speed / min_distance
                
                # 新的looming penalty计算: L = c * tanh(inverse_TTA)
                inverse_tta = basic_inverse_tta  # 保持基础inverse_tta用于记录
                
                # 记录检测信息
                detection_info = {
                    'method': 'radar_beam',
                    'data_source': data_source,  # 新增：记录数据来源
                    'total_beams': beam_count,
                    'valid_beams': valid_beams,
                    'min_distance': min_distance,
                    'agent_speed': agent_speed,
                    'relative_speed': relative_speed,  # 新增：相对速度
                    'speed_source': speed_source,     # 新增：速度来源
                    'tta': tta,
                    'basic_inverse_tta': basic_inverse_tta,
                    'final_inverse_tta': inverse_tta,
                    'detection_angle': detection_angle_degrees
                }
                self.detection_history.append(detection_info)
                self.distance_history.append(min_distance)
                self.threat_count_history.append(1 if min_distance < float('inf') else 0)
                
                logger.info(f"🎯 雷达TTA计算成功 [数据源: {data_source}]: 最小距离={min_distance:.2f}m, "
                           f"相对速度={relative_speed:.2f}m/s ({speed_source}), "
                           f"TTA={tta:.2f}s, inverse_tta={inverse_tta:.3f}")
                
                return inverse_tta
            
            # 如果没有有效的雷达数据，记录空检测信息
            detection_info = {
                'method': 'radar_beam',
                'data_source': data_source,  # 新增：记录数据来源
                'total_beams': beam_count,
                'valid_beams': valid_beams,
                'min_distance': float('inf'),
                'agent_speed': agent_speed,
                'tta': float('inf'),
                'basic_inverse_tta': 0.0,
                'final_inverse_tta': 0.1 if agent_speed > 0 else 0.0,
                'detection_angle': detection_angle_degrees
            }
            self.detection_history.append(detection_info)
            self.distance_history.append(float('inf'))
            self.threat_count_history.append(0)
            
            # 强制输出调试信息
            logger.info(f"❌ 雷达TTA计算失败 [数据源: {data_source}]: 无有效检测数据, 速度={agent_speed:.2f}m/s, "
                       f"雷达数据获取={'成功' if radar_data_found else '失败'}, "
                       f"总射线数={beam_count}, 有效射线数={valid_beams}")
            
            # 返回低威胁值
            return 0.1 if agent_speed > 0 else 0.0
            
        except Exception as e:
            logger.error(f"雷达TTA计算异常: {e}")
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
        
        # 应用新的looming penalty: L = c * tanh(inverse_TTA)
        if inverse_tta > self.tta_threshold:
            # 计算looming penalty
            looming_penalty = self.inverse_tta_coef * np.tanh(inverse_tta)
            
            # 应用偏差（惩罚高风险行为）
            adjusted_reward = original_reward - looming_penalty
            
            # 记录偏差
            self.bias_history.append(looming_penalty)
            bias_info['bias_applied'] = looming_penalty
            bias_info['bias_active'] = True
            
            # 更新统计
            self._total_bias += looming_penalty
            self._active_steps += 1
            
        else:
            # 不应用偏差
            adjusted_reward = original_reward
            self.bias_history.append(0.0)
        
        # 记录奖励历史
        self.reward_history.append(original_reward)
        
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
    

    def reset(self):
        """重置认知偏差模块状态"""
        # 清空历史记录
        self.bias_history.clear()
        self.reward_history.clear()
        self.tta_history.clear()
        self.detection_history.clear()
        self.distance_history.clear()
        self.threat_count_history.clear()
        
        # 重置统计变量
        self._step_count = 0
        self._total_bias = 0.0
        self._active_steps = 0
        
        logger.info("CognitiveBiasModule已重置")
    
    def detach_from_env(self):
        """从环境中分离偏差模块"""
        self.attached_env = None
        logger.info("认知偏差模块已从环境分离")
    
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
            'average_bias': self._total_bias / max(1, self._step_count)
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