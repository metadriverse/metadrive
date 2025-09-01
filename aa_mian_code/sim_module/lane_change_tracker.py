"""
变道跟踪器模块

该模块专门负责变道检测、冷却时间管理和惩罚计算。
从原始PPOCheckpointSimulator中提取变道相关功能。
"""

import argparse
from typing import Dict, Tuple, Optional


class LaneChangeTracker:
    """变道跟踪器
    
    专门负责：
    1. 变道检测（基于车道索引变化）
    2. 变道冷却时间管理
    3. 变道惩罚计算（基于速度和冷却状态）
    4. 变道统计信息收集
    """
    
    def __init__(self, args: Optional[argparse.Namespace] = None):
        """
        初始化变道跟踪器
        
        Args:
            args: 命令行参数，包含变道惩罚配置
        """
        self.args = args
        
        # 变道状态跟踪
        self._last_lane_change_step = {}  # 存储最后变道的时间步
        self._last_lane_index = {}        # 存储每个agent的上一个车道索引
        self._lane_change_cooldown_steps = None  # 冷却时间步数
        
        # 变道惩罚参数（从命令行参数获取或使用默认值）
        self.w_lc = getattr(args, 'w_lc', 0.6) if args else 0.6  # 基础变道惩罚权重
        self.w_lc_cool = getattr(args, 'w_lc_cool', 0.6) if args else 0.6  # 冷却期惩罚权重
        self.k_speed = getattr(args, 'k_speed', 1.0) if args else 1.0  # 速度放大系数
        self.v_limit = getattr(args, 'v_limit', 15.0) if args else 15.0  # 速度限制
        self.lc_cooldown_s = getattr(args, 'lc_cooldown_s', 4.0) if args else 4.0  # 冷却时间（秒）
        
        print(f"🚗 变道跟踪器初始化:")
        print(f"   基础惩罚权重: {self.w_lc}")
        print(f"   冷却期惩罚权重: {self.w_lc_cool}")
        print(f"   速度放大系数: {self.k_speed}")
        print(f"   速度限制: {self.v_limit} m/s")
        print(f"   冷却时间: {self.lc_cooldown_s}s")
    
    def setup_cooldown(self, env):
        """
        动态设置变道冷却时间步数
        
        Args:
            env: MetaDrive环境实例
        """
        try:
            if env and hasattr(env, 'config'):
                # 获取物理步长和决策重复次数
                physics_step_size = env.config.get('physics_world_step_size', 0.02)
                decision_repeat = env.config.get('decision_repeat', 5)
                
                # 计算实际有效频率
                effective_time_step = physics_step_size * decision_repeat
                effective_frequency = 1.0 / effective_time_step
                
                # 计算冷却时间步数
                self._lane_change_cooldown_steps = int(self.lc_cooldown_s * effective_frequency)
                
                print(f"🔧 变道冷却时间设置:")
                print(f"   物理步长: {physics_step_size:.3f}s")
                print(f"   决策重复: {decision_repeat}")
                print(f"   有效频率: {effective_frequency:.1f}Hz")
                print(f"   冷却时间: {self.lc_cooldown_s}s → {self._lane_change_cooldown_steps}步")
            else:
                # 如果无法获取环境配置，使用默认值
                self._lane_change_cooldown_steps = int(self.lc_cooldown_s * 10)
                print(f"⚠️  无法获取环境配置，使用默认10Hz假设")
                print(f"   冷却时间: {self.lc_cooldown_s}s → {self._lane_change_cooldown_steps}步")
                
        except Exception as e:
            # 异常处理，使用默认值
            self._lane_change_cooldown_steps = int(self.lc_cooldown_s * 10)
            print(f"⚠️  设置冷却时间失败: {e}，使用默认10Hz假设")
            print(f"   冷却时间: {self.lc_cooldown_s}s → {self._lane_change_cooldown_steps}步")
    
    def detect_lane_change(self, env, step_count: int, current_speed: float) -> Tuple[bool, float, Dict]:
        """
        检测变道并计算惩罚
        
        Args:
            env: MetaDrive环境实例
            step_count: 当前步数
            current_speed: 当前车辆速度
            
        Returns:
            tuple: (是否发生变道, 惩罚值, 统计信息)
        """
        lane_change_detected = False
        penalty = 0.0
        stats = {
            'speed_ratio': 0.0,
            'cooldown_violation': 0,
            'base_penalty': 0.0,
            'speed_penalty': 0.0,
            'cooldown_penalty': 0.0
        }
        
        # 检查是否有车道索引信息
        if not hasattr(env.agent, 'lane_index'):
            return lane_change_detected, penalty, stats
        
        current_lane_index = env.agent.lane_index
        agent_id = getattr(env.agent, 'id', id(env.agent))
        
        # 检查是否存储了上一个车道索引
        if agent_id in self._last_lane_index:
            if self._last_lane_index[agent_id] != current_lane_index:
                lane_change_detected = True
        else:
            # 第一次运行，记录初始车道索引
            self._last_lane_index[agent_id] = current_lane_index
            return lane_change_detected, penalty, stats
        
        # 如果检测到变道，计算惩罚
        if lane_change_detected:
            # 计算速度比例
            speed_ratio = abs(current_speed) / self.v_limit
            speed_ratio = min(speed_ratio, 2.0)  # 限制最大比例
            stats['speed_ratio'] = speed_ratio
            
            # 基础变道惩罚
            base_penalty = self.w_lc
            stats['base_penalty'] = base_penalty
            
            # 高速放大惩罚
            speed_penalty = base_penalty * (1 + self.k_speed * speed_ratio)
            stats['speed_penalty'] = speed_penalty
            
            # 检查冷却时间违规
            cooldown_penalty = 0.0
            if agent_id in self._last_lane_change_step:
                steps_since_last_change = step_count - self._last_lane_change_step[agent_id]
                if steps_since_last_change < self._lane_change_cooldown_steps:
                    # 冷却期内，附加惩罚
                    cooldown_penalty = self.w_lc_cool
                    stats['cooldown_violation'] = 1
            
            stats['cooldown_penalty'] = cooldown_penalty
            
            # 总惩罚
            penalty = speed_penalty + cooldown_penalty
            
            # 更新最后变道时间
            self._last_lane_change_step[agent_id] = step_count
        
        return lane_change_detected, penalty, stats
    
    def update_lane_index(self, env):
        """
        更新车道索引记录（无论是否变道都要调用）
        
        Args:
            env: MetaDrive环境实例
        """
        if hasattr(env.agent, 'lane_index'):
            current_lane_index = env.agent.lane_index
            agent_id = getattr(env.agent, 'id', id(env.agent))
            self._last_lane_index[agent_id] = current_lane_index
    
    def reset(self):
        """重置变道跟踪状态"""
        self._last_lane_change_step.clear()
        self._last_lane_index.clear()
        print("🔄 变道跟踪器状态已重置")
    
    def get_statistics(self) -> Dict:
        """获取变道跟踪统计信息"""
        return {
            'total_agents_tracked': len(self._last_lane_index),
            'agents_with_lane_changes': len(self._last_lane_change_step),
            'cooldown_steps': self._lane_change_cooldown_steps,
            'config': {
                'w_lc': self.w_lc,
                'w_lc_cool': self.w_lc_cool,
                'k_speed': self.k_speed,
                'v_limit': self.v_limit,
                'lc_cooldown_s': self.lc_cooldown_s
            }
        } 