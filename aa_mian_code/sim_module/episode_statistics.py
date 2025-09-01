"""
Episode统计器模块

该模块专门负责Episode过程中的数据收集、统计和分析。
从原始PPOCheckpointSimulator中提取统计相关功能。
"""

import numpy as np
import argparse
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .lane_change_tracker import LaneChangeTracker


class EpisodeStatistics:
    """Episode统计器
    
    专门负责：
    1. Episode统计数据结构的初始化和管理
    2. 每步统计信息的更新
    3. Episode结束时的统计计算
    4. 与变道跟踪器的协作
    """
    
    def __init__(self, lane_change_tracker: Optional['LaneChangeTracker'] = None, args: Optional[argparse.Namespace] = None):
        """
        初始化Episode统计器
        
        Args:
            lane_change_tracker: 变道跟踪器实例
            args: 命令行参数
        """
        self.lane_change_tracker = lane_change_tracker
        self.args = args
        self.current_episode_stats = {}
        
        print(f"📊 Episode统计器初始化完成")
        print(f"   变道跟踪: {'启用' if lane_change_tracker else '禁用'}")
    
    def initialize_episode(self) -> Dict:
        """
        初始化Episode统计数据结构
        
        Returns:
            初始化的统计数据字典
        """
        self.current_episode_stats = {
            "total_reward": 0.0,
            "episode_length": 0,
            "success": False,
            "collision": False,
            "out_of_road": False,
            "max_speed": 0.0,
            "avg_speed": 0.0,
            "path_completion": 0.0,
            "actions": [],
            "speeds": [],
            "rewards": [],
            
            # 变道统计
            "lane_change_penalties": [],
            "lane_change_speed_ratios": [],
            "cooldown_violations": [],
            "lane_changes": 0,
            
            # 认知模块统计
            "cognitive_bias_info": [],
            "cognitive_delay_info": [],
            "cognitive_perception_info": []
        }
        
        return self.current_episode_stats
    
    def update_step_stats(self, action: np.ndarray, reward: float, speed: float, 
                         info: Dict, env=None, step_count: int = 0) -> float:
        """
        更新每步的统计信息
        
        Args:
            action: 执行的动作
            reward: 获得的奖励
            speed: 当前速度
            info: 环境信息
            env: 环境实例
            step_count: 当前步数
            
        Returns:
            调整后的奖励（考虑变道惩罚）
        """
        # 基础统计更新
        self.current_episode_stats["total_reward"] += reward
        self.current_episode_stats["episode_length"] += 1
        self.current_episode_stats["rewards"].append(reward)
        self.current_episode_stats["actions"].append(action.copy())
        
        # 速度统计更新
        if speed is not None:
            self.current_episode_stats["speeds"].append(speed)
            self.current_episode_stats["max_speed"] = max(self.current_episode_stats["max_speed"], speed)
        
        # 变道检测和惩罚处理
        adjusted_reward = reward
        if self.lane_change_tracker and env:
            lane_change_detected, penalty, lane_stats = self.lane_change_tracker.detect_lane_change(
                env, step_count, speed if speed is not None else 0.0
            )
            
            if lane_change_detected:
                self.current_episode_stats["lane_changes"] += 1
                self.current_episode_stats["lane_change_penalties"].append(penalty)
                self.current_episode_stats["lane_change_speed_ratios"].append(lane_stats['speed_ratio'])
                self.current_episode_stats["cooldown_violations"].append(lane_stats['cooldown_violation'])
                
                # 应用变道惩罚到总奖励
                self.current_episode_stats["total_reward"] -= penalty
                adjusted_reward = reward - penalty
                
                # 调试信息（前几步）
                if step_count < 5:
                    print(f"    🚗 变道检测！惩罚: {penalty:.3f}, 速度比: {lane_stats['speed_ratio']:.3f}")
            
            # 更新车道索引（无论是否变道）
            self.lane_change_tracker.update_lane_index(env)
        
        return adjusted_reward
    
    def update_cognitive_stats(self, bias_info: Dict = None, delay_info: Dict = None, perception_info: Dict = None):
        """
        更新认知模块统计信息
        
        Args:
            bias_info: 认知偏差信息
            delay_info: 认知延迟信息  
            perception_info: 认知感知信息
        """
        if bias_info:
            try:
                # 只保存简化版本，避免复杂对象
                if isinstance(bias_info, dict):
                    simplified_info = {
                        'bias_strength': float(bias_info.get('bias_strength', 0.0)),
                        'bias_active': bool(bias_info.get('bias_active', False))
                    }
                else:
                    simplified_info = {'status': 'active', 'value': str(bias_info)[:50]}
                self.current_episode_stats["cognitive_bias_info"].append(simplified_info)
            except Exception as e:
                print(f"⚠️ 认知偏差统计收集失败: {e}")
        
        if delay_info:
            try:
                if isinstance(delay_info, dict):
                    simplified_info = {k: v for k, v in delay_info.items() if isinstance(v, (int, float, str, bool))}
                else:
                    simplified_info = {'delay_steps': delay_info}
                self.current_episode_stats["cognitive_delay_info"].append(simplified_info)
            except Exception as e:
                print(f"⚠️ 认知延迟统计收集失败: {e}")
        
        if perception_info:
            try:
                if isinstance(perception_info, dict):
                    simplified_info = {k: v for k, v in perception_info.items() if isinstance(v, (int, float, str, bool))}
                else:
                    simplified_info = {'status': 'active'}
                self.current_episode_stats["cognitive_perception_info"].append(simplified_info)
            except Exception as e:
                print(f"⚠️ 认知感知统计收集失败: {e}")
    
    def update_termination_stats(self, info: Dict):
        """
        更新终止条件统计
        
        Args:
            info: 环境终止信息
        """
        # 成功到达
        if info.get("arrive_dest", False):
            self.current_episode_stats["success"] = True
        
        # 碰撞
        if info.get("crash", False) or info.get("crash_vehicle", False):
            self.current_episode_stats["collision"] = True
        
        # 冲出道路
        if info.get("out_of_road", False):
            self.current_episode_stats["out_of_road"] = True
    
    def finalize_episode(self, env=None, info: Dict = None) -> Dict:
        """
        完成Episode统计，计算最终指标
        
        Args:
            env: 环境实例
            info: 最终环境信息
            
        Returns:
            完整的Episode统计信息
        """
        # 获取路径完成度信息
        try:
            if info and "route_completion" in info:
                self.current_episode_stats["path_completion"] = info["route_completion"]
            elif env and hasattr(env, 'agent') and hasattr(env.agent, 'navigation'):
                nav = env.agent.navigation
                if hasattr(nav, 'route_completion'):
                    self.current_episode_stats["path_completion"] = nav.route_completion
                elif hasattr(nav, 'get_current_lane_progress'):
                    self.current_episode_stats["path_completion"] = nav.get_current_lane_progress()
        except Exception as e:
            print(f"⚠️ 路径完成度获取失败: {e}")
            self.current_episode_stats["path_completion"] = 0.0
        
        # 计算平均速度
        if self.current_episode_stats["speeds"]:
            self.current_episode_stats["avg_speed"] = np.mean(self.current_episode_stats["speeds"])
        
        return self.current_episode_stats.copy()
    
    def get_current_stats(self) -> Dict:
        """获取当前Episode统计信息"""
        return self.current_episode_stats.copy()
    
    def reset_episode(self):
        """重置Episode统计"""
        self.current_episode_stats.clear()
        if self.lane_change_tracker:
            self.lane_change_tracker.reset()
        print("🔄 Episode统计已重置")
    
    @staticmethod
    def compute_summary_stats(all_episode_stats: List[Dict]) -> Dict:
        """
        计算多个Episode的总体统计
        
        Args:
            all_episode_stats: 所有Episode统计信息列表
            
        Returns:
            总体统计信息
        """
        if not all_episode_stats:
            return {}
        
        summary = {
            "num_episodes": len(all_episode_stats),
            "success_rate": np.mean([s['success'] for s in all_episode_stats]),
            "collision_rate": np.mean([s['collision'] for s in all_episode_stats]),
            "out_of_road_rate": np.mean([s['out_of_road'] for s in all_episode_stats]),
            "avg_reward": np.mean([s['total_reward'] for s in all_episode_stats]),
            "std_reward": np.std([s['total_reward'] for s in all_episode_stats]),
            "avg_episode_length": np.mean([s['episode_length'] for s in all_episode_stats]),
            "avg_speed": np.mean([s['avg_speed'] for s in all_episode_stats]),
            "avg_path_completion": np.mean([s['path_completion'] for s in all_episode_stats]),
            
            # 变道统计
            "total_lane_changes": np.sum([s['lane_changes'] for s in all_episode_stats]),
            "avg_lane_change_penalty": np.mean([np.mean(s['lane_change_penalties']) if s['lane_change_penalties'] else 0 for s in all_episode_stats]),
            "avg_lane_change_speed_ratio": np.mean([np.mean(s['lane_change_speed_ratios']) if s['lane_change_speed_ratios'] else 0 for s in all_episode_stats]),
            "total_cooldown_violations": np.sum([np.sum(s['cooldown_violations']) if s['cooldown_violations'] else 0 for s in all_episode_stats])
        }
        
        return summary
    
    @staticmethod
    def print_episode_results(episode_num: int, stats: Dict):
        """
        打印单个Episode结果
        
        Args:
            episode_num: Episode编号
            stats: Episode统计信息
        """
        print(f"📊 Episode {episode_num} 结果:")
        print(f"   💰 总奖励: {stats['total_reward']:.2f}")
        print(f"   📏 Episode长度: {stats['episode_length']}")
        print(f"   🏁 成功到达: {'✅' if stats['success'] else '❌'}")
        print(f"   💥 发生碰撞: {'❌' if stats['collision'] else '✅'}")
        print(f"   🛣️  冲出道路: {'❌' if stats['out_of_road'] else '✅'}")
        print(f"   🚀 最高速度: {stats['max_speed']:.2f} m/s")
        print(f"   📈 平均速度: {stats['avg_speed']:.2f} m/s")
        print(f"   🎯 路径完成度: {stats['path_completion']:.1%}")
        
        # 变道统计输出
        if stats['lane_changes'] > 0:
            avg_penalty = np.mean(stats['lane_change_penalties']) if stats['lane_change_penalties'] else 0
            avg_speed_ratio = np.mean(stats['lane_change_speed_ratios']) if stats['lane_change_speed_ratios'] else 0
            cooldown_violations = np.sum(stats['cooldown_violations']) if stats['cooldown_violations'] else 0
            print(f"   🚗 变道次数: {stats['lane_changes']}")
            print(f"   💸 平均变道惩罚: {avg_penalty:.3f}")
            print(f"   ⚡ 平均变道速度比: {avg_speed_ratio:.3f}")
            print(f"   ⏰ 冷却期违规: {cooldown_violations}")
        else:
            print(f"   🚗 变道次数: 0")
    
    @staticmethod
    def print_summary_stats(summary: Dict):
        """
        打印总体统计信息
        
        Args:
            summary: 总体统计信息
        """
        print(f"\n📈 总体统计 ({summary['num_episodes']} episodes)")
        print("=" * 60)
        
        print(f"🏆 成功率: {summary['success_rate']:.1%}")
        print(f"💥 碰撞率: {summary['collision_rate']:.1%}")
        print(f"🛣️  冲出道路率: {summary['out_of_road_rate']:.1%}")
        print(f"💰 平均奖励: {summary['avg_reward']:.2f}")
        print(f"📏 平均Episode长度: {summary['avg_episode_length']:.1f}")
        print(f"🚀 平均速度: {summary['avg_speed']:.2f} m/s")
        print(f"🎯 平均路径完成度: {summary['avg_path_completion']:.1%}")
        
        # 变道统计输出
        print(f"🚗 总变道次数: {summary['total_lane_changes']}")
        print(f"💸 平均变道惩罚: {summary['avg_lane_change_penalty']:.3f}")
        print(f"⚡ 平均变道速度比: {summary['avg_lane_change_speed_ratio']:.3f}")
        print(f"⏰ 总冷却期违规: {summary['total_cooldown_violations']}")
    
    def update_lane_change_tracker(self, new_tracker: Optional['LaneChangeTracker']):
        """更新变道跟踪器"""
        self.lane_change_tracker = new_tracker
        print(f"🔄 Episode统计器变道跟踪器已更新: {'启用' if new_tracker else '禁用'}")
    
    def get_statistics_summary(self) -> Dict:
        """获取统计器状态摘要"""
        return {
            'lane_change_tracking': self.lane_change_tracker is not None,
            'current_episode_length': self.current_episode_stats.get('episode_length', 0),
            'current_total_reward': self.current_episode_stats.get('total_reward', 0.0),
            'has_active_episode': bool(self.current_episode_stats)
        } 