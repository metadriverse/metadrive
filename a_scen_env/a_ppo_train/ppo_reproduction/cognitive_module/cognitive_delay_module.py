"""
认知延迟模块 - 独立实现
基于人类驾驶员的动作延迟和平滑特性
仅在PPO模式下应用执行延迟
"""

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
from datetime import datetime


class CognitiveDelayModule:
    """
    认知延迟模块 - 独立实现，不依赖gym环境
    仅在PPO模式下应用执行延迟
    
    该模块模拟人类驾驶员的执行延迟特性：
    1. 动作平滑：避免突然的控制变化
    2. 执行延迟：模拟从决策到执行的时间间隔
    3. 缓冲机制：使用队列实现延迟效果
    4. 可视化：记录和展示延迟效果
    """
    
    def __init__(self, delay_steps=2, enable_smoothing=False, smoothing_factor=0.3, enable_visualization=True):
        """
        初始化认知延迟模块
        
        Args:
            delay_steps (int): 延迟步数，模拟从决策到执行的时间间隔
            enable_smoothing (bool): 是否启用动作平滑
            smoothing_factor (float): 平滑系数，范围[0,1]，值越大平滑效果越强
            enable_visualization (bool): 是否启用可视化记录
        """
        self.delay_steps = delay_steps
        self.enable_smoothing = enable_smoothing
        self.smoothing_factor = smoothing_factor
        self.enable_visualization = enable_visualization
        
        # 延迟缓冲区，使用双端队列实现FIFO
        self.buffer = deque(maxlen=delay_steps + 1)
        
        # 记录上一个动作用于平滑
        self.previous_action = np.array([0.0, 0.0])
        
        # 用于可视化和调试的内部状态
        self._last_commanded_action = np.array([0.0, 0.0])
        
        # 可视化数据记录
        if self.enable_visualization:
            self._reset_visualization_data()
        
    def _reset_visualization_data(self):
        """重置可视化数据记录"""
        self.visualization_data = {
            'step': [],
            'original_action_throttle': [],
            'original_action_steering': [],
            'smoothed_action_throttle': [],
            'smoothed_action_steering': [],
            'delayed_action_throttle': [],
            'delayed_action_steering': [],
            'is_ppo_mode': [],
            'buffer_size': []
        }
        self._step_count = 0
        
    def process_action(self, action, is_ppo_mode=False):
        """
        处理动作，仅在PPO模式下应用延迟
        
        Args:
            action: 原始动作，通常是[油门/刹车, 转向]
            is_ppo_mode: 是否为PPO专家模式
            
        Returns:
            处理后的动作（如果非PPO模式则直接返回原始动作）
        """
        if not is_ppo_mode:
            # 记录可视化数据（非PPO模式）
            if self.enable_visualization:
                self._record_visualization_data(
                    original_action=action,
                    smoothed_action=action,
                    delayed_action=action,
                    is_ppo_mode=False
                )
            return action  # 非PPO模式直接返回原始动作
            
        # 确保action是numpy数组
        action = np.array(action, dtype=np.float32)
        
        # 记录原始命令供可视化使用
        self._last_commanded_action = action.copy()

        # 动作平滑（如果启用）
        if self.enable_smoothing:
            smoothed_action = (
                self.smoothing_factor * action +
                (1 - self.smoothing_factor) * self.previous_action
            )
            self.previous_action = smoothed_action.copy()
        else:
            smoothed_action = action
        
        # 添加到延迟缓冲区
        self.buffer.append(smoothed_action.copy())
        
        # 获取延迟后的动作
        if len(self.buffer) <= self.delay_steps:
            # 初始几步返回零动作（冷启动阶段）
            delayed_action = np.array([0.0, 0.0])
        else:
            # 返回延迟的动作（队列头部的动作）
            delayed_action = self.buffer[0]
        
        # 记录可视化数据（PPO模式）
        if self.enable_visualization:
            self._record_visualization_data(
                original_action=action,
                smoothed_action=smoothed_action,
                delayed_action=delayed_action,
                is_ppo_mode=True
            )
        
        return delayed_action
    
    def _record_visualization_data(self, original_action, smoothed_action, delayed_action, is_ppo_mode):
        """记录可视化数据"""
        self.visualization_data['step'].append(self._step_count)
        self.visualization_data['original_action_throttle'].append(float(original_action[0]))
        self.visualization_data['original_action_steering'].append(float(original_action[1]))
        self.visualization_data['smoothed_action_throttle'].append(float(smoothed_action[0]))
        self.visualization_data['smoothed_action_steering'].append(float(smoothed_action[1]))
        self.visualization_data['delayed_action_throttle'].append(float(delayed_action[0]))
        self.visualization_data['delayed_action_steering'].append(float(delayed_action[1]))
        self.visualization_data['is_ppo_mode'].append(is_ppo_mode)
        self.visualization_data['buffer_size'].append(len(self.buffer))
        
        self._step_count += 1
    
    def reset(self):
        """重置延迟模块状态"""
        self.buffer.clear()
        self.previous_action = np.array([0.0, 0.0])
        self._last_commanded_action = np.array([0.0, 0.0])
        
        # 重置可视化数据
        if self.enable_visualization:
            self._reset_visualization_data()
    
    def get_status(self):
        """
        获取模块当前状态信息
        
        Returns:
            dict: 包含模块状态的字典
        """
        status = {
            'delay_steps': self.delay_steps,
            'enable_smoothing': self.enable_smoothing,
            'smoothing_factor': self.smoothing_factor,
            'buffer_size': len(self.buffer),
            'last_commanded_action': self._last_commanded_action.tolist(),
            'previous_action': self.previous_action.tolist(),
            'enable_visualization': self.enable_visualization
        }
        
        if self.enable_visualization:
            status['recorded_steps'] = len(self.visualization_data['step'])
            
        return status
    
    def update_config(self, **kwargs):
        """
        动态更新模块配置
        
        Args:
            **kwargs: 配置参数
        """
        if 'delay_steps' in kwargs:
            old_delay = self.delay_steps
            self.delay_steps = kwargs['delay_steps']
            # 调整缓冲区大小
            new_buffer = deque(maxlen=self.delay_steps + 1)
            for item in self.buffer:
                new_buffer.append(item)
            self.buffer = new_buffer
            print(f"延迟步数已更新: {old_delay} -> {self.delay_steps}")
            
        if 'enable_smoothing' in kwargs:
            self.enable_smoothing = kwargs['enable_smoothing']
            print(f"动作平滑已{'启用' if self.enable_smoothing else '禁用'}")
            
        if 'smoothing_factor' in kwargs:
            self.smoothing_factor = kwargs['smoothing_factor']
            print(f"平滑系数已更新: {self.smoothing_factor}")
            
        if 'enable_visualization' in kwargs:
            self.enable_visualization = kwargs['enable_visualization']
            if self.enable_visualization and not hasattr(self, 'visualization_data'):
                self._reset_visualization_data()
            print(f"可视化记录已{'启用' if self.enable_visualization else '禁用'}")
    
    def generate_delay_visualization(self, output_dir=None, session_name=None):
        """
        生成延迟效果可视化图表
        
        Args:
            output_dir (str): 输出目录路径
            session_name (str): 会话名称
            
        Returns:
            str: 保存的图片文件路径
        """
        if not self.enable_visualization or not self.visualization_data['step']:
            print("❌ 没有可视化数据可供生成图表")
            return None
        
        # 设置输出目录
        if output_dir is None:
            # 默认使用带时间戳的认知分析目录下的延迟子目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = "/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/fig_cog"
            output_dir = os.path.join(base_dir, f"cognitive_analysis_{timestamp}", "cognitive_delay")
        
        if session_name is None:
            session_name = f"delay_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取数据
        steps = np.array(self.visualization_data['step'])
        original_throttle = np.array(self.visualization_data['original_action_throttle'])
        original_steering = np.array(self.visualization_data['original_action_steering'])
        smoothed_throttle = np.array(self.visualization_data['smoothed_action_throttle'])
        smoothed_steering = np.array(self.visualization_data['smoothed_action_steering'])
        delayed_throttle = np.array(self.visualization_data['delayed_action_throttle'])
        delayed_steering = np.array(self.visualization_data['delayed_action_steering'])
        is_ppo_mode = np.array(self.visualization_data['is_ppo_mode'])
        
        # 创建图表
        plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        try:
            fig.suptitle(f'Cognitive Delay Module Effect Analysis - {session_name}', fontsize=16, fontweight='bold')
        except:
            fig.suptitle(f'Cognitive Delay Module Analysis - {session_name}', fontsize=16, fontweight='bold')
        
        # 第一个子图：油门/刹车动作对比
        ax1 = axes[0, 0]
        ax1.plot(steps, original_throttle, 'b-', linewidth=2, label='Original Action', alpha=0.8)
        if self.enable_smoothing:
            ax1.plot(steps, smoothed_throttle, 'g--', linewidth=1.5, label='Smoothed Action', alpha=0.7)
        ax1.plot(steps, delayed_throttle, 'r-', linewidth=2, label=f'Delayed Action (Delay {self.delay_steps} Steps)', alpha=0.8)
        
        # 标记PPO模式区域
        ppo_regions = np.where(is_ppo_mode)[0]
        if len(ppo_regions) > 0:
            ax1.fill_between(steps, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                           where=is_ppo_mode, alpha=0.1, color='yellow', label='PPO Mode')
        
        try:
            ax1.set_title('Throttle/Brake Control Action Time Series', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Simulation Steps')
            ax1.set_ylabel('Action Value')
        except:
            ax1.set_title('Throttle/Brake Action Time Series', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Simulation Steps')
            ax1.set_ylabel('Action Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 第二个子图：转向动作对比
        ax2 = axes[0, 1]
        ax2.plot(steps, original_steering, 'b-', linewidth=2, label='Original Action', alpha=0.8)
        if self.enable_smoothing:
            ax2.plot(steps, smoothed_steering, 'g--', linewidth=1.5, label='Smoothed Action', alpha=0.7)
        ax2.plot(steps, delayed_steering, 'r-', linewidth=2, label=f'Delayed Action (Delay {self.delay_steps} Steps)', alpha=0.8)
        
        # 标记PPO模式区域
        if len(ppo_regions) > 0:
            ax2.fill_between(steps, ax2.get_ylim()[0], ax2.get_ylim()[1], 
                           where=is_ppo_mode, alpha=0.1, color='yellow', label='PPO Mode')
        
        try:
            ax2.set_title('Steering Control Action Time Series', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Simulation Steps')
            ax2.set_ylabel('Action Value')
        except:
            ax2.set_title('Steering Action Time Series', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Simulation Steps')
            ax2.set_ylabel('Action Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 第三个子图：延迟效果分析
        ax3 = axes[1, 0]
        # 计算延迟差异
        throttle_delay_diff = np.abs(original_throttle - delayed_throttle)
        steering_delay_diff = np.abs(original_steering - delayed_steering)
        
        ax3.plot(steps, throttle_delay_diff, 'r-', linewidth=2, label='Throttle Delay Difference', alpha=0.8)
        ax3.plot(steps, steering_delay_diff, 'b-', linewidth=2, label='Steering Delay Difference', alpha=0.8)
        
        # 标记PPO模式区域
        if len(ppo_regions) > 0:
            ax3.fill_between(steps, 0, ax3.get_ylim()[1], 
                           where=is_ppo_mode, alpha=0.1, color='yellow', label='PPO Mode')
        
        try:
            ax3.set_title('Delay Effect Difference Analysis', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Simulation Steps')
            ax3.set_ylabel('|Original Action - Delayed Action|')
        except:
            ax3.set_title('Delay Effect Difference Analysis', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Simulation Steps')
            ax3.set_ylabel('|Original - Delayed Action|')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 第四个子图：统计信息
        ax4 = axes[1, 1]
        ax4.axis('off')  # 关闭坐标轴
        
        # 计算统计信息
        total_steps = len(steps)
        ppo_steps = np.sum(is_ppo_mode)
        avg_throttle_delay = np.mean(throttle_delay_diff[is_ppo_mode]) if ppo_steps > 0 else 0
        avg_steering_delay = np.mean(steering_delay_diff[is_ppo_mode]) if ppo_steps > 0 else 0
        max_throttle_delay = np.max(throttle_delay_diff) if len(throttle_delay_diff) > 0 else 0
        max_steering_delay = np.max(steering_delay_diff) if len(steering_delay_diff) > 0 else 0
        
        # 显示统计信息
        stats_text = f"""
Delay Module Statistics:

Configuration Parameters:
• Delay Steps: {self.delay_steps}
• Smoothing: {'Enabled' if self.enable_smoothing else 'Disabled'}
• Smoothing Factor: {self.smoothing_factor:.2f}

Running Statistics:
• Total Simulation Steps: {total_steps}
• PPO Mode Steps: {ppo_steps} ({ppo_steps/total_steps*100:.1f}%)
• Non-PPO Mode Steps: {total_steps - ppo_steps}

Delay Effects (PPO Mode):
• Average Throttle Delay: {avg_throttle_delay:.4f}
• Average Steering Delay: {avg_steering_delay:.4f}
• Maximum Throttle Delay: {max_throttle_delay:.4f}
• Maximum Steering Delay: {max_steering_delay:.4f}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 调整子图间距
        plt.tight_layout()
        
        # 保存图片
        output_file = os.path.join(output_dir, f"{session_name}_delay_analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 延迟效果可视化图表已保存: {output_file}")
        
        # 生成数据摘要报告
        self._generate_delay_report(output_dir, session_name, {
            'total_steps': total_steps,
            'ppo_steps': ppo_steps,
            'avg_throttle_delay': avg_throttle_delay,
            'avg_steering_delay': avg_steering_delay,
            'max_throttle_delay': max_throttle_delay,
            'max_steering_delay': max_steering_delay
        })
        
        return output_file
    
    def _generate_delay_report(self, output_dir, session_name, stats):
        """生成详细的延迟分析报告"""
        report_file = os.path.join(output_dir, f"{session_name}_delay_report.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"认知延迟模块分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"会话名称: {session_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("模块配置:\n")
            f.write(f"  延迟步数: {self.delay_steps}\n")
            f.write(f"  动作平滑: {'启用' if self.enable_smoothing else '禁用'}\n")
            f.write(f"  平滑系数: {self.smoothing_factor:.3f}\n")
            f.write(f"  可视化记录: {'启用' if self.enable_visualization else '禁用'}\n\n")
            
            f.write("运行统计:\n")
            f.write(f"  总仿真步数: {stats['total_steps']}\n")
            f.write(f"  PPO模式步数: {stats['ppo_steps']} ({stats['ppo_steps']/stats['total_steps']*100:.1f}%)\n")
            f.write(f"  非PPO模式步数: {stats['total_steps'] - stats['ppo_steps']}\n\n")
            
            f.write("延迟效果分析 (PPO模式):\n")
            f.write(f"  平均油门/刹车延迟差异: {stats['avg_throttle_delay']:.6f}\n")
            f.write(f"  平均转向延迟差异: {stats['avg_steering_delay']:.6f}\n")
            f.write(f"  最大油门/刹车延迟差异: {stats['max_throttle_delay']:.6f}\n")
            f.write(f"  最大转向延迟差异: {stats['max_steering_delay']:.6f}\n\n")
            
            f.write("说明:\n")
            f.write("  - 延迟差异 = |原始动作 - 延迟后动作|\n")
            f.write("  - 延迟效果仅在PPO专家模式下生效\n")
            f.write("  - 平滑效果可以减少动作的突变\n")
            f.write("  - 延迟模拟人类驾驶员的反应延迟\n")
        
        print(f"✅ 延迟分析报告已保存: {report_file}")
    
    def get_visualization_data(self):
        """
        获取可视化数据
        
        Returns:
            dict: 可视化数据字典
        """
        if not self.enable_visualization:
            return None
        return self.visualization_data.copy() 