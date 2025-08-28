"""
认知参数采样器 - 独立实现
在训练过程中动态随机采样认知模块的参数
支持视觉厌恶、感知噪声、动作延迟等参数的随机化
"""

import numpy as np
import random
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import json
import os


class CognitiveParameterSampler:
    """
    认知参数采样器 - 独立实现，不依赖gym环境
    
    该模块在训练过程中动态随机采样认知模块的参数：
    1. 视觉厌恶参数：bias_inverse_tta_coef
    2. 感知噪声参数：sigma0, k
    3. 动作延迟参数：delay_steps
    4. 支持参数范围配置和更新频率控制
    5. 记录参数采样历史用于分析和可视化
    """
    
    def __init__(self, 
                 update_steps: int = 1000,
                 bias_inverse_tta_coef_range: Tuple[float, float] = (0.5, 2.0),
                 perception_sigma0_range: Tuple[float, float] = (0.02, 0.20),
                 perception_k_range: Tuple[float, float] = (0.002, 0.01),
                 delay_steps_range: Tuple[int, int] = (0, 3),
                 enable_visualization: bool = True,
                 save_history: bool = True):
        """
        初始化认知参数采样器
        
        Args:
            update_steps (int): 参数更新频率（环境步数）
            bias_inverse_tta_coef_range (tuple): 视觉厌恶系数范围 [min, max]
            perception_sigma0_range (tuple): 感知噪声标准差范围 [min, max]
            perception_k_range (tuple): 距离相关系数范围 [min, max]
            delay_steps_range (tuple): 动作延迟步数范围 [min, max]
            enable_visualization (bool): 是否启用可视化记录
            save_history (bool): 是否保存参数采样历史
        """
        # 参数更新频率
        self.update_steps = update_steps
        
        # 参数采样范围
        self.bias_inverse_tta_coef_range = bias_inverse_tta_coef_range
        self.perception_sigma0_range = perception_sigma0_range
        self.perception_k_range = perception_k_range
        self.delay_steps_range = delay_steps_range
        
        # 当前参数值
        self.current_params = self._sample_initial_parameters()
        
        # 参数采样历史
        self.param_history = []
        self.step_history = []
        self.timestamp_history = []
        
        # 功能开关
        self.enable_visualization = enable_visualization
        self.save_history = save_history
        
        # 内部状态
        self._last_update_step = 0
        self._total_updates = 0
        
        # 记录初始参数
        if self.save_history:
            self._record_parameter_update(0, "initialization")
    
    def _sample_initial_parameters(self) -> Dict[str, Any]:
        """采样初始参数"""
        return {
            'bias_inverse_tta_coef': np.random.uniform(*self.bias_inverse_tta_coef_range),
            'perception_sigma0': np.random.uniform(*self.perception_sigma0_range),
            'perception_k': np.random.uniform(*self.perception_k_range),
            'delay_steps': np.random.randint(*self.delay_steps_range)
        }
    
    def should_update_parameters(self, current_step: int) -> bool:
        """
        检查是否应该更新参数
        
        Args:
            current_step (int): 当前环境步数
            
        Returns:
            bool: 是否需要更新参数
        """
        return current_step - self._last_update_step >= self.update_steps
    
    def update_parameters(self, current_step: int, force_update: bool = False) -> Dict[str, Any]:
        """
        更新认知参数
        
        Args:
            current_step (int): 当前环境步数
            force_update (bool): 是否强制更新
            
        Returns:
            Dict[str, Any]: 更新后的参数字典
        """
        if not force_update and not self.should_update_parameters(current_step):
            return self.current_params
        
        # 采样新参数
        new_params = {
            'bias_inverse_tta_coef': np.random.uniform(*self.bias_inverse_tta_coef_range),
            'perception_sigma0': np.random.uniform(*self.perception_sigma0_range),
            'perception_k': np.random.uniform(*self.perception_k_range),
            'delay_steps': np.random.randint(*self.delay_steps_range)
        }
        
        # 更新当前参数
        self.current_params = new_params
        self._last_update_step = current_step
        self._total_updates += 1
        
        # 记录参数更新
        if self.save_history:
            self._record_parameter_update(current_step, "sampling")
        
        return self.current_params
    
    def _record_parameter_update(self, step: int, update_type: str):
        """记录参数更新历史"""
        timestamp = datetime.now().isoformat()
        
        self.param_history.append(self.current_params.copy())
        self.step_history.append(step)
        self.timestamp_history.append(timestamp)
        
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        获取当前参数值
        
        Returns:
            Dict[str, Any]: 当前参数字典
        """
        return self.current_params.copy()
    
    def get_parameter_history(self) -> Dict[str, list]:
        """
        获取参数采样历史
        
        Returns:
            Dict[str, list]: 包含参数历史、步数历史和时间戳的字典
        """
        return {
            'parameters': self.param_history,
            'steps': self.step_history,
            'timestamps': self.timestamp_history
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取参数采样统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        if not self.param_history:
            return {}
        
        # 计算各参数的统计信息
        bias_values = [p['bias_inverse_tta_coef'] for p in self.param_history]
        sigma0_values = [p['perception_sigma0'] for p in self.param_history]
        k_values = [p['perception_k'] for p in self.param_history]
        delay_values = [p['delay_steps'] for p in self.param_history]
        
        stats = {
            'total_updates': self._total_updates,
            'last_update_step': self._last_update_step,
            'update_frequency': self.update_steps,
            
            'bias_inverse_tta_coef': {
                'mean': np.mean(bias_values),
                'std': np.std(bias_values),
                'min': np.min(bias_values),
                'max': np.max(bias_values),
                'range': self.bias_inverse_tta_coef_range
            },
            
            'perception_sigma0': {
                'mean': np.mean(sigma0_values),
                'std': np.std(sigma0_values),
                'min': np.min(sigma0_values),
                'max': np.max(sigma0_values),
                'range': self.perception_sigma0_range
            },
            
            'perception_k': {
                'mean': np.mean(k_values),
                'std': np.std(k_values),
                'min': np.min(k_values),
                'max': np.max(k_values),
                'range': self.perception_k_range
            },
            
            'delay_steps': {
                'mean': np.mean(delay_values),
                'std': np.std(delay_values),
                'min': np.min(delay_values),
                'max': np.max(delay_values),
                'range': self.delay_steps_range,
                'value_counts': {i: delay_values.count(i) for i in set(delay_values)}
            }
        }
        
        return stats
    
    def update_config(self, **kwargs):
        """
        动态更新采样器配置
        
        Args:
            **kwargs: 配置参数
        """
        if 'update_steps' in kwargs:
            old_steps = self.update_steps
            self.update_steps = kwargs['update_steps']

            
        if 'bias_inverse_tta_coef_range' in kwargs:
            self.bias_inverse_tta_coef_range = kwargs['bias_inverse_tta_coef_range']
            
            
        if 'perception_sigma0_range' in kwargs:
            self.perception_sigma0_range = kwargs['perception_sigma0_range']

            
        if 'perception_k_range' in kwargs:
            self.perception_k_range = kwargs['perception_k_range']

            
        if 'delay_steps_range' in kwargs:
            self.delay_steps_range = kwargs['delay_steps_range']

    
    def reset(self):
        """重置采样器状态"""
        self._last_update_step = 0
        self._total_updates = 0
        
        # 重新采样初始参数
        self.current_params = self._sample_initial_parameters()
        
        # 清空历史记录
        if not self.save_history:
            self.param_history.clear()
            self.step_history.clear()
            self.timestamp_history.clear()
        

    
    def save_history_to_file(self, file_path: str):
        """
        保存参数采样历史到文件
        
        Args:
            file_path (str): 保存文件路径
        """
        if not self.save_history or not self.param_history:
 
            return

        # 转换numpy类型为Python原生类型，确保JSON序列化成功
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy标量类型
                return obj.item()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj

        history_data = {
            'sampler_config': {
                'update_steps': self.update_steps,
                'bias_inverse_tta_coef_range': self.bias_inverse_tta_coef_range,
                'perception_sigma0_range': self.perception_sigma0_range,
                'perception_k_range': self.perception_k_range,
                'delay_steps_range': self.delay_steps_range
            },
            'parameter_history': convert_numpy_types(self.param_history),
            'step_history': convert_numpy_types(self.step_history),
            'timestamp_history': self.timestamp_history,
            'statistics': convert_numpy_types(self.get_statistics()),
            'export_timestamp': datetime.now().isoformat()
        }

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            print(f"参数采样历史已保存: {file_path}")
        except Exception as e:
            print(f"保存参数采样历史失败: {e}")
            # 尝试更简单的保存方式
            try:
                simple_history = {
                    'total_updates': self._total_updates,
                    'last_update_step': self._last_update_step,
                    'current_parameters': convert_numpy_types(self.current_params),
                    'export_timestamp': datetime.now().isoformat()
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(simple_history, f, indent=2, ensure_ascii=False)
                print(f"简化版参数历史已保存: {file_path}")
            except Exception as e2:
                print(f"简化版保存也失败: {e2}")
    
    def load_history_from_file(self, file_path: str):
        """
        从文件加载参数采样历史
        
        Args:
            file_path (str): 加载文件路径
        """
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            # 恢复配置
            if 'sampler_config' in history_data:
                config = history_data['sampler_config']
                self.update_steps = config.get('update_steps', self.update_steps)
                self.bias_inverse_tta_coef_range = config.get('bias_inverse_tta_coef_range', self.bias_inverse_tta_coef_range)
                self.perception_sigma0_range = config.get('perception_sigma0_range', self.perception_sigma0_range)
                self.perception_k_range = config.get('perception_k_range', self.perception_k_range)
                self.delay_steps_range = config.get('delay_steps_range', self.delay_steps_range)
            
            # 恢复历史数据
            if 'parameter_history' in history_data:
                self.param_history = history_data['parameter_history']
                self.step_history = history_data['step_history']
                self.timestamp_history = history_data['timestamp_history']
                
                # 恢复当前参数为最后一次采样的参数
                if self.param_history:
                    self.current_params = self.param_history[-1].copy()
                    self._last_update_step = self.step_history[-1] if self.step_history else 0
                    self._total_updates = len(self.param_history)
            
            print(f"参数采样历史已加载: {file_path}")
            print(f"历史记录数: {len(self.param_history)}")
            print(f"最后更新步数: {self._last_update_step}")
            
        except Exception as e:
            print(f"加载参数采样历史失败: {e}")
    
    def generate_parameter_visualization(self, output_dir: str = None, session_name: str = None):
        """
        生成参数采样可视化图表
        
        Args:
            output_dir (str): 输出目录路径
            session_name (str): 会话名称
            
        Returns:
            str: 保存的图片文件路径
        """
        if not self.enable_visualization or not self.param_history:
            print("没有可视化数据可供生成图表")
            return None
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib未安装，无法生成可视化图表")
            return None
        
        try:
            # 设置输出目录
            if output_dir is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_dir = "/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/fig_cog"
                output_dir = os.path.join(base_dir, f"cognitive_analysis_{timestamp}", "parameter_sampling")
            
            if session_name is None:
                session_name = f"parameter_sampling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 提取数据
            steps = np.array(self.step_history)
            bias_values = [p['bias_inverse_tta_coef'] for p in self.param_history]
            sigma0_values = [p['perception_sigma0'] for p in self.param_history]
            k_values = [p['perception_k'] for p in self.param_history]
            delay_values = [p['delay_steps'] for p in self.param_history]
            
            # 创建图表
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Cognitive Parameter Sampling Analysis - {session_name}', fontsize=16, fontweight='bold')
            
            # 第一个子图：视觉厌恶系数变化
            ax1 = axes[0, 0]
            ax1.plot(steps, bias_values, 'b-o', linewidth=2, markersize=6, alpha=0.8)
            ax1.axhline(y=self.bias_inverse_tta_coef_range[0], color='r', linestyle='--', alpha=0.5, label='Min Range')
            ax1.axhline(y=self.bias_inverse_tta_coef_range[1], color='r', linestyle='--', alpha=0.5, label='Max Range')
            ax1.set_title('Visual Aversion Coefficient (bias_inverse_tta_coef)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Simulation Steps')
            ax1.set_ylabel('Coefficient Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 第二个子图：感知噪声标准差变化
            ax2 = axes[0, 1]
            ax2.plot(steps, sigma0_values, 'g-o', linewidth=2, markersize=6, alpha=0.8)
            ax2.axhline(y=self.perception_sigma0_range[0], color='r', linestyle='--', alpha=0.5, label='Min Range')
            ax2.axhline(y=self.perception_sigma0_range[1], color='r', linestyle='--', alpha=0.5, label='Max Range')
            ax2.set_title('Perception Noise Standard Deviation (sigma0)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Simulation Steps')
            ax2.set_ylabel('Sigma Value (m)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 第三个子图：距离相关系数变化
            ax3 = axes[1, 0]
            ax3.plot(steps, k_values, 'm-o', linewidth=2, markersize=6, alpha=0.8)
            ax3.axhline(y=self.perception_k_range[0], color='r', linestyle='--', alpha=0.5, label='Min Range')
            ax3.axhline(y=self.perception_k_range[1], color='r', linestyle='--', alpha=0.5, label='Max Range')
            ax3.set_title('Distance Correlation Coefficient (k)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Simulation Steps')
            ax3.set_ylabel('k Value')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 第四个子图：动作延迟步数变化
            ax4 = axes[1, 1]
            ax4.plot(steps, delay_values, 'c-o', linewidth=2, markersize=6, alpha=0.8)
            ax4.axhline(y=self.delay_steps_range[0], color='r', linestyle='--', alpha=0.5, label='Min Range')
            ax4.axhline(y=self.delay_steps_range[1], color='r', linestyle='--', alpha=0.5, label='Max Range')
            ax4.set_title('Action Delay Steps (delay_steps)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Simulation Steps')
            ax4.set_ylabel('Delay Steps')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 调整子图间距
            plt.tight_layout()
            
            # 保存图片
            output_file = os.path.join(output_dir, f"{session_name}_parameter_analysis.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"参数采样可视化图表已保存: {output_file}")
            
            # 生成参数分析报告
            self._generate_parameter_report(output_dir, session_name)
            
            return output_file
            
        except Exception as e:
            print(f"生成参数采样可视化失败: {e}")
            return None
    
    def _generate_parameter_report(self, output_dir: str, session_name: str):
        """生成详细的参数采样分析报告"""
        report_file = os.path.join(output_dir, f"{session_name}_parameter_report.txt")
        
        stats = self.get_statistics()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"认知参数采样分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"会话名称: {session_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("采样器配置:\n")
            f.write(f"  参数更新频率: {self.update_steps} 步\n")
            f.write(f"  总更新次数: {stats.get('total_updates', 0)}\n")
            f.write(f"  最后更新步数: {stats.get('last_update_step', 0)}\n\n")
            
            f.write("参数范围配置:\n")
            f.write(f"  视觉厌恶系数: {self.bias_inverse_tta_coef_range}\n")
            f.write(f"  感知噪声标准差: {self.perception_sigma0_range} (米)\n")
            f.write(f"  距离相关系数: {self.perception_k_range}\n")
            f.write(f"  动作延迟步数: {self.delay_steps_range}\n\n")
            
            f.write("参数采样统计:\n")
            for param_name, param_stats in stats.items():
                if isinstance(param_stats, dict) and 'mean' in param_stats:
                    f.write(f"  {param_name}:\n")
                    f.write(f"    均值: {param_stats['mean']:.6f}\n")
                    f.write(f"    标准差: {param_stats['std']:.6f}\n")
                    f.write(f"    最小值: {param_stats['min']:.6f}\n")
                    f.write(f"    最大值: {param_stats['max']:.6f}\n")
                    f.write(f"    配置范围: {param_stats['range']}\n")
                    if 'value_counts' in param_stats:
                        f.write(f"    值分布: {param_stats['value_counts']}\n")
                    f.write("\n")
            
            f.write("说明:\n")
            f.write("  - 参数在指定范围内均匀随机采样\n")
            f.write("  - 每{self.update_steps}步更新一次参数\n")
            f.write("  - 参数更新后立即应用到相应的认知模块\n")
            f.write("  - 支持参数历史记录和可视化分析\n")
        
        print(f"参数采样分析报告已保存: {report_file}")
    
    def get_status(self):
        """
        获取采样器当前状态信息
        
        Returns:
            dict: 包含采样器状态的字典
        """
        return {
            'update_steps': self.update_steps,
            'current_parameters': self.current_params.copy(),
            'total_updates': self._total_updates,
            'last_update_step': self._last_update_step,
            'enable_visualization': self.enable_visualization,
            'save_history': self.save_history,
            'history_count': len(self.param_history)
        } 