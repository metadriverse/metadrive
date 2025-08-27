import numpy as np
import itertools
import random
from typing import Dict, Any, Tuple, List


class DiscreteCognitiveParameterSampler:
    """
    认知参数采样器（离散化版本）
    在给定范围内按密度生成参数候选集合，然后在离散组合中进行采样
    
    支持参数:
      - 视觉厌恶系数 bias_inverse_tta_coef
      - 感知噪声 sigma0, k
      - 动作延迟 delay_steps
    """
    
    def __init__(self,
                 update_steps: int = 5,
                 bias_inverse_tta_coef_range: Tuple[float, float] = (0.5, 2.0),
                 bias_inverse_tta_coef_density: int = 4,
                 perception_sigma0_range: Tuple[float, float] = (0.02, 0.20),
                 perception_sigma0_density: int = 4,
                 perception_k_range: Tuple[float, float] = (0.002, 0.01),
                 perception_k_density: int = 4,
                 delay_steps_range: Tuple[int, int] = (1, 3),
                 delay_steps_density: int = 3,
                 shuffle: bool = True,
                 enable_visualization: bool = True,
                 save_history: bool = True):
        """
        初始化
        
        Args:
            update_steps (int): 参数更新频率（环境步数）
            *_range: 参数取值范围 (min, max)
            *_density: 在范围内离散点的个数
            shuffle (bool): 是否打乱组合顺序
            enable_visualization (bool): 是否启用可视化记录
            save_history (bool): 是否保存参数采样历史
        """
        # 参数更新频率
        self.update_steps = update_steps
        
        # 参数采样范围和密度
        self.bias_inverse_tta_coef_range = bias_inverse_tta_coef_range
        self.perception_sigma0_range = perception_sigma0_range
        self.perception_k_range = perception_k_range
        self.delay_steps_range = delay_steps_range
        
        # 功能开关
        self.enable_visualization = enable_visualization
        self.save_history = save_history
        self.bias_inverse_tta_coef_values = np.linspace(*bias_inverse_tta_coef_range, bias_inverse_tta_coef_density)
        self.perception_sigma0_values = np.linspace(*perception_sigma0_range, perception_sigma0_density)
        self.perception_k_values = np.linspace(*perception_k_range, perception_k_density)
        self.delay_steps_values = np.linspace(*delay_steps_range, delay_steps_density, dtype=int)
        
        # 生成所有组合
        self.param_grid: List[Dict[str, Any]] = []
        for bias, sigma0, k, delay in itertools.product(
            self.bias_inverse_tta_coef_values,
            self.perception_sigma0_values,
            self.perception_k_values,
            self.delay_steps_values
        ):
            self.param_grid.append({
                'bias_inverse_tta_coef': float(bias),
                'perception_sigma0': float(sigma0),
                'perception_k': float(k),
                'delay_steps': int(delay)
            })
        
        # 打乱组合顺序
        if shuffle:
            random.shuffle(self.param_grid)
        
        self.index = 0
        
        # 当前参数（初始化为第一个组合）
        self.current_params = self.param_grid[0].copy() if self.param_grid else {}
        
        # 参数采样历史
        self.param_history = []
        self.step_history = []
        self.timestamp_history = []
        
        # 内部状态
        self._last_update_step = 0
        self._total_updates = 0
        
        # 记录初始参数
        if self.save_history:
            self._record_parameter_update(0, "initialization")
    
    def sample(self) -> Dict[str, Any]:
        """随机采样一个组合"""
        return random.choice(self.param_grid)
    
    def next(self) -> Dict[str, Any]:
        """顺序取下一个组合（循环）"""
        if self.index >= len(self.param_grid):
            self.index = 0
        params = self.param_grid[self.index]
        self.index += 1
        return params
    
    def all_combinations(self) -> List[Dict[str, Any]]:
        """返回所有参数组合"""
        return self.param_grid
    
    def size(self) -> int:
        """返回总组合数"""
        return len(self.param_grid)
    
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
        
        # 采样新参数（随机选择一个组合）
        self.current_params = self.sample()
        self._last_update_step = current_step
        self._total_updates += 1
        
        # 记录参数更新
        if self.save_history:
            self._record_parameter_update(current_step, "sampling")
        
        return self.current_params
    
    def _record_parameter_update(self, step: int, update_type: str):
        """记录参数更新历史"""
        from datetime import datetime
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
            'total_combinations': len(self.param_grid),
            
            'bias_inverse_tta_coef': {
                'mean': np.mean(bias_values),
                'std': np.std(bias_values),
                'min': np.min(bias_values),
                'max': np.max(bias_values),
                'range': self.bias_inverse_tta_coef_range,
                'discrete_values': self.bias_inverse_tta_coef_values.tolist()
            },
            
            'perception_sigma0': {
                'mean': np.mean(sigma0_values),
                'std': np.std(sigma0_values),
                'min': np.min(sigma0_values),
                'max': np.max(sigma0_values),
                'range': self.perception_sigma0_range,
                'discrete_values': self.perception_sigma0_values.tolist()
            },
            
            'perception_k': {
                'mean': np.mean(k_values),
                'std': np.std(k_values),
                'min': np.min(k_values),
                'max': np.max(k_values),
                'range': self.perception_k_range,
                'discrete_values': self.perception_k_values.tolist()
            },
            
            'delay_steps': {
                'mean': np.mean(delay_values),
                'std': np.std(delay_values),
                'min': np.min(delay_values),
                'max': np.max(delay_values),
                'range': self.delay_steps_range,
                'discrete_values': self.delay_steps_values.tolist(),
                'value_counts': {int(i): delay_values.count(i) for i in set(delay_values)}
            }
        }
        
        return stats
    
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

        from datetime import datetime
        history_data = {
            'sampler_config': {
                'sampler_type': 'discrete',
                'update_steps': self.update_steps,
                'bias_inverse_tta_coef_range': self.bias_inverse_tta_coef_range,
                'perception_sigma0_range': self.perception_sigma0_range,
                'perception_k_range': self.perception_k_range,
                'delay_steps_range': self.delay_steps_range,
                'total_combinations': len(self.param_grid)
            },
            'parameter_history': convert_numpy_types(self.param_history),
            'step_history': convert_numpy_types(self.step_history),
            'timestamp_history': self.timestamp_history,
            'statistics': convert_numpy_types(self.get_statistics()),
            'export_timestamp': datetime.now().isoformat()
        }

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(history_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # 如果完整保存失败，尝试保存简化版
            try:
                simplified_data = {
                    'sampler_config': history_data['sampler_config'],
                    'total_updates': len(self.param_history),
                    'current_parameters': convert_numpy_types(self.current_params),
                    'export_timestamp': datetime.now().isoformat()
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(simplified_data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
    
    def generate_parameter_visualization(self, output_dir: str = "cognitive_visualization", 
                                       session_name: str = None) -> str:
        """
        生成参数采样可视化图表
        
        Args:
            output_dir (str): 输出目录
            session_name (str): 会话名称
            
        Returns:
            str: 生成的图表文件路径
        """
        if not self.enable_visualization or len(self.param_history) < 2:
            return None
        
        try:
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime
            
            if session_name is None:
                session_name = f"discrete_sampling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建可视化图表
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Discrete Cognitive Parameter Sampling - {session_name}', fontsize=14)
            
            # 参数历史数据
            bias_values = [p['bias_inverse_tta_coef'] for p in self.param_history]
            sigma0_values = [p['perception_sigma0'] for p in self.param_history]
            k_values = [p['perception_k'] for p in self.param_history]
            delay_values = [p['delay_steps'] for p in self.param_history]
            
            # 子图1: bias_inverse_tta_coef
            axes[0, 0].plot(self.step_history, bias_values, 'o-', alpha=0.7)
            axes[0, 0].set_title('Bias Inverse TTA Coefficient')
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].grid(True)
            
            # 子图2: perception_sigma0
            axes[0, 1].plot(self.step_history, sigma0_values, 's-', alpha=0.7, color='orange')
            axes[0, 1].set_title('Perception Sigma0')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].grid(True)
            
            # 子图3: perception_k
            axes[1, 0].plot(self.step_history, k_values, '^-', alpha=0.7, color='green')
            axes[1, 0].set_title('Perception K')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].grid(True)
            
            # 子图4: delay_steps
            axes[1, 1].plot(self.step_history, delay_values, 'd-', alpha=0.7, color='red')
            axes[1, 1].set_title('Delay Steps')
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Steps')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 保存图表
            output_file = os.path.join(output_dir, f"{session_name}_parameter_sampling.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
            
        except Exception as e:
            return None
