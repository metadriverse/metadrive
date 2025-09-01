"""
动作处理器模块

该模块专门负责PPO策略的动作生成和处理。
从原始PPOCheckpointSimulator中提取动作处理相关功能。
"""

import torch
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .network import PPONetwork
    from .cognitive_module_manager import CognitiveModuleManager


class ActionProcessor:
    """PPO动作处理器
    
    专门负责：
    1. PPO策略的动作生成
    2. 认知参数的观测拼接
    3. 动作范围限制和后处理
    4. 与认知模块的协作
    """
    
    def __init__(self, network: 'PPONetwork', cognitive_manager: Optional['CognitiveModuleManager'] = None, device: str = "cpu"):
        """
        初始化动作处理器
        
        Args:
            network: PPO网络实例
            cognitive_manager: 认知模块管理器
            device: 计算设备
        """
        self.network = network
        self.cognitive_manager = cognitive_manager
        self.device = torch.device(device)
        
        # 确保网络在正确的设备上
        if hasattr(self.network, 'to'):
            self.network = self.network.to(self.device)
        
        # 确保网络处于评估模式
        if hasattr(self.network, 'eval'):
            self.network.eval()
        
        print(f"🎯 动作处理器初始化完成")
        print(f"   网络观测维度: {getattr(self.network, 'obs_dim', 'Unknown')}")
        print(f"   计算设备: {self.device}")
        print(f"   认知模块: {'启用' if cognitive_manager else '禁用'}")
    
    def get_action(self, observation: np.ndarray, deterministic: bool = True, env=None, step_count: int = 0) -> np.ndarray:
        """
        根据观察获取PPO策略动作
        
        Args:
            observation: 环境观察
            deterministic: 是否使用确定性策略
            env: 环境实例（用于认知模块处理）
            step_count: 当前步数（用于调试）
            
        Returns:
            动作数组 [steering, acceleration]
        """
        # 第一步：通过认知模块处理观测
        processed_obs = self._process_observation(observation, env, step_count)
        
        # 第二步：通过PPO网络获取原始动作
        raw_action = self._get_raw_action(processed_obs, deterministic, step_count)
        
        # 第三步：通过认知模块处理动作（延迟等）
        final_action = self._process_action(raw_action, step_count)
        
        # 第四步：动作范围限制
        final_action = self._clip_action(final_action)
        
        return final_action
    
    def _process_observation(self, observation: np.ndarray, env=None, step_count: int = 0) -> np.ndarray:
        """处理观测（认知感知模块 + 认知参数拼接）"""
        if self.cognitive_manager:
            # 认知感知模块已在传感器层处理噪声
            # 这里主要负责认知参数拼接
            processed_obs = self.cognitive_manager.process_observation(observation)
            
            # 调试信息（前几步）
            if step_count < 3:
                print(f"    观测处理: {observation.shape} → {processed_obs.shape}")
            
            return processed_obs
        else:
            return observation
    
    def _get_raw_action(self, observation: np.ndarray, deterministic: bool = True, step_count: int = 0) -> np.ndarray:
        """通过PPO网络获取原始动作"""
        # 转换为tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # 获取动作
        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(obs_tensor, deterministic=deterministic)
        
        # 转换回numpy
        action = action.cpu().numpy().flatten()
        
        # 调试信息（前几步）
        if step_count < 3:
            print(f"    网络输出: [{action[0]:.3f}, {action[1]:.3f}]")
        
        return action
    
    def _process_action(self, action: np.ndarray, step_count: int = 0) -> np.ndarray:
        """通过认知模块处理动作（延迟等）"""
        if self.cognitive_manager:
            processed_action, delay_applied = self.cognitive_manager.process_action(action)
            
            # 调试信息（前几步）
            if step_count < 3 and delay_applied:
                print(f"    延迟处理: [{action[0]:.3f}, {action[1]:.3f}] → [{processed_action[0]:.3f}, {processed_action[1]:.3f}]")
            
            return processed_action
        else:
            return action
    
    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """限制动作范围"""
        clipped_action = np.clip(action, -1.0, 1.0)
        return clipped_action
    
    def update_network(self, new_network: 'PPONetwork'):
        """更新网络实例"""
        self.network = new_network
        if hasattr(self.network, 'to'):
            self.network = self.network.to(self.device)
        if hasattr(self.network, 'eval'):
            self.network.eval()
        
        print(f"🔄 动作处理器网络已更新")
        print(f"   新网络观测维度: {getattr(self.network, 'obs_dim', 'Unknown')}")
    
    def update_cognitive_manager(self, new_cognitive_manager: Optional['CognitiveModuleManager']):
        """更新认知模块管理器"""
        self.cognitive_manager = new_cognitive_manager
        print(f"🔄 动作处理器认知管理器已更新: {'启用' if new_cognitive_manager else '禁用'}")
    
    def get_network_info(self) -> dict:
        """获取网络信息"""
        return {
            'obs_dim': getattr(self.network, 'obs_dim', 'Unknown'),
            'action_dim': getattr(self.network, 'action_dim', 'Unknown'),
            'device': str(self.device),
            'network_type': type(self.network).__name__ if self.network else 'None',
            'cognitive_enabled': self.cognitive_manager is not None
        } 