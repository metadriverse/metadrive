"""
检查点加载器模块

该模块专门负责PPO检查点的加载、权重处理和网络初始化。
从原始PPOCheckpointSimulator中提取检查点相关功能。
"""

import os
import torch
from typing import Optional
from pathlib import Path
import sys

# 添加网络模块路径
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from .network import PPONetwork


class CheckpointLoader:
    """PPO检查点加载器
    
    专门负责：
    1. 检查点文件的加载
    2. 观测维度的检测和转换
    3. 网络权重的适配（275维 ⟷ 283维）
    4. PPONetwork的初始化
    """
    
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """
        初始化检查点加载器
        
        Args:
            checkpoint_path: 检查点文件路径
            device: 计算设备
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.checkpoint = None
        self.network = None
        self._checkpoint_obs_dim = None
        
        # 验证检查点文件存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    def load_checkpoint(self, use_cognitive_modules: bool = False) -> PPONetwork:
        """
        加载检查点并创建网络
        
        Args:
            use_cognitive_modules: 是否使用认知模块（影响观测维度）
            
        Returns:
            加载权重后的PPONetwork实例
        """
        print(f"📥 开始加载检查点: {os.path.basename(self.checkpoint_path)}")
        
        # 加载检查点数据
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # 检测检查点中的观测维度
        self._detect_checkpoint_obs_dim()
        
        # 确定目标观测维度
        target_obs_dim = 283 if use_cognitive_modules else 275
        
        # 创建网络
        self.network = PPONetwork(use_cognitive_modules=use_cognitive_modules).to(self.device)
        
        # 处理维度转换（如果需要）
        self._handle_dimension_conversion(target_obs_dim)
        
        # 加载权重
        self._load_network_weights()
        
        # 设置为评估模式
        self.network.eval()
        
        print(f"✅ 检查点加载完成")
        print(f"🎯 训练迭代: {self.checkpoint.get('iteration', 'Unknown')}")
        print(f"🚀 全局步数: {self.checkpoint.get('global_step', 'Unknown')}")
        
        return self.network
    
    def _detect_checkpoint_obs_dim(self):
        """检测检查点中的观测维度"""
        if 'network_state_dict' not in self.checkpoint:
            print("⚠️ 检查点中未找到network_state_dict")
            return
        
        # 从第一个线性层的权重推断观测维度
        first_layer_key = None
        for key in self.checkpoint['network_state_dict'].keys():
            if 'actor_fc1.weight' in key:
                first_layer_key = key
                break
        
        if first_layer_key:
            self._checkpoint_obs_dim = self.checkpoint['network_state_dict'][first_layer_key].shape[1]
            print(f"🔍 检查点检测到观测维度: {self._checkpoint_obs_dim}")
        else:
            print("⚠️ 无法从检查点推断观测维度")
    
    def _handle_dimension_conversion(self, target_obs_dim: int):
        """处理观测维度转换"""
        if not self._checkpoint_obs_dim:
            print("🔧 无法检测检查点维度，跳过转换")
            return
        
        if self._checkpoint_obs_dim == target_obs_dim:
            print(f"✅ 维度匹配: {target_obs_dim}")
            return
        
        print(f"⚠️ 维度不匹配，检查点: {self._checkpoint_obs_dim}维, 目标: {target_obs_dim}维")
        
        if self._checkpoint_obs_dim == 275 and target_obs_dim == 283:
            print("🔧 275维 → 283维: 认知参数部分将随机初始化")
            self._extend_weights_275_to_283()
        elif self._checkpoint_obs_dim == 283 and target_obs_dim == 275:
            print("🔧 283维 → 275维: 截取前275维权重")
            self._truncate_weights_283_to_275()
        else:
            print(f"❌ 不支持的维度转换: {self._checkpoint_obs_dim} → {target_obs_dim}")
            print("🔧 将使用随机初始化的权重")
    
    def _extend_weights_275_to_283(self):
        """扩展275维度权重到283维度（为认知参数部分随机初始化）"""
        checkpoint_weights = self.checkpoint['network_state_dict']
        
        # 扩展第一层权重（为新增的8维随机初始化）
        for layer_name in ['actor_fc1.weight', 'critic_fc1.weight']:
            if layer_name in checkpoint_weights:
                old_weight = checkpoint_weights[layer_name]  # [hidden_dim, 275]
                hidden_dim = old_weight.shape[0]
                
                # 为认知参数和mask部分随机初始化权重
                additional_weights = torch.randn(hidden_dim, 8) * 0.1  # 8 = 4参数 + 4mask
                new_weight = torch.cat([old_weight, additional_weights], dim=1)  # [hidden_dim, 283]
                
                checkpoint_weights[layer_name] = new_weight
                print(f"   {layer_name}: {old_weight.shape} → {new_weight.shape} (认知参数部分随机初始化)")
        
        # 更新检查点
        self.checkpoint['network_state_dict'] = checkpoint_weights
    
    def _truncate_weights_283_to_275(self):
        """截取283维度权重到275维度（移除认知参数部分）"""
        checkpoint_weights = self.checkpoint['network_state_dict']
        
        # 截取第一层权重（只保留前275维）
        for layer_name in ['actor_fc1.weight', 'critic_fc1.weight']:
            if layer_name in checkpoint_weights:
                old_weight = checkpoint_weights[layer_name]  # [hidden_dim, 283]
                new_weight = old_weight[:, :275]  # 截取前275维 [hidden_dim, 275]
                
                checkpoint_weights[layer_name] = new_weight
                print(f"   {layer_name}: {old_weight.shape} → {new_weight.shape} (移除认知参数部分)")
        
        # 更新检查点
        self.checkpoint['network_state_dict'] = checkpoint_weights
    
    def _load_network_weights(self):
        """加载网络权重"""
        try:
            self.network.load_state_dict(self.checkpoint['network_state_dict'])
            print("✅ 权重加载成功")
        except Exception as e:
            print(f"⚠️ 权重加载失败: {e}")
            print("🔧 将使用随机初始化的权重")
    
    def get_network(self) -> Optional[PPONetwork]:
        """获取加载的网络实例"""
        return self.network
    
    def get_checkpoint_info(self) -> dict:
        """获取检查点信息"""
        if not self.checkpoint:
            return {}
        
        return {
            'iteration': self.checkpoint.get('iteration', 'Unknown'),
            'global_step': self.checkpoint.get('global_step', 'Unknown'),
            'checkpoint_obs_dim': self._checkpoint_obs_dim,
            'checkpoint_path': self.checkpoint_path
        } 