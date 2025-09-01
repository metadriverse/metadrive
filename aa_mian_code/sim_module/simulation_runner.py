"""
仿真运行器模块

该模块是整个仿真系统的主控制器，协调所有子模块完成完整的仿真流程。
这是重构后的PPOCheckpointSimulator的现代化替代品。
"""

import os
import json
import argparse
import time
from typing import Dict, List, Optional
from pathlib import Path

# 导入所有子模块
from .checkpoint_loader import CheckpointLoader
from .lane_change_tracker import LaneChangeTracker
from .environment_factory import EnvironmentFactory
from .cognitive_module_manager import CognitiveModuleManager
from .action_processor import ActionProcessor
from .episode_statistics import EpisodeStatistics
from .visualization_manager import VisualizationManager


class SimulationRunner:
    """仿真运行器 - 主控制器
    
    这是重构后的PPOCheckpointSimulator的现代化替代品。
    专门负责：
    1. 协调所有子模块完成仿真流程
    2. 管理Episode的完整生命周期
    3. 提供简洁的外部接口
    4. 保持向后兼容性
    """
    
    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None, 
                 device: str = "auto", args: Optional[argparse.Namespace] = None):
        """
        初始化仿真运行器
        
        Args:
            checkpoint_path: 检查点文件路径
            config_path: 配置文件路径（可选）
            device: 计算设备
            args: 命令行参数
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.args = args
        
        print(f"🚀 仿真运行器初始化开始...")
        print(f"📁 检查点: {os.path.basename(checkpoint_path)}")
        print(f"🔧 设备: {device}")
        
        # 第一步：加载配置
        self.config = self._load_config(config_path)
        
        # 第二步：初始化子模块
        self._initialize_modules()
        
        # 第三步：加载检查点并创建网络
        self._load_checkpoint_and_network()
        
        # 第四步：创建动作处理器（需要网络和认知管理器）
        self._create_action_processor()
        
        print(f"✅ 仿真运行器初始化完成")
        self._print_initialization_summary()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"📋 配置已从文件加载: {config_path}")
        else:
            # 使用环境工厂的默认配置
            env_factory = EnvironmentFactory()
            config = env_factory.get_config()
            print(f"📋 使用默认配置")
        
        return config
    
    def _initialize_modules(self):
        """初始化所有子模块"""
        print(f"🔧 初始化子模块...")
        
        # 1. 变道跟踪器
        self.lane_change_tracker = LaneChangeTracker(self.args)
        
        # 2. 环境工厂
        self.environment_factory = EnvironmentFactory(self.config, self.args)
        
        # 3. 认知模块管理器（如果启用）
        if self.args and getattr(self.args, 'use_cognitive_modules', False):
            self.cognitive_manager = CognitiveModuleManager(self.args)
        else:
            self.cognitive_manager = None
        
        # 4. Episode统计器
        self.episode_statistics = EpisodeStatistics(self.lane_change_tracker, self.args)
        
        # 5. 可视化管理器
        self.visualization_manager = VisualizationManager(self.args)
        
        # 动作处理器将在网络加载后创建
        self.action_processor = None
    
    def _load_checkpoint_and_network(self):
        """加载检查点并创建网络"""
        print(f"📥 加载检查点...")
        
        # 创建检查点加载器
        self.checkpoint_loader = CheckpointLoader(self.checkpoint_path, self.device)
        
        # 加载检查点（决定观测维度）
        use_cognitive_modules = bool(self.cognitive_manager)
        self.network = self.checkpoint_loader.load_checkpoint(use_cognitive_modules)
        
        # 获取检查点信息
        self.checkpoint_info = self.checkpoint_loader.get_checkpoint_info()
    
    def _create_action_processor(self):
        """创建动作处理器"""
        self.action_processor = ActionProcessor(
            network=self.network,
            cognitive_manager=self.cognitive_manager,
            device=self.device
        )
    
    def _print_initialization_summary(self):
        """打印初始化摘要"""
        print(f"\n📊 仿真系统初始化摘要:")
        print(f"   🎯 训练迭代: {self.checkpoint_info.get('iteration', 'Unknown')}")
        print(f"   🚀 全局步数: {self.checkpoint_info.get('global_step', 'Unknown')}")
        print(f"   🧠 认知模块: {'启用' if self.cognitive_manager else '禁用'}")
        print(f"   🚗 变道跟踪: 启用")
        print(f"   🎨 可视化: {'启用' if (self.visualization_manager.enable_cognitive_visualization or self.visualization_manager.enable_speed_control_visualization) else '禁用'}")
        print(f"   📈 统计收集: 启用")
    
    def run_single_episode(self, render: bool = True, max_steps: int = 1000, 
                          scenario_seed: Optional[int] = None, deterministic: bool = True) -> Dict:
        """
        运行单个仿真episode
        
        Args:
            render: 是否渲染
            max_steps: 最大步数
            scenario_seed: 场景种子
            deterministic: 是否使用确定性策略
            
        Returns:
            episode统计信息
        """
        # 1. 创建环境
        env = self.environment_factory.create_environment(render=render, scenario_seed=scenario_seed)
        
        # 2. 设置变道冷却时间
        self.lane_change_tracker.setup_cooldown(env)
        
        # 3. 重置环境
        obs, info = env.reset()
        
        # 4. 认知模块附加到环境并重置
        if self.cognitive_manager:
            self.cognitive_manager.attach_to_env(env)
            self.cognitive_manager.reset_modules()
        
        # 5. 初始化Episode统计
        episode_stats = self.episode_statistics.initialize_episode()
        
        # 6. 运行Episode主循环
        step_count = 0
        
        try:
            for step in range(max_steps):
                # 获取动作
                action = self.action_processor.get_action(
                    obs, deterministic=deterministic, env=env, step_count=step_count
                )
                
                # 收集认知可视化数据（动作执行前）
                if self.visualization_manager.enable_cognitive_visualization:
                    self.visualization_manager.collect_cognitive_data(
                        step_count=step_count,
                        cognitive_manager=self.cognitive_manager,
                        original_obs=obs.copy(),
                        original_action=action.copy()
                    )
                
                # 调试：打印前几步的动作
                if step < 5:
                    print(f"  步骤 {step}: 动作=[{action[0]:.3f}, {action[1]:.3f}]")
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                original_reward = reward
                
                # 认知偏差模块处理奖励
                if self.cognitive_manager:
                    reward, bias_applied, bias_info = self.cognitive_manager.process_reward(reward, env, info)
                    if bias_applied and step < 5:
                        print(f"🧠 认知偏差: {original_reward:.3f} → {reward:.3f}")
                
                # 获取速度信息
                speed = getattr(env.agent, 'speed', None)
                if speed is not None and step < 5:
                    print(f"    速度: {speed:.3f} m/s, 奖励: {reward:.3f}")
                
                # 更新Episode统计（包括变道检测）
                adjusted_reward = self.episode_statistics.update_step_stats(
                    action=action,
                    reward=reward,
                    speed=speed,
                    info=info,
                    env=env,
                    step_count=step_count
                )
                
                # 收集可视化数据（动作执行后）
                if self.visualization_manager.enable_cognitive_visualization:
                    self.visualization_manager.collect_action_data(step_count, action)
                    # 更新奖励数据
                    if len(self.visualization_manager.cognitive_viz_data['original_rewards']) <= step_count:
                        self.visualization_manager.cognitive_viz_data['original_rewards'].append(float(original_reward))
                        self.visualization_manager.cognitive_viz_data['modified_rewards'].append(float(reward))
                
                # 收集速度控制可视化数据
                if self.visualization_manager.enable_speed_control_visualization:
                    self.visualization_manager.collect_speed_control_data(step_count, env)
                
                # 更新认知模块统计
                if self.cognitive_manager:
                    self.episode_statistics.update_cognitive_stats(
                        bias_info=self.cognitive_manager.get_bias_info(),
                        delay_info=self.cognitive_manager.get_delay_info(),
                        perception_info={'status': 'active'} if self.cognitive_manager.cognitive_perception_module else None
                    )
                
                # 检查终止条件
                if terminated or truncated:
                    self.episode_statistics.update_termination_stats(info)
                    break
                
                step_count += 1
                
                # 渲染时添加延迟
                if render:
                    time.sleep(0.05)  # 50ms延迟，便于观察
        
        finally:
            # 完成Episode统计
            final_stats = self.episode_statistics.finalize_episode(env, info)
            
            # 生成可视化
            self._generate_visualizations(final_stats)
            
            # 认知模块清理
            if self.cognitive_manager:
                self.cognitive_manager.generate_visualizations(
                    save_dir=f"cognitive_visualization/cognitive_visualization_{time.strftime('%Y%m%d_%H%M%S')}",
                    env=env
                )
                self.cognitive_manager.detach_from_env()
            
            # 关闭环境
            env.close()
        
        return final_stats
    
    def _generate_visualizations(self, episode_stats: Dict):
        """生成可视化图表"""
        timestamp_dir = f"cognitive_visualization_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # 生成认知可视化
        if self.visualization_manager.enable_cognitive_visualization:
            try:
                viz_path = self.visualization_manager.generate_cognitive_visualization(
                    episode_stats, save_dir=timestamp_dir
                )
                if viz_path:
                    episode_stats["cognitive_visualization_path"] = viz_path
                
                # 清空数据为下一个episode准备
                self.visualization_manager.clear_cognitive_visualization_data()
            except Exception as e:
                print(f"⚠️ 认知可视化生成失败: {e}")
        
        # 生成速度控制可视化
        if self.visualization_manager.enable_speed_control_visualization:
            try:
                viz_path = self.visualization_manager.generate_speed_control_visualization(episode_stats)
                if viz_path:
                    episode_stats["speed_control_visualization_path"] = viz_path
                
                # 清空数据为下一个episode准备
                self.visualization_manager.clear_speed_control_visualization_data()
            except Exception as e:
                print(f"⚠️ 速度控制可视化生成失败: {e}")
    
    def run_simulation(self, num_episodes: int = 5, render: bool = True, 
                      max_steps: int = 1000, deterministic: bool = True) -> List[Dict]:
        """
        运行多个仿真episodes
        
        Args:
            num_episodes: episode数量
            render: 是否渲染
            max_steps: 每个episode最大步数
            deterministic: 是否使用确定性策略
            
        Returns:
            所有episodes的统计信息列表
        """
        print(f"\n🚗 开始PPO仿真 ({num_episodes} episodes)")
        print("=" * 60)
        
        all_stats = []
        
        for episode in range(num_episodes):
            print(f"\n🎮 Episode {episode + 1}/{num_episodes}")
            
            # 固定第一个场景种子，确保背景车初始状态可重复；后续场景可以不同
            if episode == 0:
                scenario_seed = 8888  # 第一个场景始终使用固定种子
            else:
                scenario_seed = 8888 + episode  # 后续场景使用不同种子
            
            # 运行episode
            stats = self.run_single_episode(
                render=render,
                max_steps=max_steps,
                scenario_seed=scenario_seed,
                deterministic=deterministic
            )
            
            all_stats.append(stats)
            
            # 打印episode结果
            EpisodeStatistics.print_episode_results(episode + 1, stats)
            
            # 打印可视化路径
            if 'cognitive_visualization_path' in stats and stats['cognitive_visualization_path']:
                print(f"   🎨 认知可视化: {stats['cognitive_visualization_path']}")
            
            if 'speed_control_visualization_path' in stats and stats['speed_control_visualization_path']:
                print(f"   🚀 速度控制可视化: {stats['speed_control_visualization_path']}")
        
        # 打印总体统计
        summary = EpisodeStatistics.compute_summary_stats(all_stats)
        EpisodeStatistics.print_summary_stats(summary)
        
        # 打印认知模块统计
        self._print_cognitive_summary()
        
        return all_stats
    
    def _print_cognitive_summary(self):
        """打印认知模块总体统计"""
        if not self.cognitive_manager:
            return
        
        print(f"\n🧠 认知模块总体统计:")
        
        # 获取认知模块统计
        cognitive_stats = self.cognitive_manager.get_statistics()
        
        for module_name, module_stats in cognitive_stats.get('modules_enabled', {}).items():
            if module_name == 'bias':
                print(f"   💭 认知偏差:")
                if 'average_bias' in module_stats:
                    print(f"      平均偏差强度: {module_stats['average_bias']:.3f}")
                else:
                    print(f"      状态: 活跃")
            
            elif module_name == 'delay':
                print(f"   ⏰ 认知延迟:")
                delay_steps = module_stats.get('delay_steps', 'Unknown')
                print(f"      延迟步数: {delay_steps}")
            
            elif module_name == 'perception':
                print(f"   👁️ 认知感知:")
                if 'noise_config' in module_stats:
                    noise_config = module_stats['noise_config']
                    sigma0 = noise_config.get('sigma0', 0.0)
                    print(f"      基准噪声水平: {sigma0:.3f}")
                else:
                    print(f"      状态: 活跃")
    
    def evaluate_model(self, num_episodes: int = 20, render: bool = False) -> Dict:
        """
        评估模型性能
        
        Args:
            num_episodes: 评估episodes数量
            render: 是否渲染
            
        Returns:
            评估结果字典
        """
        print(f"\n🔍 模型性能评估 ({num_episodes} episodes)")
        print("=" * 60)
        
        all_stats = self.run_simulation(
            num_episodes=num_episodes,
            render=render,
            max_steps=1000,
            deterministic=True
        )
        
        # 计算详细评估指标
        evaluation = EpisodeStatistics.compute_summary_stats(all_stats)
        
        # 添加检查点信息
        evaluation.update({
            "checkpoint_path": self.checkpoint_path,
            "training_iteration": self.checkpoint_info.get('iteration', 'Unknown'),
            "training_global_step": self.checkpoint_info.get('global_step', 'Unknown')
        })
        
        # 添加认知模块评估指标
        if self.cognitive_manager:
            evaluation["cognitive_modules"] = self.cognitive_manager.get_statistics()
        
        return evaluation
    
    def get_system_status(self) -> Dict:
        """获取系统状态摘要"""
        status = {
            'checkpoint_info': self.checkpoint_info,
            'modules': {
                'checkpoint_loader': bool(self.checkpoint_loader),
                'environment_factory': bool(self.environment_factory),
                'cognitive_manager': bool(self.cognitive_manager),
                'lane_change_tracker': bool(self.lane_change_tracker),
                'action_processor': bool(self.action_processor),
                'episode_statistics': bool(self.episode_statistics),
                'visualization_manager': bool(self.visualization_manager)
            },
            'cognitive_modules': self.cognitive_manager.get_statistics() if self.cognitive_manager else None,
            'visualization_status': self.visualization_manager.get_visualization_status(),
            'lane_change_stats': self.lane_change_tracker.get_statistics(),
            'statistics_summary': self.episode_statistics.get_statistics_summary()
        }
        
        return status 