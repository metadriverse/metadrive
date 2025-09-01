#!/usr/bin/env python3
"""
PPO检查点仿真控制器 - 模块化版本
基于训练好的PPO检查点在MetaDrive仿真环境中控制主车行为
支持加载指定检查点、可视化仿真、性能评估等功能
集成认知模块：认知偏差、认知延迟、认知感知

使用重构后的模块化架构，提高代码的可维护性和可扩展性

python ppo_checkpoint_simulation_with_cog.py \
  --checkpoint /home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/runs/ppo_expert_reproduction_20250820_154108/checkpoints/checkpoint_2270.pt \
  --use_cognitive_modules \
  --use_cognitive_bias \
  --use_cognitive_delay \
  --use_cognitive_perception \
  --enable_cognitive_viz \
  --episodes 1 \
  --max_steps 500 \
  --no_render
"""

import os
import sys
import argparse
from pathlib import Path

# 添加metadrive到路径
current_dir = Path(__file__).parent.absolute()
metadrive_root = current_dir.parent.parent.parent
sys.path.insert(0, str(metadrive_root))

# 添加认知模块路径
cognitive_module_path = current_dir.parent.parent / "cognitive_module"
sys.path.insert(0, str(cognitive_module_path))

# 🔧 新版本：导入重构后的模块化架构
from sim_module.simulation_runner import SimulationRunner


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PPO检查点仿真控制器 - 模块化版本")
    
    parser.add_argument("--checkpoint", type=str,
                       default="/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/ckpt_0827/AB_1_13_latest_model.pt",
                       help="检查点文件路径")
    
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径（可选）")
    
    parser.add_argument("--episodes", type=int, default=5,
                       help="仿真episodes数量")
    
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="每个episode最大步数")
    
    parser.add_argument("--no_render", action="store_true",
                       help="禁用渲染")
    
    parser.add_argument("--stochastic", action="store_true",
                       help="使用随机策略（默认确定性）")
    
    parser.add_argument("--evaluate", action="store_true",
                       help="运行模型评估")
    
    parser.add_argument("--eval_episodes", type=int, default=20,
                       help="评估episodes数量")
    
    # ===== 变道惩罚配置参数 =====
    parser.add_argument("--w_lc", type=float, default=0.6,
                       help="基础变道成本 (默认: 0.6)")
    parser.add_argument("--k_speed", type=float, default=1.0,
                       help="高速放大系数 (默认: 1.0)")
    parser.add_argument("--v_limit", type=float, default=15.0,
                       help="用于速度归一的限速 (默认: 15.0)")
    parser.add_argument("--lc_cooldown_s", type=float, default=4.0,
                       help="变道冷却时间，秒 (默认: 4.0)")
    parser.add_argument("--w_lc_cool", type=float, default=1,
                       help="冷却期内附加惩罚 (默认: 1)")
    
    # ===== 认知模块配置参数 =====
    parser.add_argument("--use_cognitive_modules", action="store_true",
                       help="启用认知模块 (默认禁用)")
    
    parser.add_argument("--use_cognitive_bias", action="store_true",
                       help="启用认知偏差模块 (默认禁用)")
    parser.add_argument("--bias_visual_aversion", action="store_true",
                       help="认知偏差模块启用视觉厌恶 (默认启用)")
    parser.add_argument("--bias_visual_distance", type=float, default=50.0,
                       help="认知偏差模块视觉距离 (默认: 50.0)")
    parser.add_argument("--bias_inverse_tta_coef", type=float, default=1.5,
                       help="认知偏差模块looming penalty系数 c (默认: 1.5)")
    parser.add_argument("--bias_tta_threshold", type=float, default=0.1,
                       help="认知偏差模块TTA阈值 (默认: 0.1)")
    
    parser.add_argument("--use_cognitive_delay", action="store_true",
                       help="启用认知延迟模块 (默认禁用)")
    parser.add_argument("--delay_steps", type=int, default=2,
                       help="认知延迟模块延迟步数 (默认: 2)")  # 一个step是0.1s
    
    parser.add_argument("--use_cognitive_perception", action="store_true",
                       help="启用认知感知模块 (默认禁用)")
    parser.add_argument("--perception_sigma0", type=float, default=0.1,
                       help="基础噪声标准差（米）（默认0.1）")
    parser.add_argument("--perception_sigma_max", type=float, default=0.8,
                       help="最大噪声标准差（米）（默认0.8）")
    parser.add_argument("--perception_p_miss0", type=float, default=0.0,
                       help="基础漏检概率（默认0.0，已关闭）")
    parser.add_argument("--perception_p_false", type=float, default=0.0,
                       help="误检概率（默认0.0，已关闭）")
    parser.add_argument("--perception_use_kf", action="store_true", default=True,
                       help="启用卡尔曼滤波（默认开启）")
    parser.add_argument("--perception_kf_dt", type=float, default=0.1,
                       help="卡尔曼滤波步长（默认0.1）")
    parser.add_argument("--perception_kf_q_scale", type=float, default=100.0,
                       help="卡尔曼滤波过程噪声缩放（默认100.0）")
    
    # ===== 认知可视化参数 =====
    parser.add_argument("--enable_radar_beam_viz", action="store_true",
                       help="启用雷达束可视化 (默认禁用)")
    
    # ===== 速度控制奖励配置参数 =====
    parser.add_argument("--use_speed_control_reward", action="store_true",
                       help="启用速度控制奖励 (默认禁用)")
    parser.add_argument("--speed_control_k", type=float, default=1.0,
                       help="速度跟踪系数 (默认: 1.0)")
    parser.add_argument("--speed_control_kappa", type=float, default=0.5,
                       help="超速软墙系数 (默认: 0.5)")
    parser.add_argument("--speed_control_mu", type=float, default=0.3,
                       help="超速刹车奖励系数 (默认: 0.3)")
    parser.add_argument("--speed_control_nu", type=float, default=0.2,
                       help="超速加速惩罚系数 (默认: 0.2)")
    parser.add_argument("--speed_control_v_tolerance", type=float, default=1.0,
                       help="速度跟踪容差 (默认: 1.0)")
    parser.add_argument("--speed_control_v_ref", type=float, default=15.0,
                       help="目标参考速度 (默认: 15.0)")
    parser.add_argument("--speed_control_enable_tracking", action="store_true",
                       help="启用速度跟踪子模块 (默认禁用)")
    parser.add_argument("--speed_control_enable_soft_wall", action="store_true",
                       help="启用超速软墙子模块 (默认禁用)")
    parser.add_argument("--speed_control_enable_behavior_guidance", action="store_true",
                       help="启用行为导向子模块 (默认禁用)")
    
    # ===== 认知可视化配置参数 =====
    parser.add_argument("--enable_cognitive_viz", action="store_true",
                       help="启用认知模块可视化 (默认禁用)")
    
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["auto", "cpu", "cuda"],
                       help="计算设备")
    
    args = parser.parse_args()
    
    # 检查检查点文件
    if not os.path.exists(args.checkpoint):
        print(f"❌ 检查点文件不存在: {args.checkpoint}")
        sys.exit(1)
    
    print(f"🚀 使用模块化架构创建仿真运行器...")
    
    # 🔧 新版本：使用模块化的SimulationRunner
    try:
        runner = SimulationRunner(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=args.device,
            args=args
        )
        
        # 打印系统状态摘要
        status = runner.get_system_status()
        print(f"\n📊 系统状态检查:")
        print(f"   所有模块: {'✅ 正常' if all(status['modules'].values()) else '❌ 异常'}")
        if status['cognitive_modules']:
            print(f"   认知模块: ✅ 已启用")
        print(f"   可视化: {'✅ 已启用' if any(status['visualization_status'].values()) else '❌ 禁用'}")
        
    except Exception as e:
        print(f"❌ 仿真运行器初始化失败: {e}")
        sys.exit(1)
    
    # 打印速度控制配置（如果启用）
    if args.use_speed_control_reward:
        print(f"\n🚀 速度控制奖励配置:")
        print(f"   📊 速度跟踪系数: {args.speed_control_k}")
        print(f"   🧱 超速软墙系数: {args.speed_control_kappa}")
        print(f"   🎯 行为导向系数: μ={args.speed_control_mu}, ν={args.speed_control_nu}")
        print(f"   ⚡ 目标速度: {args.speed_control_v_ref} m/s")
        print(f"   📏 速度容差: {args.speed_control_v_tolerance} m/s")
        print(f"   ✅ 子模块状态: 跟踪={args.speed_control_enable_tracking}, 软墙={args.speed_control_enable_soft_wall}, 行为={args.speed_control_enable_behavior_guidance}")
    
    try:
        if args.evaluate:
            # 运行模型评估
            print(f"\n🔍 开始模型性能评估...")
            evaluation = runner.evaluate_model(
                num_episodes=args.eval_episodes,
                render=not args.no_render
            )
            
            print(f"\n📊 评估结果总结:")
            print(f"🏆 成功率: {evaluation['success_rate']:.1%}")
            print(f"💰 平均奖励: {evaluation['avg_reward']:.2f} ± {evaluation.get('std_reward', 0):.2f}")
            print(f"📏 平均Episode长度: {evaluation['avg_episode_length']:.1f}")
            print(f"🎯 平均路径完成度: {evaluation['avg_path_completion']:.1%}")
            
            # 变道统计评估输出
            print(f"🚗 总变道次数: {evaluation['total_lane_changes']}")
            print(f"💸 平均变道惩罚: {evaluation['avg_lane_change_penalty']:.3f}")
            print(f"⚡ 平均变道速度比: {evaluation['avg_lane_change_speed_ratio']:.3f}")
            print(f"⏰ 总冷却期违规: {evaluation['total_cooldown_violations']}")
            
        else:
            # 运行常规仿真
            print(f"\n🎮 开始仿真运行...")
            all_stats = runner.run_simulation(
                num_episodes=args.episodes,
                render=not args.no_render,
                max_steps=args.max_steps,
                deterministic=not args.stochastic
            )
            
            # 简单输出最终摘要
            print(f"\n🎯 仿真完成摘要:")
            print(f"   总Episodes: {len(all_stats)}")
            success_count = sum(1 for stats in all_stats if stats['success'])
            print(f"   成功Episodes: {success_count}/{len(all_stats)} ({success_count/len(all_stats)*100:.1f}%)")
            
        print(f"\n✅ 所有任务完成！")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 用户中断仿真")
    except Exception as e:
        print(f"\n❌ 仿真运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 