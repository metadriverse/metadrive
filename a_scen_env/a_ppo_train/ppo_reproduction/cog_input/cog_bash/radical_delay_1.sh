#!/bin/bash
#SBATCH -J radical_1    # 作业名称

#SBATCH --partition=intel        # 使用哪个分区

#SBATCH --nodes=1                # 申请1个节点
#SBATCH --ntasks=1               # 申请1个任务(进程)
#SBATCH --cpus-per-task=24        # 每个任务用24个cpu
#SBATCH --mem-per-cpu=10g    # 每个cpu分配10G

# 输出目录
#SBATCH -o /share/home/u22537/data/JXY/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/output/speed_30/PPO_train_with_speed_reward_AB1_10_%j.out
#SBATCH -e /share/home/u22537/data/JXY/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/output/speed_30/PPO_train_with_speed_reward_AB1_10_%j.err

# 加载 Conda 环境
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate metadrive_env

# 切换到项目目录
cd /share/home/u22537/data/JXY/metadrive/a_scen_env/a_ppo_train/ppo_reproduction

# 运行python
python ppo_expert_reproduction_with_cog.py \
  --total_timesteps 50000000 \
  --use_curriculum --curriculum_mode progress --curriculum_alpha 1.5 \
  --lr 3e-4 --lr_schedule linear --lr_min 3e-5 --warmup_ratio 0.05 \
  --n_envs 3 --n_steps 5376 --batch_size 512 --n_epochs 10 \
  --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 \
  --entropy_coef_start 0.015 --entropy_coef_end 0.005 --entropy_decay_end_ratio 0.8 \
  --vf_coef 0.5 --max_grad_norm 0.5 --target_kl 0.03 \
  --eval_freq 2 --checkpoint_freq 10 --log_freq 1 \
  --seed 101 \
  --save_dir /share/home/u22537/data/JXY/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/speed_30_2/reward/reward_speed_AB1_10 \
  --use_speed_control_reward \
  --speed_control_enable_tracking --speed_control_k 0.12 \
  --speed_control_enable_soft_wall --speed_control_kappa 0.15 \
  --speed_control_enable_behavior_guidance --speed_control_mu 0.3 --speed_control_nu 0.2 \
  --success_reward 100 --out_of_road_penalty 8 --crash_penalty 8 --crash_sidewalk_penalty 8 --driving_reward 0.4 \
  --use_lateral_reward \
  --use_cognitive_parameter_sampling --cognitive_param_update_steps 5 \
  --use_cognitive_modules \
  --use_cognitive_parameter_sampling --cognitive_sampler_type discrete \
  --bias_inverse_tta_coef_range 0 0.1 --bias_inverse_tta_coef_density 4 \
  --perception_sigma0_range 0 0.1 --perception_sigma0_density 4 \
  --perception_k_range 0 0.1 --perception_k_density 4 \
  --delay_steps_range 0 1 --delay_steps_density 2 \
  --use_cognitive_bias \
  --use_cognitive_delay \
  --use_cognitive_perception \
  --resume_from /home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/cog_input/cog_ckpt/reward_speed_AB1_10_922_latest_model.pt




本地
# 运行python
python ppo_expert_reproduction_with_cog.py \
  --total_timesteps 50000000 \
  --use_curriculum --curriculum_mode progress --curriculum_alpha 1.5 \
  --lr 3e-4 --lr_schedule linear --lr_min 3e-5 --warmup_ratio 0.05 \
  --n_envs 3 --n_steps 5376 --batch_size 512 --n_epochs 10 \
  --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 \
  --entropy_coef_start 0.015 --entropy_coef_end 0.005 --entropy_decay_end_ratio 0.8 \
  --vf_coef 0.5 --max_grad_norm 0.5 --target_kl 0.03 \
  --eval_freq 2 --checkpoint_freq 10 --log_freq 1 \
  --seed 101 \
  --save_dir /home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/cog_input/cog_save/radical_1 \
  --use_speed_control_reward \
  --speed_control_enable_tracking --speed_control_k 0.12 \
  --speed_control_enable_soft_wall --speed_control_kappa 0.15 \
  --speed_control_enable_behavior_guidance --speed_control_mu 0.3 --speed_control_nu 0.2 \
  --success_reward 100 --out_of_road_penalty 8 --crash_penalty 8 --crash_sidewalk_penalty 8 --driving_reward 0.4 \
  --use_lateral_reward \
  --use_cognitive_parameter_sampling --cognitive_param_update_steps 5 \
  --use_cognitive_modules \
  --use_cognitive_parameter_sampling --cognitive_sampler_type discrete \
  --bias_inverse_tta_coef_range 0 0.1 --bias_inverse_tta_coef_density 4 \
  --perception_sigma0_range 0 0.1 --perception_sigma0_density 4 \
  --perception_k_range 0 0.1 --perception_k_density 4 \
  --delay_steps_range 0 1 --delay_steps_density 2 \
  --use_cognitive_bias \
  --use_cognitive_delay \
  --use_cognitive_perception \
  --resume_from /home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/cog_input/cog_ckpt/reward_speed_AB1_10_922_latest_model.pt
