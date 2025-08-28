#!/bin/bash
#SBATCH -J PPO_train_with_speed_reward_AB1_11    # 作业名称

#SBATCH --partition=intel        # 使用哪个分区

#SBATCH --nodes=1                # 申请1个节点
#SBATCH --ntasks=1               # 申请1个任务(进程)
#SBATCH --cpus-per-task=24        # 每个任务用24个cpu
#SBATCH --mem-per-cpu=10g    # 每个cpu分配10G

# 输出目录
#SBATCH -o /share/home/u22537/data/JXY/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/output/speed_30/PPO_train_with_speed_reward_AB1_11_%j.out
#SBATCH -e /share/home/u22537/data/JXY/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/output/speed_30/PPO_train_with_speed_reward_AB1_11_%j.err

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
  --save_dir /share/home/u22537/data/JXY/metadrive/a_scen_env/a_ppo_train/ppo_reproduction/speed_30_2/speed_reward_AB1_11 \
  --use_speed_control_reward \
  --speed_control_enable_tracking --speed_control_k 0.12 \
  --speed_control_enable_soft_wall --speed_control_kappa 0.20
