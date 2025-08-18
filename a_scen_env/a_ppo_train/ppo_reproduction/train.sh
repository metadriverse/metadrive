#!/bin/bash

# MetaDrive PPO Expert 复现训练启动脚本
# 提供多种预设配置的快速启动选项

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/ppo_expert_reproduction.py"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示logo
echo -e "${BLUE}"
echo "=================================================="
echo "   MetaDrive PPO Expert 复现训练系统"
echo "=================================================="
echo -e "${NC}"

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}❌ 训练脚本不存在: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# 显示帮助信息
show_help() {
    echo -e "${YELLOW}使用方法:${NC}"
    echo "  $0 [配置名称] [额外参数...]"
    echo ""
    echo -e "${YELLOW}可用配置:${NC}"
    echo "  default      - 默认配置 (推荐新手)"
    echo "  fast         - 快速训练配置 (较少步数)"
    echo "  stable       - 稳定训练配置 (保守参数)"
    echo "  aggressive   - 激进训练配置 (高学习率)"
    echo "  large        - 大规模训练配置 (更多环境)"
    echo "  debug        - 调试配置 (最少步数)"
    echo "  custom       - 自定义配置 (需要手动指定参数)"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  $0 default                    # 使用默认配置"
    echo "  $0 fast --seed 123            # 快速训练，指定种子"
    echo "  $0 custom --lr 1e-3 --n_steps 1024  # 自定义参数"
    echo ""
    echo -e "${YELLOW}额外参数:${NC}"
    echo "  --help                        # 显示完整参数列表"
    echo "  --seed <数值>                 # 随机种子"
    echo "  --lr <数值>                   # 学习率"
    echo "  --n_steps <数值>              # rollout步数"
    echo "  --n_envs <数值>               # 并行环境数"
    echo "  --total_timesteps <数值>      # 总训练步数"
    echo ""
}

# 如果没有参数或请求帮助，显示帮助信息
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# 获取配置名称
CONFIG_NAME="$1"
shift # 移除第一个参数，剩余的作为额外参数

# 根据配置名称设置参数
case "$CONFIG_NAME" in
    "default")
        echo -e "${GREEN}🚀 启动默认配置训练${NC}"
        ARGS=(
            --lr 3e-4
            --n_steps 2048
            --n_envs 8
            --batch_size 256
            --n_epochs 10
            --gamma 0.99
            --gae_lambda 0.95
            --clip_range 0.2
            --entropy_coef 0.01
            --total_timesteps 1000000
        )
        ;;
    
    "fast")
        echo -e "${GREEN}🏃 启动快速训练配置${NC}"
        ARGS=(
            --lr 5e-4
            --n_steps 1024
            --n_envs 16
            --batch_size 128
            --n_epochs 8
            --gamma 0.99
            --gae_lambda 0.95
            --clip_range 0.2
            --entropy_coef 0.02
            --total_timesteps 500000
            --eval_freq 5
            --checkpoint_freq 25
        )
        ;;
    
    "stable")
        echo -e "${GREEN}🛡️ 启动稳定训练配置${NC}"
        ARGS=(
            --lr 1e-4
            --n_steps 4096
            --n_envs 4
            --batch_size 512
            --n_epochs 15
            --gamma 0.995
            --gae_lambda 0.98
            --clip_range 0.15
            --entropy_coef 0.005
            --total_timesteps 2000000
            --max_grad_norm 0.3
        )
        ;;
    
    "aggressive")
        echo -e "${GREEN}⚡ 启动激进训练配置${NC}"
        ARGS=(
            --lr 1e-3
            --n_steps 1024
            --n_envs 16
            --batch_size 128
            --n_epochs 5
            --gamma 0.98
            --gae_lambda 0.9
            --clip_range 0.3
            --entropy_coef 0.05
            --total_timesteps 800000
        )
        ;;
    
    "large")
        echo -e "${GREEN}🌐 启动大规模训练配置${NC}"
        ARGS=(
            --lr 3e-4
            --n_steps 2048
            --n_envs 32
            --batch_size 1024
            --n_epochs 12
            --gamma 0.99
            --gae_lambda 0.95
            --clip_range 0.2
            --entropy_coef 0.01
            --total_timesteps 3000000
        )
        ;;
    
    "debug")
        echo -e "${GREEN}🐛 启动调试配置${NC}"
        ARGS=(
            --lr 1e-3
            --n_steps 256
            --n_envs 2
            --batch_size 64
            --n_epochs 3
            --gamma 0.99
            --gae_lambda 0.95
            --clip_range 0.2
            --entropy_coef 0.01
            --total_timesteps 10000
            --eval_freq 1
            --checkpoint_freq 5
            --log_freq 1
        )
        ;;
    
    "custom")
        echo -e "${GREEN}🔧 启动自定义配置${NC}"
        echo -e "${YELLOW}注意: 自定义配置需要手动指定所有参数${NC}"
        ARGS=()
        ;;
    
    *)
        echo -e "${RED}❌ 未知配置: $CONFIG_NAME${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

# 合并额外参数
FINAL_ARGS=("${ARGS[@]}" "$@")

# 显示最终命令
echo -e "${BLUE}执行命令:${NC}"
echo "python $PYTHON_SCRIPT ${FINAL_ARGS[*]}"
echo ""

# 确认是否继续
echo -e "${YELLOW}按 Enter 继续，Ctrl+C 取消...${NC}"
read -r

# 执行训练
echo -e "${GREEN}开始训练...${NC}"
echo ""

# 记录开始时间
START_TIME=$(date)
echo "训练开始时间: $START_TIME"
echo ""

# 执行Python脚本
python "$PYTHON_SCRIPT" "${FINAL_ARGS[@]}"

# 检查执行结果
EXIT_CODE=$?
END_TIME=$(date)

echo ""
echo "训练结束时间: $END_TIME"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 训练成功完成！${NC}"
    
    # 尝试找到最新的实验目录
    RUNS_DIR="$(dirname "$SCRIPT_DIR")/ppo_reproduction/runs"
    if [ -d "$RUNS_DIR" ]; then
        LATEST_EXP=$(ls -t "$RUNS_DIR" | head -n 1)
        if [ -n "$LATEST_EXP" ]; then
            echo ""
            echo -e "${BLUE}实验结果位置:${NC}"
            echo "$RUNS_DIR/$LATEST_EXP"
            echo ""
            echo -e "${YELLOW}后续操作:${NC}"
            echo "1. 查看TensorBoard:"
            echo "   tensorboard --logdir '$RUNS_DIR/$LATEST_EXP/tensorboard'"
            echo ""
            echo "2. 评估模型:"
            echo "   python evaluate_model.py '$RUNS_DIR/$LATEST_EXP'"
            echo ""
            echo "3. 查看训练报告:"
            echo "   cat '$RUNS_DIR/$LATEST_EXP/report.md'"
        fi
    fi
else
    echo -e "${RED}❌ 训练失败 (退出码: $EXIT_CODE)${NC}"
    echo ""
    echo -e "${YELLOW}故障排除建议:${NC}"
    echo "1. 检查CUDA是否可用 (如果使用GPU)"
    echo "2. 确认MetaDrive环境是否正确安装"
    echo "3. 检查系统内存是否充足"
    echo "4. 尝试使用debug配置进行测试"
fi

exit $EXIT_CODE 