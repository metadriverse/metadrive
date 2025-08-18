#!/bin/bash

# PPO Expert复现训练系统安装检查脚本
# 检查并安装所需依赖，确保系统可正常运行

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=================================================="
echo "   PPO Expert复现训练系统安装检查"
echo "=================================================="
echo -e "${NC}"

# 检查Python版本
check_python() {
    echo -e "${YELLOW}🐍 检查Python版本...${NC}"
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
        echo -e "  ✅ Python $PYTHON_VERSION"
        
        # 检查版本是否满足要求
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 8 ]; then
            echo -e "  ✅ Python版本满足要求 (>= 3.8)"
            return 0
        else
            echo -e "  ❌ Python版本过低，需要 >= 3.8"
            return 1
        fi
    else
        echo -e "  ❌ 未找到Python3"
        return 1
    fi
}

# 检查pip
check_pip() {
    echo -e "${YELLOW}📦 检查pip...${NC}"
    
    if command -v pip3 &> /dev/null; then
        echo -e "  ✅ pip3 可用"
        return 0
    elif command -v pip &> /dev/null; then
        echo -e "  ✅ pip 可用"
        return 0
    else
        echo -e "  ❌ 未找到pip"
        echo -e "  💡 请安装pip: sudo apt install python3-pip"
        return 1
    fi
}

# 检查和安装Python包
install_package() {
    local package=$1
    local pip_name=${2:-$package}
    
    echo -e "  📦 检查 $package..."
    
    if python3 -c "import $package" &> /dev/null; then
        local version=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null)
        echo -e "    ✅ $package $version (已安装)"
        return 0
    else
        echo -e "    ⚠️ $package 未安装，正在安装..."
        
        if pip3 install $pip_name; then
            echo -e "    ✅ $package 安装成功"
            return 0
        else
            echo -e "    ❌ $package 安装失败"
            return 1
        fi
    fi
}

# 检查核心依赖
check_core_dependencies() {
    echo -e "${YELLOW}📚 检查核心Python依赖...${NC}"
    
    local packages=("numpy" "pandas" "matplotlib" "seaborn")
    local failed=0
    
    for package in "${packages[@]}"; do
        if ! install_package $package; then
            ((failed++))
        fi
    done
    
    return $failed
}

# 检查PyTorch
check_pytorch() {
    echo -e "${YELLOW}🔥 检查PyTorch...${NC}"
    
    if python3 -c "import torch" &> /dev/null; then
        local version=$(python3 -c "import torch; print(torch.__version__)")
        echo -e "  ✅ PyTorch $version (已安装)"
        
        # 检查CUDA支持
        if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
            local cuda_version=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
            echo -e "  ✅ CUDA支持可用 (CUDA $cuda_version)"
        else
            echo -e "  ⚠️ CUDA不可用 (将使用CPU训练)"
        fi
        
        return 0
    else
        echo -e "  ⚠️ PyTorch 未安装，正在安装CPU版本..."
        
        if pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu; then
            echo -e "  ✅ PyTorch CPU版本安装成功"
            echo -e "  💡 如需GPU支持，请手动安装对应版本"
            return 0
        else
            echo -e "  ❌ PyTorch安装失败"
            return 1
        fi
    fi
}

# 检查MetaDrive
check_metadrive() {
    echo -e "${YELLOW}🚗 检查MetaDrive...${NC}"
    
    if python3 -c "from metadrive.envs.metadrive_env import MetaDriveEnv" &> /dev/null; then
        echo -e "  ✅ MetaDrive 已安装并可正常导入"
        
        # 测试环境创建
        echo -e "  🧪 测试环境创建..."
        if python3 -c "
from metadrive.envs.metadrive_env import MetaDriveEnv
env = MetaDriveEnv({'use_render': False, 'num_scenarios': 1})
obs, _ = env.reset()
print(f'观测维度: {obs.shape}')
env.close()
print('环境测试成功')
" &> /dev/null; then
            echo -e "    ✅ 环境创建测试通过"
            return 0
        else
            echo -e "    ❌ 环境创建测试失败"
            return 1
        fi
    else
        echo -e "  ⚠️ MetaDrive 未安装，正在安装..."
        
        if pip3 install metadrive-simulator; then
            echo -e "  ✅ MetaDrive 安装成功"
            echo -e "  🧪 测试环境创建..."
            
            if python3 -c "
from metadrive.envs.metadrive_env import MetaDriveEnv
env = MetaDriveEnv({'use_render': False, 'num_scenarios': 1})
obs, _ = env.reset()
print(f'观测维度: {obs.shape}')
env.close()
print('环境测试成功')
" &> /dev/null; then
                echo -e "    ✅ 环境创建测试通过"
                return 0
            else
                echo -e "    ❌ 环境创建测试失败"
                return 1
            fi
        else
            echo -e "  ❌ MetaDrive 安装失败"
            return 1
        fi
    fi
}

# 检查系统要求
check_system_requirements() {
    echo -e "${YELLOW}💻 检查系统要求...${NC}"
    
    # 检查内存
    if command -v free &> /dev/null; then
        local total_mem=$(free -g | awk '/^Mem:/{print $2}')
        echo -e "  💾 系统内存: ${total_mem}GB"
        
        if [ "$total_mem" -ge 4 ]; then
            echo -e "    ✅ 内存充足 (推荐 >= 4GB)"
        else
            echo -e "    ⚠️ 内存可能不足，建议至少4GB"
        fi
    fi
    
    # 检查磁盘空间
    local available_space=$(df -h . | awk 'NR==2{print $4}' | sed 's/G//')
    if [[ "$available_space" =~ ^[0-9]+$ ]] && [ "$available_space" -ge 5 ]; then
        echo -e "  💽 磁盘空间: ${available_space}GB 可用"
        echo -e "    ✅ 磁盘空间充足 (推荐 >= 5GB)"
    else
        echo -e "  ⚠️ 磁盘空间可能不足，建议至少5GB可用空间"
    fi
    
    return 0
}

# 运行完整测试
run_full_test() {
    echo -e "${YELLOW}🧪 运行完整系统测试...${NC}"
    
    local test_script="$(dirname "$0")/test_setup.py"
    
    if [ -f "$test_script" ]; then
        echo -e "  🚀 执行测试脚本..."
        if python3 "$test_script"; then
            echo -e "  ✅ 完整系统测试通过"
            return 0
        else
            echo -e "  ❌ 完整系统测试失败"
            return 1
        fi
    else
        echo -e "  ⚠️ 测试脚本不存在: $test_script"
        echo -e "  💡 请确保所有文件都已正确下载"
        return 1
    fi
}

# 主函数
main() {
    local failed=0
    local total_checks=6
    
    # 运行所有检查
    check_python || ((failed++))
    check_pip || ((failed++))
    check_core_dependencies || ((failed++))
    check_pytorch || ((failed++))
    check_metadrive || ((failed++))
    check_system_requirements || ((failed++))
    
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}            安装检查总结${NC}"
    echo -e "${BLUE}============================================${NC}"
    
    local passed=$((total_checks - failed))
    echo -e "总检查项: $total_checks"
    echo -e "通过: ${GREEN}$passed${NC}"
    echo -e "失败: ${RED}$failed${NC}"
    
    if [ $failed -eq 0 ]; then
        echo -e "\n🎉 ${GREEN}所有检查通过！系统已准备就绪。${NC}"
        echo -e "\n🚀 后续步骤:"
        echo -e "1. 运行完整测试: ${BLUE}python3 test_setup.py${NC}"
        echo -e "2. 快速训练测试: ${BLUE}./train.sh debug${NC}"
        echo -e "3. 开始正式训练: ${BLUE}./train.sh default${NC}"
        echo -e "\n📚 查看使用文档: ${BLUE}cat README.md${NC}"
        
        # 询问是否运行完整测试
        echo -e "\n${YELLOW}是否现在运行完整系统测试？(y/n)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            run_full_test
        fi
        
        return 0
    else
        echo -e "\n⚠️ ${YELLOW}部分检查失败，请解决以下问题：${NC}"
        echo -e "\n🔧 常见解决方案:"
        echo -e "1. 更新pip: ${BLUE}pip3 install --upgrade pip${NC}"
        echo -e "2. 安装依赖: ${BLUE}pip3 install torch numpy pandas matplotlib seaborn${NC}"
        echo -e "3. 安装MetaDrive: ${BLUE}pip3 install metadrive-simulator${NC}"
        echo -e "4. 检查Python版本: ${BLUE}python3 --version${NC}"
        echo -e "\n💡 如果问题持续，请检查网络连接和权限设置。"
        
        return 1
    fi
}

# 检查是否在正确的目录
check_directory() {
    local current_dir=$(basename "$(pwd)")
    local script_dir=$(dirname "$0")
    
    if [ "$script_dir" != "." ] && [ "$script_dir" != "" ]; then
        echo -e "${YELLOW}💡 建议在ppo_reproduction目录下运行此脚本${NC}"
        echo -e "当前目录: $(pwd)"
        echo -e "脚本目录: $script_dir"
        echo -e "\n按Enter继续，或Ctrl+C退出..."
        read -r
    fi
}

# 脚本入口
echo -e "${BLUE}开始系统检查...${NC}\n"

check_directory
main

exit_code=$?

echo -e "\n${BLUE}检查完成。${NC}"
exit $exit_code 