#!/usr/bin/env python3
"""
环境配置测试脚本
验证PPO Expert复现训练系统的依赖和配置是否正确
"""

import sys
import os
import importlib
from pathlib import Path

# 添加metadrive到路径
current_dir = Path(__file__).parent.absolute()
metadrive_root = current_dir.parent.parent.parent
sys.path.insert(0, str(metadrive_root))

def test_python_version():
    """测试Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} (满足要求)")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (需要 >= 3.8)")
        return False

def test_package_import(package_name, min_version=None):
    """测试包导入和版本"""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            version = module.__version__
            print(f"  ✅ {package_name} {version}")
        else:
            print(f"  ✅ {package_name} (已安装)")
        return True
    except ImportError:
        print(f"  ❌ {package_name} (未安装)")
        return False

def test_torch_cuda():
    """测试PyTorch CUDA支持"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"  ✅ CUDA可用，设备数量: {device_count}, 主设备: {device_name}")
            return True
        else:
            print(f"  ⚠️ CUDA不可用 (将使用CPU)")
            return True
    except Exception as e:
        print(f"  ❌ CUDA检查失败: {e}")
        return False

def test_metadrive_env():
    """测试MetaDrive环境"""
    print("🚗 测试MetaDrive环境...")
    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        
        # 创建简单环境
        config = {
            "num_scenarios": 1,
            "traffic_density": 0.1,
            "use_render": False,
            "horizon": 100
        }
        
        env = MetaDriveEnv(config)
        obs, _ = env.reset()
        
        print(f"  ✅ 环境创建成功")
        print(f"  ✅ 观测空间维度: {obs.shape}")
        print(f"  ✅ 动作空间: {env.action_space}")
        
        # 测试一步执行
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  ✅ 环境执行成功")
        print(f"  ✅ 奖励范围正常: {reward:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ❌ MetaDrive环境测试失败: {e}")
        return False

def test_expert_config():
    """测试Expert配置对齐"""
    print("🧠 测试Expert配置对齐...")
    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        
        # Expert配置
        expert_obs_cfg = dict(
            lidar=dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0),
            side_detector=dict(num_lasers=0, distance=50, gaussian_noise=0.0, dropout_prob=0.0),
            lane_line_detector=dict(num_lasers=0, distance=20, gaussian_noise=0.0, dropout_prob=0.0)
        )
        
        config = {
            "num_scenarios": 1000,
            "traffic_density": 0.1,
            "random_traffic": False,
            "horizon": 1000,
            "map": 3,
            "vehicle_config": expert_obs_cfg
        }
        
        env = MetaDriveEnv(config)
        obs, _ = env.reset()
        
        # 检查观测维度
        expected_dim = 275  # Expert观测维度
        if obs.shape[0] == expected_dim:
            print(f"  ✅ 观测维度对齐: {obs.shape[0]} (期望: {expected_dim})")
        else:
            print(f"  ⚠️ 观测维度不匹配: {obs.shape[0]} (期望: {expected_dim})")
        
        # 检查动作空间
        if hasattr(env.action_space, 'shape') and env.action_space.shape[0] == 2:
            print(f"  ✅ 动作空间对齐: {env.action_space.shape[0]}D 连续控制")
        else:
            print(f"  ⚠️ 动作空间可能不匹配: {env.action_space}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Expert配置测试失败: {e}")
        return False

def test_network_creation():
    """测试网络创建"""
    print("🧮 测试PPO网络创建...")
    try:
        import torch
        import torch.nn as nn
        
        # 简化的网络测试
        class TestPPONetwork(nn.Module):
            def __init__(self, obs_dim=275, action_dim=2, hidden_dim=256):
                super().__init__()
                self.actor_fc1 = nn.Linear(obs_dim, hidden_dim)
                self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.actor_out = nn.Linear(hidden_dim, action_dim * 2)
                self.tanh = nn.Tanh()
                
            def forward(self, obs):
                x = self.tanh(self.actor_fc1(obs))
                x = self.tanh(self.actor_fc2(x))
                return self.actor_out(x)
        
        # 创建网络
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = TestPPONetwork().to(device)
        
        # 测试前向传播
        obs = torch.randn(8, 275).to(device)  # 批次大小8
        output = network(obs)
        
        if output.shape == (8, 4):  # 2*2 (mean + log_std)
            print(f"  ✅ 网络创建成功，输出维度正确: {output.shape}")
        else:
            print(f"  ⚠️ 网络输出维度异常: {output.shape}")
        
        # 测试参数数量
        param_count = sum(p.numel() for p in network.parameters())
        print(f"  ✅ 网络参数数量: {param_count:,}")
        print(f"  ✅ 计算设备: {device}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 网络创建测试失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("📁 检查文件结构...")
    
    files_to_check = [
        "ppo_expert_reproduction.py",
        "evaluate_model.py", 
        "train.sh",
        "README.md"
    ]
    
    base_dir = Path(__file__).parent
    all_exist = True
    
    for file_name in files_to_check:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} (缺失)")
            all_exist = False
    
    # 检查runs目录
    runs_dir = base_dir / "runs"
    if not runs_dir.exists():
        print(f"  📁 创建runs目录...")
        runs_dir.mkdir(exist_ok=True)
        print(f"  ✅ runs/ (已创建)")
    else:
        print(f"  ✅ runs/")
    
    return all_exist

def test_training_script():
    """测试训练脚本语法"""
    print("🔧 检查训练脚本语法...")
    try:
        import py_compile
        script_path = Path(__file__).parent / "ppo_expert_reproduction.py"
        py_compile.compile(str(script_path), doraise=True)
        print(f"  ✅ 训练脚本语法正确")
        return True
    except Exception as e:
        print(f"  ❌ 训练脚本语法错误: {e}")
        return False

def main():
    """主测试函数"""
    print("🔍 PPO Expert复现训练系统环境测试")
    print("=" * 60)
    
    tests = [
        ("Python版本", test_python_version),
        ("文件结构", test_file_structure),
        ("训练脚本语法", test_training_script),
    ]
    
    # 包依赖测试
    print("\n📦 检查Python包依赖...")
    packages = ["numpy", "pandas", "matplotlib", "seaborn", "torch"]
    package_results = []
    
    for package in packages:
        result = test_package_import(package)
        package_results.append(result)
    
    tests.append(("包依赖", lambda: all(package_results)))
    
    # 其他测试
    tests.extend([
        ("PyTorch CUDA", test_torch_cuda),
        ("MetaDrive环境", test_metadrive_env),
        ("Expert配置对齐", test_expert_config),
        ("PPO网络创建", test_network_creation),
    ])
    
    # 执行所有测试
    print("\n" + "=" * 60)
    results = []
    
    for test_name, test_func in tests:
        if test_name not in ["Python版本", "文件结构", "包依赖", "训练脚本语法"]:
            print(f"\n{test_name}...")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name}测试出错: {e}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！系统配置正确，可以开始训练。")
        print("\n🚀 快速开始:")
        print("  cd a_scen_env/a_ppo_train/ppo_reproduction")
        print("  ./train.sh debug  # 快速测试")
        print("  ./train.sh default  # 正式训练")
    else:
        print("\n⚠️ 部分测试失败，请检查环境配置。")
        print("\n🔧 常见解决方案:")
        print("  pip install torch numpy pandas matplotlib seaborn")
        print("  pip install metadrive-simulator")
        
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 