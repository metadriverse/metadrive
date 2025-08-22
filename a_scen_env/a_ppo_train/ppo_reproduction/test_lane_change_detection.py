#!/usr/bin/env python3
"""
测试MetaDrive车道变更检测功能
验证相邻车道信息的获取和使用
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加metadrive到路径
current_dir = Path(__file__).parent.absolute()
metadrive_root = current_dir.parent.parent.parent
sys.path.insert(0, str(metadrive_root))

from metadrive.envs.metadrive_env import MetaDriveEnv

def test_lane_neighbor_info():
    """测试MetaDrive的相邻车道信息"""
    print("🧪 测试MetaDrive相邻车道信息...")
    
    # 创建环境
    env_config = {
        "num_scenarios": 1,
        "map": "SSSSS",  # 5段直线道路
        "traffic_density": 0.1,
        "use_render": False,
        "debug": False,
        "image_observation": False,
        "start_seed": 42
    }
    
    env = MetaDriveEnv(env_config)
    obs = env.reset()
    
    print(f"✅ 环境创建成功")
    print(f"   地图: {env_config['map']}")
    print(f"   随机种子: {env_config['start_seed']}")
    
    # 获取agent和车道信息
    agent = env.agent
    if agent is None:
        print("❌ 无法获取agent")
        return
    
    print(f"✅ Agent获取成功")
    print(f"   Agent类型: {type(agent).__name__}")
    
    # 检查车道信息
    if hasattr(agent, 'lane') and agent.lane:
        current_lane = agent.lane
        print(f"✅ 当前车道信息:")
        print(f"   车道类型: {type(current_lane).__name__}")
        print(f"   车道索引: {getattr(current_lane, 'index', 'N/A')}")
        print(f"   车道宽度: {getattr(current_lane, 'width', 'N/A')}")
        print(f"   车道长度: {getattr(current_lane, 'length', 'N/A')}")
        
        # 检查相邻车道
        left_lanes = getattr(current_lane, 'left_lanes', [])
        right_lanes = getattr(current_lane, 'right_lanes', [])
        
        print(f"   左相邻车道数量: {len(left_lanes)}")
        if left_lanes:
            print(f"   左相邻车道详情: {left_lanes}")
        
        print(f"   右相邻车道数量: {len(right_lanes)}")
        if right_lanes:
            print(f"   右相邻车道详情: {right_lanes}")
        
        # 检查车道索引
        if hasattr(agent, 'lane_index'):
            print(f"   Agent车道索引: {agent.lane_index}")
        else:
            print(f"   Agent车道索引: 未找到")
        
        # 检查导航信息
        if hasattr(agent, 'navigation') and agent.navigation:
            nav = agent.navigation
            print(f"✅ 导航信息:")
            print(f"   导航类型: {type(nav).__name__}")
            if hasattr(nav, 'current_lane'):
                print(f"   导航当前车道: {getattr(nav.current_lane, 'index', 'N/A')}")
            if hasattr(nav, 'current_ref_lanes'):
                print(f"   导航参考车道数量: {len(nav.current_ref_lanes) if nav.current_ref_lanes else 0}")
                
                # 详细检查参考车道
                if nav.current_ref_lanes:
                    print(f"   参考车道详情:")
                    for i, ref_lane in enumerate(nav.current_ref_lanes):
                        print(f"     车道{i}: {getattr(ref_lane, 'index', 'N/A')} (类型: {type(ref_lane).__name__})")
                        
                        # 检查每个参考车道的相邻车道
                        if hasattr(ref_lane, 'left_lanes'):
                            left_lanes = getattr(ref_lane, 'left_lanes', [])
                            print(f"       左相邻: {len(left_lanes)} 个")
                        if hasattr(ref_lane, 'right_lanes'):
                            right_lanes = getattr(ref_lane, 'right_lanes', [])
                            print(f"       右相邻: {len(right_lanes)} 个")
        
        # 检查道路网络信息
        if hasattr(env, 'map') and env.map:
            road_network = env.map.road_network
            print(f"✅ 道路网络信息:")
            print(f"   道路网络类型: {type(road_network).__name__}")
            
            # 获取所有车道索引
            if hasattr(road_network, 'indices'):
                all_lanes = road_network.indices
                print(f"   总车道数量: {len(all_lanes)}")
                print(f"   所有车道索引: {all_lanes[:5]}...")  # 只显示前5个
                
                # 检查是否有相邻车道关系
                for lane_idx in all_lanes[:3]:  # 检查前3个车道
                    try:
                        lane = road_network.get_lane(lane_idx)
                        if hasattr(lane, 'left_lanes'):
                            left_count = len(getattr(lane, 'left_lanes', []))
                            right_count = len(getattr(lane, 'right_lanes', []))
                            print(f"   车道 {lane_idx}: 左相邻={left_count}, 右相邻={right_count}")
                    except Exception as e:
                        print(f"   无法获取车道 {lane_idx} 信息: {e}")
        
    else:
        print("❌ 无法获取车道信息")
    
    # 测试车道变更检测
    print(f"\n🧪 测试车道变更检测...")
    
    # 模拟一些动作来观察车道变化
    for step in range(10):
        # 随机动作
        action = np.random.uniform(-1, 1, 2)
        try:
            result = env.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
            elif len(result) == 5:
                obs, reward, done, truncated, info = result
            else:
                print(f"   步数{step}: 环境返回了{len(result)}个值，跳过")
                continue
        except Exception as e:
            print(f"   步数{step}: 环境步进失败: {e}")
            break
        
        # 检查车道信息
        if hasattr(agent, 'lane') and agent.lane:
            current_lane = agent.lane
            current_lane_index = getattr(current_lane, 'index', None)
            
            if step == 0:
                print(f"   初始车道索引: {current_lane_index}")
                initial_lane_index = current_lane_index
            else:
                if current_lane_index != initial_lane_index:
                    print(f"   🚗 检测到车道变更! 步数{step}: {initial_lane_index} → {current_lane_index}")
                    break
                else:
                    print(f"   步数{step}: 车道索引未变化 ({current_lane_index})")
        
        if done:
            print(f"   环境在第{step}步结束")
            break
    
    # 关闭环境
    env.close()
    print(f"\n✅ 测试完成")

def test_force_lane_change():
    """测试强制车道变更"""
    print(f"\n🧪 测试强制车道变更...")
    
    # 创建环境
    env_config = {
        "num_scenarios": 1,
        "map": "SSSSS",  # 5段直线道路
        "traffic_density": 0.1,
        "use_render": False,
        "debug": False,
        "image_observation": False,
        "start_seed": 42
    }
    
    env = MetaDriveEnv(env_config)
    obs = env.reset()
    
    agent = env.agent
    if agent is None:
        print("❌ 无法获取agent")
        return
    
    print(f"✅ 开始强制车道变更测试")
    
    # 获取初始车道信息
    initial_lane = agent.lane
    initial_lane_index = getattr(initial_lane, 'index', None)
    print(f"   初始车道: {initial_lane_index}")
    
    # 尝试通过大转向来触发车道变更
    for step in range(20):
        # 使用大转向动作
        if step < 10:
            # 前10步：大左转
            action = np.array([-0.8, 0.5])  # 大左转 + 前进
        else:
            # 后10步：大右转
            action = np.array([0.8, 0.5])   # 大右转 + 前进
        
        try:
            result = env.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
            elif len(result) == 5:
                obs, reward, done, truncated, info = result
            else:
                continue
        except Exception as e:
            print(f"   步数{step}: 环境步进失败: {e}")
            break
        
        # 检查车道信息
        if hasattr(agent, 'lane') and agent.lane:
            current_lane = agent.lane
            current_lane_index = getattr(current_lane, 'index', None)
            
            if current_lane_index != initial_lane_index:
                print(f"   🚗 检测到车道变更! 步数{step}: {initial_lane_index} → {current_lane_index}")
                break
            else:
                print(f"   步数{step}: 车道索引未变化 ({current_lane_index})")
        
        # 检查是否冲出道路
        if info.get('out_of_road', False):
            print(f"   步数{step}: 车辆冲出道路")
            break
        
        if done:
            print(f"   环境在第{step}步结束")
            break
    
    # 关闭环境
    env.close()
    print(f"✅ 强制车道变更测试完成")

def test_lane_change_detection_methods():
    """测试不同的车道变更检测方法"""
    print(f"\n🧪 测试车道变更检测方法...")
    
    # 创建环境
    env_config = {
        "num_scenarios": 1,
        "map": "SSSSS",  # 5段直线道路
        "traffic_density": 0.1,
        "use_render": False,
        "debug": False,
        "image_observation": False,
        "start_seed": 42
    }
    
    env = MetaDriveEnv(env_config)
    obs = env.reset()
    
    agent = env.agent
    if agent is None:
        print("❌ 无法获取agent")
        return
    
    # 方法1: 检查info中的车道变更标志
    print(f"方法1: 检查info中的车道变更标志")
    # 先执行一步来获取info
    action = np.random.uniform(-1, 1, 2)
    try:
        result = env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
        elif len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            print(f"   ❌ 环境返回了{len(result)}个值")
            info = {}
    except Exception as e:
        print(f"   ❌ 环境步进失败: {e}")
        info = {}
    
    if 'lane_change' in info:
        print(f"   ✅ info中有lane_change字段: {info['lane_change']}")
    else:
        print(f"   ❌ info中没有lane_change字段")
        print(f"   info内容: {list(info.keys()) if isinstance(info, dict) else 'N/A'}")
    
    # 方法2: 检查车道索引变化
    print(f"方法2: 检查车道索引变化")
    if hasattr(agent, 'lane_index'):
        print(f"   ✅ agent有lane_index属性: {agent.lane_index}")
    else:
        print(f"   ❌ agent没有lane_index属性")
    
    if hasattr(agent, 'lane') and agent.lane:
        current_lane = agent.lane
        lane_index = getattr(current_lane, 'index', None)
        print(f"   ✅ agent.lane.index: {lane_index}")
    else:
        print(f"   ❌ 无法获取agent.lane.index")
    
    # 方法3: 检查相邻车道信息
    print(f"方法3: 检查相邻车道信息")
    if hasattr(agent, 'lane') and agent.lane:
        current_lane = agent.lane
        left_lanes = getattr(current_lane, 'left_lanes', [])
        right_lanes = getattr(current_lane, 'right_lanes', [])
        
        print(f"   左相邻车道: {len(left_lanes)} 个")
        print(f"   右相邻车道: {len(right_lanes)} 个")
        
        if left_lanes or right_lanes:
            print(f"   ✅ 检测到相邻车道，可以进行车道变更")
        else:
            print(f"   ❌ 没有相邻车道，无法进行车道变更")
    else:
        print(f"   ❌ 无法获取相邻车道信息")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    print("🚗 MetaDrive车道变更检测测试")
    print("=" * 50)
    
    try:
        test_lane_neighbor_info()
        test_lane_change_detection_methods()
        test_force_lane_change()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("测试完成") 