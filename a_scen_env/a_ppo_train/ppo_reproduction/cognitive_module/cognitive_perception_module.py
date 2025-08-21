
#!/usr/bin/env python3
"""
认知感知模块 - 雷达噪声注入器
严格在"归一化前"的雷达原始量测（米）上注入噪声，绝不在归一化后的obs上处理，更不改动世界真值。

核心原则：
A. 不修改环境中任意实体的真实state（位置/速度/朝向等）
B. 噪声注入位置：雷达传感器类输出原始距离阵列（米）之后、被ObservationManager归一化前
C. 保持obs结构不变，噪声在上游（传感器层）注入
D. Kalman/AR(1)等滤波在"米制的原始距离"上进行
E. 保留reset()清空滤波器状态
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import sys
import os
import time
import matplotlib.pyplot as plt

# 添加MetaDrive路径
sys.path.append('/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive')

from metadrive.component.sensors.lidar import Lidar
from metadrive.component.sensors.distance_detector import DistanceDetector

logger = logging.getLogger(__name__)


class PerceptNoiseLidar(Lidar):
    """
    带噪声的雷达传感器，继承MetaDrive的Lidar类
    在原始距离测量（米）上注入各种噪声模型
    """
    
    def __init__(self, engine, noise_config=None, parent_module=None):
        """
        初始化带噪声的雷达传感器
        
        Args:
            engine: MetaDrive引擎实例
            noise_config: 噪声配置字典
            parent_module: 父认知感知模块引用（用于可视化回调）
        """
        super().__init__(engine)
        
        # 噪声配置参数
        self.noise_config = noise_config or {}
        self.parent_module = parent_module  # 父模块引用
        
        # 高斯测距噪声：sigma(d) = sigma0 + k * d
        self.sigma0 = self.noise_config.get('sigma0', 0.1)  # 基础噪声（米）
        self.k = self.noise_config.get('k', 0.02)  # 距离相关系数
        
        # 漏检参数：p_miss(d) = p_miss0 * (1 + d/far_distance)
        self.p_miss0 = self.noise_config.get('p_miss0', 0.01)  # 基础漏检概率
        self.far_distance = self.noise_config.get('far_distance', 50.0)  # 远距离参考值
        
        # 误检参数
        self.p_false = self.noise_config.get('p_false', 0)  # 误检概率
        self.near_min = self.noise_config.get('near_min', 1.0)  # 误检最近距离
        self.near_max = self.noise_config.get('near_max', 5.0)  # 误检最远距离
        
        # 角度抖动
        self.angle_jitter_steps = self.noise_config.get('angle_jitter_steps', 0)  # 角度抖动束数
        
        # 时间相关性参数
        self.use_ar1 = self.noise_config.get('use_ar1', True)  # 是否使用AR(1)模型
        self.rho = self.noise_config.get('rho', 0.8)  # AR(1)系数
        self.use_lowpass = self.noise_config.get('use_lowpass', False)  # 是否使用低通滤波
        self.alpha = self.noise_config.get('alpha', 0.7)  # 低通滤波系数

        # === 新增：卡尔曼滤波（每束 1D-CV）相关参数 ===
        self.use_kf = self.noise_config.get('use_kf', True)            # 是否启用KF（如启用，将优先于低通）
        self.kf_dt = float(self.noise_config.get('kf_dt', 0.1))        # KF步长
        self.kf_q = float(self.noise_config.get('kf_q', 0.5))          # 白加速度谱密度（过程噪声）
        self.kf_sigma_a = float(self.noise_config.get('kf_sigma_a', 3.0))
        self.kf_q_scale = float(self.noise_config.get('kf_q_scale', 100.0))
        self.kf_r_floor = float(self.noise_config.get('kf_r_floor', 1e-4))  # 量测方差下限，防数值问题
        self.kf_init_std_pos = float(self.noise_config.get('kf_init_std_pos', 5.0))
        self.kf_init_std_vel = float(self.noise_config.get('kf_init_std_vel', 10.0))
        
        # 状态记录（每个beam独立）
        self.num_beams = 0  # 将在第一次perceive时初始化
        self.ar1_states = None  # AR(1)噪声状态
        self.prev_distances = None  # 上次滤波后距离
        self.initialized = False

        # === 新增：KF内部状态（每束 [r, r_dot] 与协方差2x2） ===
        self.kf_state = None  # shape: (N, 2)
        self.kf_P = None      # shape: (N, 2, 2)
        
        # === 新增：实时数据记录（用于可视化） ===
        self.last_original_distances = None  # 原始距离（米）
        self.last_noisy_distances = None     # 加噪后距离（米）
        self.last_noise_levels = None        # 每束的噪声强度
        self.front_beam_index = None         # 正前方雷达束索引
        
        # === 新增：时序数据记录（用于图表生成） ===
        self.front_beam_history = {
            'original_distances': [],         # 正前方雷达束原始距离时序
            'noisy_distances': [],           # 正前方雷达束加噪距离时序
            'kf_estimates': [],              # 卡尔曼滤波估计时序
            'timestamps': []                 # 时间戳
        }
        
        # === 新增：噪声裁剪统计 ===
        self._clip_log_counter = 0           # 噪声裁剪计数器
        
        logger.info(f"PerceptNoiseLidar初始化: sigma0={self.sigma0:.3f}, k={self.k:.3f}, "
                   f"p_miss0={self.p_miss0:.3f}, p_false={self.p_false:.3f}")
    
    def _initialize_states(self, num_beams: int):
        """初始化状态变量"""
        if self.num_beams != num_beams or not self.initialized:
            self.num_beams = num_beams
            self.ar1_states = np.zeros(num_beams, dtype=np.float32)
            self.prev_distances = np.ones(num_beams, dtype=np.float32) * 50.0  # 初始为最大距离
            self.initialized = True

            # === 新增：初始化每束KF状态 ===
            if self.use_kf:
                self.kf_state = np.zeros((num_beams, 2), dtype=np.float32)
                self.kf_state[:, 0] = 50.0  # 初始距离置为量程上限（与上面prev一致）
                self.kf_state[:, 1] = 0.0   # 初始径向速度为0
                self.kf_P = np.zeros((num_beams, 2, 2), dtype=np.float32)
                self.kf_P[:, 0, 0] = self.kf_init_std_pos ** 2
                self.kf_P[:, 1, 1] = self.kf_init_std_vel ** 2

            logger.info(f"PerceptNoiseLidar状态初始化: {num_beams} beams")
    
    def reset(self):
        """重置所有滤波器状态"""
        if self.initialized:
            self.ar1_states.fill(0.0)
            self.prev_distances.fill(50.0)

            # === 新增：重置KF状态 ===
            if self.use_kf and (self.kf_state is not None):
                self.kf_state[:, 0] = 50.0
                self.kf_state[:, 1] = 0.0
                if self.kf_P is not None:
                    self.kf_P[:, 0, 0] = self.kf_init_std_pos ** 2
                    self.kf_P[:, 1, 1] = self.kf_init_std_vel ** 2

            logger.info("PerceptNoiseLidar状态已重置")
        
        # === 新增：清空时序数据记录 ===
        self.front_beam_history = {
            'original_distances': [],
            'noisy_distances': [],
            'kf_estimates': [],
            'timestamps': []
        }
    
    def perceive(self, base_vehicle, physics_world, num_lasers, distance, 
                 height=None, detector_mask=None, show=False):
        """
        重写perceive方法，在原始距离上注入噪声
        
        Args:
            base_vehicle: 基础车辆
            physics_world: 物理世界
            num_lasers: 激光数量
            distance: 最大探测距离（米）
            height: 高度
            detector_mask: 探测器掩码
            show: 是否显示
            
        Returns:
            (cloud_points, detected_objects): 处理后的距离数组和检测对象
        """
        # 初始化状态
        self._initialize_states(num_lasers)
        
        # 设置正前方雷达束索引（通常是中间的束）
        if self.front_beam_index is None:
            self.front_beam_index = num_lasers // 2
        
        # 调用父类方法获取原始观测
        result, detected_objects = super().perceive(
            base_vehicle, physics_world, num_lasers, distance, 
            height, detector_mask, show
        )
        
        # 获取原始归一化距离数组（0-1）
        normalized_distances = np.array(result, dtype=np.float32)
        
        # 转换为实际距离（米）
        actual_distances = normalized_distances * distance
        
        # 记录原始距离
        self.last_original_distances = actual_distances.copy()
        
        # 在实际距离上应用噪声模型
        noisy_distances = self._apply_noise_models(actual_distances, distance)
        
        # 记录加噪后距离
        self.last_noisy_distances = noisy_distances.copy()
        
        # 计算噪声强度（原始距离与加噪距离的差值）
        self.last_noise_levels = np.abs(noisy_distances - actual_distances)
        
        # === 新增：记录正前方雷达束的时序数据 ===
        if self.front_beam_index is not None:
            import time
            self.front_beam_history['original_distances'].append(float(actual_distances[self.front_beam_index]))
            self.front_beam_history['noisy_distances'].append(float(noisy_distances[self.front_beam_index]))
            self.front_beam_history['timestamps'].append(time.time())
            
            # 如果启用了卡尔曼滤波，记录滤波结果
            if self.use_kf and self.kf_state is not None:
                kf_estimate = float(self.kf_state[self.front_beam_index, 0])  # 位置估计
                self.front_beam_history['kf_estimates'].append(kf_estimate)
            else:
                # 如果没有卡尔曼滤波，使用加噪距离作为"估计"
                self.front_beam_history['kf_estimates'].append(float(noisy_distances[self.front_beam_index]))
        
        # 转换回归一化格式 [0, 1]
        normalized_noisy = np.clip(noisy_distances / distance, 0.0, 1.0)
        
        # === 更新雷达束可视化 ===
        if self.parent_module and hasattr(self.parent_module, '_update_radar_visualization'):
            try:
                # 获取环境引用用于可视化更新
                env = getattr(self.parent_module, 'attached_env', None)
                if env:
                    self.parent_module._update_radar_visualization(env)
            except Exception as e:
                # 可视化失败不应影响主要功能
                pass
        
        return normalized_noisy.tolist(), detected_objects
    
    def get_front_beam_data(self):
        """
        获取正前方雷达束的实时数据
        
        Returns:
            dict: {
                'original_distance': float,      # 原始距离（米）
                'noisy_distance': float,         # 加噪后距离（米）
                'noise_level': float,            # 噪声强度（米）
                'beam_index': int                # 雷达束索引
            }
        """
        if (self.last_original_distances is None or 
            self.last_noisy_distances is None or 
            self.last_noise_levels is None or
            self.front_beam_index is None):
            return {
                'original_distance': 0.0,
                'noisy_distance': 0.0,
                'noise_level': 0.0,
                'beam_index': -1
            }
        
        return {
            'original_distance': float(self.last_original_distances[self.front_beam_index]),
            'noisy_distance': float(self.last_noisy_distances[self.front_beam_index]),
            'noise_level': float(self.last_noise_levels[self.front_beam_index]),
            'beam_index': int(self.front_beam_index)
        }
    
    def get_front_beam_history(self):
        """
        获取正前方雷达束的时序数据（用于图表生成）
        
        Returns:
            dict: {
                'original_distances': list,      # 原始距离时序（米）
                'noisy_distances': list,         # 加噪距离时序（米）
                'kf_estimates': list,            # 卡尔曼滤波估计时序（米）
                'timestamps': list,              # 时间戳时序
                'length': int                    # 数据长度
            }
        """
        return {
            'original_distances': self.front_beam_history['original_distances'].copy(),
            'noisy_distances': self.front_beam_history['noisy_distances'].copy(),
            'kf_estimates': self.front_beam_history['kf_estimates'].copy(),
            'timestamps': self.front_beam_history['timestamps'].copy(),
            'length': len(self.front_beam_history['original_distances'])
        }

    def get_latest_processed_distances(self):
        """
        获取最新的处理后雷达距离数组（供认知偏差模块使用）
        
        Returns:
            np.ndarray or None: 最新的处理后雷达距离数组（米），如果不可用则返回None
        """
        if self.last_noisy_distances is not None:
            return self.last_noisy_distances.copy()
        return None

    # === 新增：每束 1D-CV 卡尔曼滤波实现（状态 [r, r_dot]，量测 z=r） ===
    def _kf_filter(self, z_distances: np.ndarray, sigma_array: np.ndarray, return_var: bool = False):
        """
        对每束距离做 1D-CV KF。
        z_distances: 当前帧各束的“带噪量测”（米）
        sigma_array: 各束测量标准差（米），R_i = max(sigma_i^2, kf_r_floor)
        返回：滤波后的距离（米）
        """
        if not self.use_kf or self.kf_state is None or self.kf_P is None:
            if return_var:
                return z_distances, np.zeros_like(z_distances)
            return z_distances

        dt = float(self.kf_dt)
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float32)
        dt2, dt3, dt4 = dt*dt, dt**3, dt**4
        # q = float(self.kf_q)
        q = self.kf_sigma_a ** 2
        Q_base = np.array([[dt4/4.0, dt3/2.0],
                      [dt3/2.0, dt2      ]], dtype=np.float32) * q
        Q = float(self.kf_q_scale or 1.0) * Q_base   # 建议 10~50 起步
        H = np.array([[1.0, 0.0]], dtype=np.float32)  # z = [1,0]@[r,r_dot]

        out = np.empty_like(z_distances, dtype=np.float32)
        var_out = np.empty_like(z_distances, dtype=np.float32) if return_var else None

        for i in range(self.num_beams):
            # 预测
            x = self.kf_state[i]   # [r, r_dot]
            P = self.kf_P[i]       # 2x2
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # 测量
            z = float(z_distances[i])
            sigma_i = max(float(sigma_array[i]), 1e-6)
            R = np.array([[max(sigma_i * sigma_i, self.kf_r_floor)]], dtype=np.float32)

            # 更新
            y = np.array([[z]]) - (H @ x_pred).reshape(1, 1)
            S = H @ P_pred @ H.T + R
            try:
                K = (P_pred @ H.T) @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = (P_pred @ H.T) @ np.linalg.pinv(S)

            x_new = x_pred + (K @ y).reshape(2)
            I_KH = np.eye(2, dtype=np.float32) - (K @ H)
            P_new = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

            # 写回
            self.kf_state[i] = x_new
            self.kf_P[i] = 0.5 * (P_new + P_new.T)  # 保证对称
            out[i] = x_new[0]  # 输出距离
            
            if return_var:
                var_out[i] = P_new[0, 0]  # 位置方差

        if return_var:
            return out, var_out
        return out
    
    def _apply_noise_models(self, distances: np.ndarray, max_range: float) -> np.ndarray:
        """
        在实际距离（米）上应用各种噪声模型
        
        Args:
            distances: 原始距离数组（米）
            max_range: 最大探测距离（米）
            
        Returns:
            处理后的距离数组（米）
        """
        noisy_distances = distances.copy()
        
        # 1. 高斯测距噪声：sigma(d) = sigma0 + k * d
        if self.sigma0 > 0 or self.k > 0:
            sigma_array = self.sigma0 + self.k * distances
            gaussian_noise = np.random.normal(0, sigma_array)
            
            # === 新增：基于3σ的噪声裁剪，避免极端异常值 ===
            # 将噪声限制在±3σ范围内，覆盖99.7%的正常情况
            noise_limit = 3.0 * sigma_array
            original_noise = gaussian_noise.copy()
            gaussian_noise = np.clip(gaussian_noise, -noise_limit, noise_limit)
            
            # 检查是否有噪声被裁剪（用于调试）
            clipped_count = np.sum(np.abs(original_noise) > noise_limit)
            if clipped_count > 0:
                self._clip_log_counter = getattr(self, '_clip_log_counter', 0) + 1
                # 每50次记录一次，避免日志过多
                if self._clip_log_counter % 50 == 0 or self._clip_log_counter <= 5:
                    max_original = np.max(np.abs(original_noise))
                    max_clipped = np.max(np.abs(gaussian_noise))
                    max_sigma = np.max(sigma_array)
                    logger.info(f"🔧 噪声裁剪第{self._clip_log_counter}次: {clipped_count}/{len(distances)}束被裁剪, "
                               f"最大噪声{max_original:.2f}→{max_clipped:.2f}米 (3σ={3*max_sigma:.2f}米)")
            
            # AR(1)时间相关性
            if self.use_ar1 and self.rho > 0:
                # n_t = ρ * n_{t-1} + sqrt(1-ρ^2) * ξ_t
                self.ar1_states = (self.rho * self.ar1_states + 
                                  np.sqrt(1 - self.rho**2) * gaussian_noise)
                gaussian_noise = self.ar1_states
                
                # AR(1)后再次应用3σ裁剪（因为AR(1)可能会累积极端值）
                gaussian_noise = np.clip(gaussian_noise, -noise_limit, noise_limit)
            
            noisy_distances += gaussian_noise
        
        # 2. 漏检：以p_miss(d)概率将该束置为max_range
        if self.p_miss0 > 0:
            p_miss = self.p_miss0 * (1 + distances / self.far_distance)
            miss_mask = np.random.random(len(distances)) < p_miss
            noisy_distances[miss_mask] = max_range
        
        # 3. 误检：以p_false概率将该束覆写为U(near_min, near_max)
        if self.p_false > 0:
            false_mask = np.random.random(len(distances)) < self.p_false
            false_distances = np.random.uniform(
                self.near_min, self.near_max, size=np.sum(false_mask)
            )
            noisy_distances[false_mask] = false_distances
        
        # 4. 角度抖动：整段距离数组做小幅roll
        if self.angle_jitter_steps > 0:
            jitter = np.random.randint(-self.angle_jitter_steps, self.angle_jitter_steps + 1)
            if jitter != 0:
                noisy_distances = np.roll(noisy_distances, jitter)
        
        # === 新增：5. 卡尔曼滤波（若启用，则优先于低通） ===
        if self.use_kf:
            # 测量方差使用 sigma(d)^2（基于“原始距离”distances，而非加噪后）
            sigma_meas = self.sigma0 + self.k * distances
            sigma_meas = np.maximum(sigma_meas, 1e-6)
            noisy_distances = self._kf_filter(noisy_distances, sigma_meas)
        
        # 6. 时间低通滤波（可选，且仅在未启用KF时生效）
        elif self.use_lowpass and self.alpha > 0:
            # d_filt = α * d_noisy + (1-α) * d_prev
            noisy_distances = (self.alpha * noisy_distances + 
                             (1 - self.alpha) * self.prev_distances)
            self.prev_distances = noisy_distances.copy()
        
        # 确保距离在合理范围内
        noisy_distances = np.clip(noisy_distances, 0.0, max_range)

        # （保持与原逻辑一致）记录prev供低通使用/观察
        self.prev_distances = noisy_distances.copy()
        
        return noisy_distances

    # === 可视化功能 ===
    
    @staticmethod
    def _ensure_figdir(save_dir: str) -> str:
        """确保图片保存目录存在"""
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
    
    def plot_noise_comparison(self, d_true: np.ndarray, d_noisy: np.ndarray, savepath: Optional[str] = None):
        """
        绘制噪声效果对比图：原始距离 vs 加噪后距离 + 噪声分布直方图
        
        Args:
            d_true: 原始真实距离数组（米）
            d_noisy: 加噪后距离数组（米）
            savepath: 可选的保存路径
        """
        assert d_true.shape == d_noisy.shape, "距离数组形状必须一致"
        
        noise = d_noisy - d_true
        x = np.arange(len(d_true))
        
        fig = plt.figure(figsize=(12, 5))
        
        # 子图1：距离对比
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(x, d_true, 'b-', label="True Distance (m)", linewidth=2)
        ax1.plot(x, d_noisy, 'r--', marker="o", markersize=3, label="Noisy Distance (m)", alpha=0.7)
        ax1.set_xlabel("Beam Index")
        ax1.set_ylabel("Distance (m)")
        ax1.set_title("Before vs After Noise Injection")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        
        # 子图2：噪声分布直方图
        ax2 = fig.add_subplot(1, 2, 2)
        n, bins, patches = ax2.hist(noise, bins=30, density=True, alpha=0.7, color='orange', edgecolor='black')
        
        # 叠加高斯拟合曲线
        if len(noise) > 1:
            mu = np.mean(noise)
            sigma = np.std(noise)
            x_gauss = np.linspace(bins[0], bins[-1], 100)
            y_gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_gauss - mu) / sigma) ** 2)
            ax2.plot(x_gauss, y_gauss, 'r-', linewidth=2, label=f"Gaussian Fit (μ={mu:.3f}, σ={sigma:.3f})")
            ax2.legend()
        
        ax2.set_xlabel("Noise (m)")
        ax2.set_ylabel("Density")
        ax2.set_title("Distribution of Injected Noise")
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            logger.info(f"噪声对比图已保存到: {savepath}")
        
        plt.close(fig)
    
    def plot_kf_timeseries(self, gt: np.ndarray, meas: np.ndarray, kf: np.ndarray, 
                          kf_std: Optional[np.ndarray] = None, savepath: Optional[str] = None):
        """
        绘制卡尔曼滤波效果时序图：Ground Truth, Noisy Measurement, KF Estimate（含±1σ不确定性带）
        
        Args:
            gt: 地面真值时间序列（米）
            meas: 噪声测量时间序列（米）
            kf: 卡尔曼滤波估计时间序列（米）
            kf_std: 可选的卡尔曼滤波标准差（米）
            savepath: 可选的保存路径
        """
        assert len(gt) == len(meas) == len(kf), "时间序列长度必须一致"
        
        t = np.arange(len(gt))
        
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        # 绘制三条曲线
        ax.plot(t, gt, 'g-', linewidth=2, label="Ground Truth (m)")
        ax.scatter(t, meas, s=20, alpha=0.6, color='red', label="Noisy Measurement (m)")
        ax.plot(t, kf, 'b-', linewidth=2, label="KF Estimate (m)")
        
        # 绘制不确定性带
        if kf_std is not None:
            ax.fill_between(t, kf - kf_std, kf + kf_std, alpha=0.3, color='blue', label="KF ±1σ")
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Distance (m)")
        ax.set_title("Kalman Filter: Prediction / Observation / Update")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            logger.info(f"卡尔曼滤波时序图已保存到: {savepath}")
        
        plt.close(fig)
    
    def visualize_noise_and_kf_once(self, distances_true_m: np.ndarray, max_range: float, save_dir: str = "figs"):
        """
        便捷可视化入口：对给定真实距离执行噪声注入和KF滤波，生成对比图
        
        Args:
            distances_true_m: 真实距离数组（米）
            max_range: 最大探测距离（米）
            save_dir: 图片保存目录
        """
        self._ensure_figdir(save_dir)
        timestamp = int(time.time())
        
        # 初始化状态
        self._initialize_states(len(distances_true_m))
        
        # 1. 噪声对比图
        d_noisy = self._apply_noise_models(distances_true_m.copy(), max_range)
        noise_path = os.path.join(save_dir, f"noise_compare_{timestamp}.png")
        self.plot_noise_comparison(distances_true_m, d_noisy, noise_path)
        
        # 2. 构造时间序列进行KF测试
        T = 200  # 时间步数
        mean_dist = np.mean(distances_true_m)
        
        # 构造简单的正弦波动
        t_seq = np.arange(T)
        gt_seq = mean_dist + 5 * np.sin(0.1 * t_seq)  # 围绕均值波动±5米
        gt_seq = np.clip(gt_seq, 0.0, max_range)
        
        # 生成测量序列和KF估计序列
        meas_seq = np.zeros(T)
        kf_seq = np.zeros(T)
        kf_var_seq = np.zeros(T)
        
        # 重置KF状态为单束测试
        self._initialize_states(1)
        
        for i in range(T):
            # 加噪测量
            d_true_single = np.array([gt_seq[i]])
            d_meas_single = self._apply_noise_models(d_true_single, max_range)
            meas_seq[i] = d_meas_single[0]
            
            # KF滤波
            if self.use_kf:
                sigma_array = np.array([self.sigma0 + self.k * gt_seq[i]])
                kf_result, kf_var = self._kf_filter(d_meas_single, sigma_array, return_var=True)
                kf_seq[i] = kf_result[0]
                kf_var_seq[i] = kf_var[0]
            else:
                kf_seq[i] = meas_seq[i]
                kf_var_seq[i] = 0.0
        
        # 绘制KF时序图
        kf_std_seq = np.sqrt(kf_var_seq) if self.use_kf else None
        kf_path = os.path.join(save_dir, f"kf_timeseries_{timestamp}.png")
        self.plot_kf_timeseries(gt_seq, meas_seq, kf_seq, kf_std_seq, kf_path)
        
        logger.info(f"可视化完成，图片保存在: {save_dir}")

    @classmethod 
    def get_visualization_dir_from_env(cls, env=None):
        """
        从环境获取可视化目录路径，并创建cog_influence子文件夹
        
        Args:
            env: TrajectoryReplayEnvCognitive实例，如果为None则使用默认路径
            
        Returns:
            str: cog_influence文件夹的完整路径
        """
        if env and hasattr(env, 'visualization_output_dir'):
            base_dir = env.visualization_output_dir
        else:
            # 默认路径
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = f"/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/fig_cog/cognitive_analysis_{timestamp}"
        
        # 创建cog_influence子文件夹
        cog_influence_dir = os.path.join(base_dir, "cog_influence")
        os.makedirs(cog_influence_dir, exist_ok=True)
        
        return cog_influence_dir


# StandaloneNoiseProcessor类已被移除，仅使用运行时的PerceptNoiseLidar实例


class CognitivePerceptionModule:
    """
    认知感知模块 - 主接口类
    负责将噪声雷达集成到MetaDrive环境中
    """
    
    def __init__(self, noise_config: Optional[Dict[str, Any]] = None):
        """
        初始化认知感知模块
        
        Args:
            noise_config: 噪声配置参数
        """
        self.noise_config = noise_config or self._get_default_config()
        self.original_lidar = None  # 保存原始雷达传感器引用
        self.noise_lidar = None  # 噪声雷达传感器实例
        self.attached_env = None  # 附加的环境实例
        
        # 可视化记录变量（用于兼容现有接口）
        self._last_true_x = 0.0
        self._last_true_y = 0.0
        self._last_noisy_x = 0.0
        self._last_noisy_y = 0.0
        self._last_filtered_x = 0.0
        self._last_filtered_y = 0.0
        self._last_effective_sigma = 0.0
        
        # === 新增：雷达束可视化 ===
        self.radar_visualization_enabled = False
        self.radar_beam_node = None  # 存储雷达束的可视化节点
        self.radar_beam_drawer = None  # 线条绘制器
        
        logger.info("CognitivePerceptionModule初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认噪声配置"""
        return {
            'sigma0': 0.1,          # 基础高斯噪声（米）
            'k': 0.02,              # 距离相关系数
            'p_miss0': 0.01,        # 基础漏检概率
            'far_distance': 50.0,   # 远距离参考值
            'p_false': 0.0001,      # 误检概率（保守值）
            'near_min': 1.0,        # 误检最近距离
            'near_max': 5.0,        # 误检最远距离
            'angle_jitter_steps': 1, # 角度抖动束数
            'use_ar1': True,        # 启用AR(1)模型
            'rho': 0.8,             # AR(1)系数
            'use_lowpass': False,   # 启用低通滤波
            'alpha': 0.7,           # 低通滤波系数

            # === 新增：KF 默认配置（不改动原有键，仅新增） ===
            'use_kf': True,
            'kf_dt': 0.1,
            'kf_q': 0.5,
            'kf_sigma_a': 3.0,
            'kf_q_scale': 100.0,
            'kf_r_floor': 1e-4,
            'kf_init_std_pos': 5.0,
            'kf_init_std_vel': 10.0,
        }
    
    def attach_to_env(self, env):
        """
        将噪声雷达附加到MetaDrive环境
        
        Args:
            env: MetaDrive环境实例
        """
        try:
            self.attached_env = env
            
            # 检查环境是否有传感器管理器
            if not hasattr(env, 'engine') or not hasattr(env.engine, 'get_sensor'):
                logger.warning("环境不支持传感器管理，跳过雷达噪声注入")
                return False
            
            # 获取原始雷达传感器
            try:
                self.original_lidar = env.engine.get_sensor("lidar")
                if self.original_lidar is None:
                    logger.warning("环境中未找到雷达传感器")
                    return False
            except:
                logger.warning("无法获取雷达传感器")
                return False
            
            # 创建噪声雷达传感器
            self.noise_lidar = PerceptNoiseLidar(env.engine, self.noise_config, parent_module=self)
            
            # 替换环境中的雷达传感器
            # 注意：这里我们替换的是sensor manager中的实例
            env.engine.sensors["lidar"] = self.noise_lidar
            
            # 设置雷达束可视化
            self._setup_radar_visualization(env)
            
            logger.info("✅ 噪声雷达已成功附加到环境")
            return True
            
        except Exception as e:
            logger.error(f"❌ 附加噪声雷达失败: {e}")
            return False
    
    def reset(self):
        """重置认知感知模块状态"""
        if self.noise_lidar:
            self.noise_lidar.reset()
        
        # 重置可视化变量
        self._last_true_x = 0.0
        self._last_true_y = 0.0
        self._last_noisy_x = 0.0
        self._last_noisy_y = 0.0
        self._last_filtered_x = 0.0
        self._last_filtered_y = 0.0
        self._last_effective_sigma = 0.0
    
    def get_front_beam_info(self):
        """
        获取正前方雷达束的实时信息（供可视化使用）
        
        Returns:
            dict: 包含正前方雷达的原始距离、加噪距离和噪声强度
        """
        return self.noise_lidar.get_front_beam_data()
    
    def get_processed_radar_distances(self):
        """
        获取认知感知模块处理后的完整雷达距离数组（供认知偏差模块使用）
        
        Returns:
            np.ndarray or None: 处理后的雷达距离数组（米），如果不可用则返回None
        """
        if self.noise_lidar and hasattr(self.noise_lidar, 'get_latest_processed_distances'):
            return self.noise_lidar.get_latest_processed_distances()
        return None
    
    def enable_radar_visualization(self, enable: bool = True):
        """
        启用或禁用雷达束可视化
        
        Args:
            enable: 是否启用可视化
        """
        self.radar_visualization_enabled = enable
        if enable:
            logger.info("🎯 雷达束可视化已启用")
        else:
            logger.info("🎯 雷达束可视化已禁用")
            self._cleanup_radar_visualization()
    
    def _setup_radar_visualization(self, env):
        """
        设置雷达束可视化
        
        Args:
            env: MetaDrive环境实例
        """
        if not self.radar_visualization_enabled:
            return
        
        try:
            # 获取引擎实例用于创建线条绘制器
            engine = env.engine
            
            # 创建线条绘制器
            self.radar_beam_drawer = engine.make_line_drawer(parent_node=engine.render, thickness=3.0)
            
            logger.info("🎯 雷达束可视化设置完成")
            
        except Exception as e:
            logger.error(f"❌ 雷达束可视化设置失败: {e}")
            self.radar_visualization_enabled = False
    
    def _update_radar_visualization(self, env):
        """
        更新雷达束可视化
        
        Args:
            env: MetaDrive环境实例
        """
        if not self.radar_visualization_enabled or not self.radar_beam_drawer:
            return
        
        try:
            # 获取车辆位置和朝向
            agent = env.agent
            position = agent.position  # [x, y]
            heading = agent.heading_theta  # 弧度
            
            # 计算正前方50米的终点坐标
            import numpy as np
            radar_range = 50.0  # 雷达束长度50米
            
            # 计算终点位置
            end_x = position[0] + radar_range * np.cos(heading)
            end_y = position[1] + radar_range * np.sin(heading)
            
            # 设置高度（稍微抬高以便可见）
            start_height = 1.5  # 车辆高度
            end_height = 1.5
            
            # 清除之前的线条
            self.radar_beam_drawer.reset()
            
            # 绘制雷达束线条
            line_points = [
                [position[0], position[1], start_height],  # 起点
                [end_x, end_y, end_height]  # 终点
            ]
            
            # 设置颜色（红色，半透明）
            line_colors = [
                [1.0, 0.0, 0.0, 0.8],  # 红色，透明度0.8
            ]
            
            # 绘制线条
            self.radar_beam_drawer.draw_lines([line_points], [line_colors])
            
        except Exception as e:
            logger.error(f"❌ 雷达束可视化更新失败: {e}")
    
    def _cleanup_radar_visualization(self):
        """
        清理雷达束可视化资源
        """
        try:
            if self.radar_beam_drawer:
                self.radar_beam_drawer.reset()
                # 注意：ColorLineNodePath可能没有destroy方法，使用remove等方法
                if hasattr(self.radar_beam_drawer, 'removeNode'):
                    self.radar_beam_drawer.removeNode()
                self.radar_beam_drawer = None
            
            self.radar_beam_node = None
            logger.info("🎯 雷达束可视化资源已清理")
            
        except Exception as e:
            logger.error(f"❌ 雷达束可视化清理失败: {e}")

    
    def detach_from_env(self):
        """从环境中分离噪声雷达，恢复原始传感器"""
        if self.attached_env and self.original_lidar:
            try:
                self.attached_env.engine.sensors["lidar"] = self.original_lidar
                logger.info("✅ 已恢复原始雷达传感器")
            except:
                logger.warning("⚠️ 恢复原始雷达传感器失败")
        
        # 清理雷达束可视化
        self._cleanup_radar_visualization()
        
        self.attached_env = None
        self.noise_lidar = None
    
    # === 兼容性接口（保留用于向后兼容） ===
    
    def process_vehicle_state(self, agent, ego_state=None, is_ppo_mode=False):
        """
        兼容性方法：不再修改车辆状态
        现在噪声在传感器层注入，此方法保留但不执行任何操作
        """
        # 不再修改agent状态，噪声已在传感器层处理
        pass
    
    def process_observation(self, obs, ego_state=None, is_ppo_mode=False):
        """
        兼容性方法：不再处理已归一化的观测
        现在噪声在传感器层注入，直接返回观测不做修改
        """
        # 不再处理已归一化的观测，直接返回
        return obs
    
    @property
    def sigma(self):
        """兼容性属性：返回基础噪声强度"""
        return self.noise_config.get('sigma0', 0.1)
    
    @property
    def enable_kalman(self):
        """兼容性属性：返回是否启用时间相关性"""
        return self.noise_config.get('use_ar1', True) or self.noise_config.get('use_lowpass', False)
    
    def generate_visualization(self, save_dir, env=None, test_distances=None, max_range=50.0):
        """
        生成认知感知模块的可视化图表
        
        Args:
            env: TrajectoryReplayEnvCognitive实例（用于获取输出目录）
            test_distances: 测试距离数组，如果为None则生成默认测试数据
            max_range: 最大探测距离
        """
        # 获取保存目录
 
        # 优先使用真实的雷达时序数据，如果没有则生成测试数据
        use_real_data = False
        if test_distances is None:
            # 尝试从运行时数据获取真实雷达数据
            if (self.noise_lidar and 
                hasattr(self.noise_lidar, 'get_front_beam_history')):
                history = self.noise_lidar.get_front_beam_history()
                if history['length'] > 0:
                    # 使用真实的雷达时序数据
                    logger.info(f"✅ 使用真实雷达数据生成可视化：{history['length']}步时序数据")
                    use_real_data = True
                    real_data = history
                else:
                    logger.warning("⚠️ 真实雷达数据不足，使用模拟数据")
            
            if not use_real_data:
                # 生成模拟雷达数据（回退方案）
                logger.info("🔄 使用模拟数据生成可视化")
                num_beams = 240
                angles = np.linspace(0, 2*np.pi, num_beams, endpoint=False)
                test_distances = np.ones(num_beams) * max_range
                
                # 在某些角度放置"虚拟车辆"
                vehicle_angles = [0, np.pi/6, np.pi/4, np.pi/2, np.pi]
                for angle in vehicle_angles:
                    idx = np.argmin(np.abs(angles - angle))
                    test_distances[idx-2:idx+3] = 15.0 + 5*np.random.randn()
                    
                # 一些中等距离的障碍物
                for i in range(20, 40):
                    test_distances[i] = 25.0 + 3*np.random.randn()
                    
                test_distances = np.clip(test_distances, 0.5, max_range)
        
        # 仅使用运行时的PerceptNoiseLidar实例
        if self.noise_lidar is None:
            
            logger.error("❌ PerceptNoiseLidar实例不存在，无法生成可视化")
            raise RuntimeError("PerceptNoiseLidar实例不存在，请确保认知感知模块已正确附加到环境")
        
        if not hasattr(self.noise_lidar, 'plot_noise_comparison'):
            logger.error("❌ PerceptNoiseLidar实例缺少可视化方法")
            raise RuntimeError("PerceptNoiseLidar实例缺少必要的可视化方法")
        
        processor = self.noise_lidar
        logger.info("✅ 使用运行时PerceptNoiseLidar实例生成可视化")
        
        # 确保processor有正确的_initialize_states方法
        if not hasattr(processor, '_initialize_states'):
            logger.error("❌ PerceptNoiseLidar实例缺少_initialize_states方法")
            raise RuntimeError("PerceptNoiseLidar实例缺少_initialize_states方法")
        
        # 生成时间戳
        timestamp = int(time.time())
        
        # 确保保存目录存在
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if use_real_data:
            # === 使用真实雷达数据生成图表 ===
            
            # 1. 噪声对比图 - 使用真实数据的时序快照
            orig_distances = np.array(real_data['original_distances'])
            noisy_distances = np.array(real_data['noisy_distances'])
            
            noise_path = os.path.join(save_dir, f"noise_compare_{timestamp}.png")
            processor.plot_noise_comparison(orig_distances, noisy_distances, noise_path)
            
            # 2. KF时序图 - 使用真实的时序数据
            gt_seq = np.array(real_data['original_distances'])
            meas_seq = np.array(real_data['noisy_distances'])
            kf_seq = np.array(real_data['kf_estimates'])
            
            # 计算KF不确定性（简化估计）
            kf_std_seq = None
            if len(gt_seq) > 1:
                residuals = np.abs(kf_seq - gt_seq)
                avg_residual = np.mean(residuals)
                kf_std_seq = np.full_like(kf_seq, avg_residual)
            
            kf_path = os.path.join(save_dir, f"kf_timeseries_{timestamp}.png")
            processor.plot_kf_timeseries(gt_seq, meas_seq, kf_seq, kf_std_seq, kf_path)
            
        else:
            # === 使用模拟数据生成图表（回退方案） ===
            
            processor._initialize_states(len(test_distances))
            
            # 1. 噪声对比图
            d_noisy = processor._apply_noise_models(test_distances.copy(), max_range)
            noise_path = os.path.join(save_dir, f"noise_compare_{timestamp}.png")
            processor.plot_noise_comparison(test_distances, d_noisy, noise_path)
            
            # 2. KF时序图 - 使用与运行时相同的配置
            T = 200
            mean_dist = np.mean(test_distances)
            t_seq = np.arange(T)
            gt_seq = mean_dist + 5 * np.sin(0.1 * t_seq)
            gt_seq = np.clip(gt_seq, 0.0, max_range)
            
            meas_seq = np.zeros(T)
            kf_seq = np.zeros(T)
            kf_var_seq = np.zeros(T)
            
            # 重新初始化为单束测试
            processor._initialize_states(1)
            
            for i in range(T):
                d_true_single = np.array([gt_seq[i]])
                d_meas_single = processor._apply_noise_models(d_true_single, max_range)
                meas_seq[i] = d_meas_single[0]
                
                # 使用运行时相同的KF配置
                if hasattr(processor, 'use_kf') and processor.use_kf:
                    sigma_array = np.array([processor.sigma0 + processor.k * gt_seq[i]])
                    kf_result, kf_var = processor._kf_filter(d_meas_single, sigma_array, return_var=True)
                    kf_seq[i] = kf_result[0]
                    kf_var_seq[i] = kf_var[0]
                else:
                    kf_seq[i] = meas_seq[i]
                    kf_var_seq[i] = 0.0
            
            kf_std_seq = np.sqrt(kf_var_seq) if (hasattr(processor, 'use_kf') and processor.use_kf) else None
            kf_path = os.path.join(save_dir, f"kf_timeseries_{timestamp}.png")
            processor.plot_kf_timeseries(gt_seq, meas_seq, kf_seq, kf_std_seq, kf_path)
        
        # 输出运行时配置信息
        if self.noise_lidar is not None:
            logger.info(f"运行时噪声配置: sigma0={getattr(processor, 'sigma0', 'N/A'):.3f}, "
                       f"k={getattr(processor, 'k', 'N/A'):.3f}, "
                       f"p_false={getattr(processor, 'p_false', 'N/A'):.3f}, "
                       f"use_kf={getattr(processor, 'use_kf', 'N/A')}")
        
        logger.info(f"认知感知可视化完成，图片保存在: {save_dir}")
        return save_dir


def create_cognitive_perception_module(config: Optional[Dict[str, Any]] = None) -> CognitivePerceptionModule:
    """
    工厂函数：创建认知感知模块实例
    
    Args:
        config: 可选的噪声配置
        
    Returns:
        CognitivePerceptionModule实例
    """
    return CognitivePerceptionModule(config)
