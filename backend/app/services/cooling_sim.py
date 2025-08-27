import numpy as np
from typing import Dict, List, Any
from scipy.signal import savgol_filter

class CoolingSimulator:
    def __init__(self):
        # 铜的物理参数
        self.copper_params = {
            "density": 8960,  # kg/m³
            "specific_heat": 385,  # J/(kg·K)
            "thermal_conductivity": 401,  # W/(m·K)
            "emissivity": 0.59,
            "surface_area": 0.01,  # m²
            "volume": 0.001,  # m³
            "mass": 8.96,  # kg
        }
        
        # 环境参数
        self.env_params = {
            "T_ambient": 25.0,  # °C
            "h_convection": 10.0,  # W/(m²·K)
            "stefan_boltzmann": 5.67e-8,  # W/(m²·K⁴)
        }
    
    def _cooling_rate(self, T: float) -> float:
        """计算冷却速率 dT/dt"""
        T_K = T + 273.15  # 转换为开尔文
        T_amb_K = self.env_params["T_ambient"] + 273.15
        
        # 对流冷却
        Q_conv = (self.env_params["h_convection"] * 
                 self.copper_params["surface_area"] * 
                 (T - self.env_params["T_ambient"]))
        
        # 辐射冷却
        Q_rad = (self.copper_params["emissivity"] * 
                self.env_params["stefan_boltzmann"] * 
                self.copper_params["surface_area"] * 
                (T_K**4 - T_amb_K**4))
        
        # 总热损失
        Q_total = Q_conv + Q_rad
        
        # 温度变化率
        dT_dt = -Q_total / (self.copper_params["mass"] * 
                           self.copper_params["specific_heat"])
        
        return dT_dt
    
    def _improved_euler_step(self, T: float, dt: float) -> float:
        """改进欧拉法求解"""
        k1 = self._cooling_rate(T)
        T_pred = T + k1 * dt
        k2 = self._cooling_rate(T_pred)
        T_new = T + 0.5 * (k1 + k2) * dt
        return T_new
    
    def _calculate_delta_ratios(self, T1: np.ndarray, T2: np.ndarray, 
                               time: np.ndarray) -> np.ndarray:
        """计算ΔT/Δt比值"""
        delta_T = T1 - T2
        
        # 使用中心差分计算dT/dt
        dT1_dt = np.gradient(T1, time)
        dT2_dt = np.gradient(T2, time)
        
        # 计算Δ(T1-T2)/Δt
        d_delta_T_dt = dT1_dt - dT2_dt
        
        # 避免除零
        delta_ratios = np.where(np.abs(d_delta_T_dt) > 1e-10, 
                               delta_T / d_delta_T_dt, 0)
        
        return delta_ratios
    
    async def simulate(self, duration: int = 3600, noise: float = 0.1) -> Dict[str, List[float]]:
        """铜块冷却仿真"""
        dt = 1.0  # 时间步长（秒）
        steps = int(duration / dt)
        
        # 初始化数组
        time = np.linspace(0, duration, steps)
        T1 = np.zeros(steps)  # 热端温度
        T2 = np.zeros(steps)  # 冷端温度
        
        # 初始条件
        T1[0] = 100.0  # °C
        T2[0] = 95.0   # °C
        
        # 数值求解
        for i in range(1, steps):
            T1[i] = self._improved_euler_step(T1[i-1], dt)
            T2[i] = self._improved_euler_step(T2[i-1], dt)
            
            # 确保T2略低于T1
            if T2[i] >= T1[i]:
                T2[i] = T1[i] - 1.0
        
        # 添加噪声
        if noise > 0:
            T1 += np.random.normal(0, noise, steps)
            T2 += np.random.normal(0, noise, steps)
        
        # 平滑处理
        if len(T1) > 51:  # Savgol滤波器需要足够的点数
            T1_smooth = savgol_filter(T1, 51, 3)
            T2_smooth = savgol_filter(T2, 51, 3)
        else:
            T1_smooth = T1
            T2_smooth = T2
        
        # 计算温度变化率
        dT1_dt = np.gradient(T1_smooth, time)
        dT2_dt = np.gradient(T2_smooth, time)
        
        # 计算ΔT/Δt比值
        delta_ratios = self._calculate_delta_ratios(T1_smooth, T2_smooth, time)
        
        return {
            "time": time.tolist(),
            "T1": T1_smooth.tolist(),
            "T2": T2_smooth.tolist(),
            "dTdt": dT1_dt.tolist(),
            "deltaRatios": delta_ratios.tolist()
        }

class CopperThermalAnalyzer:
    """铜材料热分析器"""
    
    def __init__(self, cooling_data: Dict[str, List[float]]):
        self.time = np.array(cooling_data["time"])
        self.T1 = np.array(cooling_data["T1"])
        self.T2 = np.array(cooling_data["T2"])
        self.dTdt = np.array(cooling_data["dTdt"])
    
    def calculate_temperature_gradient(self, target_temp: float) -> Dict[str, float]:
        """计算指定温度下的温度梯度"""
        # 找到最接近目标温度的索引
        idx = np.argmin(np.abs(self.T1 - target_temp))
        
        return {
            "temperature": float(self.T1[idx]),
            "time": float(self.time[idx]),
            "gradient": float(self.dTdt[idx]),
            "T1": float(self.T1[idx]),
            "T2": float(self.T2[idx])
        }
    
    def get_cooling_summary(self) -> Dict[str, Any]:
        """获取冷却过程摘要"""
        return {
            "initial_temp": float(self.T1[0]),
            "final_temp": float(self.T1[-1]),
            "total_time": float(self.time[-1]),
            "max_cooling_rate": float(np.min(self.dTdt)),
            "avg_cooling_rate": float(np.mean(self.dTdt)),
            "temperature_drop": float(self.T1[0] - self.T1[-1])
        }