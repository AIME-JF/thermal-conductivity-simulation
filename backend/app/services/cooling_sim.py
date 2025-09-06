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
            "T_ambient": 25.0,  # °C (默认值，会在仿真时更新)
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
    
    def _cooling_law_python_style(self, t: float, T: float, h: float, A: float, m: float, T_ambient: float) -> float:
        """与Python脚本相同的冷却定律函数"""
        c = self.copper_params['specific_heat']  # 比热容
        return -h * A / (m * c) * (T - T_ambient)
    
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
    
    async def simulate(self, duration: int = 3600, noise: float = 0.1, initial_temp: float = 100.0, ambient_temp: float = 25.0) -> Dict[str, List[float]]:
        """铜块冷却仿真 - 与Python脚本保持一致的算法"""
        # 更新环境温度
        self.env_params["T_ambient"] = ambient_temp
        
        # 使用与Python脚本相同的参数
        params = {
            'initial_temp': initial_temp,
            'ambient_temp': ambient_temp,
            'volume': 1e-5,  # 样品体积 (m³)
            'cooling_time': duration
        }
        
        # 计算物理量
        m = self.copper_params['density'] * params['volume']  # 质量 (kg)
        A = 6 * (params['volume']**(1/3))**2  # 表面积 (m²)
        
        # 数值求解（与Python脚本相同的时间步长）
        time = np.linspace(0, params['cooling_time'], 1000)
        T1 = np.zeros_like(time)
        T1[0] = params['initial_temp']
        h = 12  # 优化后的对流系数 (W/m²·K)
        
        # 改进的欧拉法（与Python脚本相同的算法）
        for i in range(1, len(time)):
            dt = time[i] - time[i-1]
            k1 = self._cooling_law_python_style(time[i-1], T1[i-1], h, A, m, ambient_temp)
            k2 = self._cooling_law_python_style(time[i-1] + dt/2, T1[i-1] + k1*dt/2, h, A, m, ambient_temp)
            T1[i] = T1[i-1] + k2 * dt
        
        # 添加噪声
        if noise > 0:
            T1 += np.random.normal(0, noise, T1.shape)
        
        # 数据平滑（与Python脚本相同）
        if len(T1) > 21:
            T1_smooth = savgol_filter(T1, 21, 3)
        else:
            T1_smooth = T1
        
        # 计算温度变化率
        dt_array = np.gradient(time)
        dTdt = np.gradient(T1_smooth) / dt_array
        
        # 计算内部温度T2（与Python脚本相同的方法）
        thickness = 5e-3  # 样品厚度5mm
        diameter = 0.1    # 接触面直径10cm
        contact_area = np.pi * (diameter/2)**2
        
        # 热流计算
        mass = self.copper_params['density'] * 1e-5  # 1cm³样品
        Q = -mass * self.copper_params['specific_heat'] * dTdt
        
        # T2温度计算
        T2 = T1_smooth - (Q * thickness) / (self.copper_params['thermal_conductivity'] * contact_area)
        
        # 计算ΔT/Δt比值
        delta_ratios = self._calculate_delta_ratios(T1_smooth, T2, time)
        
        return {
            "time": time.tolist(),
            "T1": T1_smooth.tolist(),
            "T2": T2.tolist(),
            "dTdt": dTdt.tolist(),
            "deltaRatios": delta_ratios.tolist()
        }
    
    async def analyze_delta_t_ratio(self, T1: float, T2: float) -> str:
        """稳态法ΔT Δt分析 - 基于用户提供的Python脚本逻辑"""
        try:
            # 生成冷却曲线数据
            cooling_data = await self.simulate(duration=1800, noise=0.2)
            
            # 转换为numpy数组便于计算
            time_array = np.array(cooling_data["time"])
            T1_array = np.array(cooling_data["T1"])
            T2_array = np.array(cooling_data["T2"])
            
            # 验证输入值范围 - 使用统一的温度范围
            if not (38.2 <= T1 <= 100.0):
                return f"错误: T1值{T1}超出数据范围(38.2-100.0°C)"
            if not (38.2 <= T2 <= 100.0):
                return f"错误: T2值{T2}超出数据范围(38.2-100.0°C)"
            
            # 反向插值（因温度随时间递减）
            t1 = np.interp(T1, T1_array[::-1], time_array[::-1])
            t2 = np.interp(T2, T2_array[::-1], time_array[::-1])
            
            delta_t = abs(t2 - t1)
            delta_T = T1 - T2
            
            if delta_t < 1e-6:  # 避免除零
                return "错误: 时间差过小，可能导致计算不稳定"
            
            ratio = delta_T / delta_t
            
            # 格式化分析结果
            result = f"""稳态法ΔT Δt分析结果：

输入参数：
- T1 = {T1}°C
- T2 = {T2}°C

分析结果：
- T1={T1}°C 发生在 {t1:.2f}s
- T2={T2}°C 发生在 {t2:.2f}s
- 时间差 Δt = {delta_t:.2f}s
- 温度差 ΔT = {delta_T:.2f}°C
- 特征比值 ΔT/Δt = {ratio:.4f} °C/s

实验参数摘要：
- 温度采样点数量: {len(time_array)}
- 温度范围 T1: {T1_array.min():.1f}°C - {T1_array.max():.1f}°C
- 温度范围 T2: {T2_array.min():.1f}°C - {T2_array.max():.1f}°C
- 最大时间差: {time_array[-1]:.1f}s

物理意义：
该比值反映了铜材料在指定温度条件下的热传导特性，
可用于导热系数的进一步计算和分析。"""
            
            return result
            
        except Exception as e:
            return f"分析过程发生错误: {str(e)}"

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