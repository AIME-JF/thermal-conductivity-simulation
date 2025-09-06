import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.gridspec import GridSpec
import sys

# ========================
# 铜材料物理参数
# ========================
COPPER = {
    'density': 8960,            # 密度 (kg/m³)
    'specific_heat': 385,        # 比热容 (J/kg·K)
    'thermal_conductivity': 401, # 导热系数 (W/m·K)
    'emissivity': 0.07           # 发射率
}

# ========================
# 冷却定律函数（优化版）
# ========================
def cooling_law(t, T, h, A, m, c, T_env):
    """组合冷却定律：牛顿冷却 + 辐射散热"""
    convection = h * A * (T - T_env)
    radiation = 5.67e-8 * COPPER['emissivity'] * A * (T**4 - T_env**4)
    dTdt = -(convection + radiation) / (m * c)
    return dTdt

# ========================
# 铜冷却曲线生成器
# ========================
def generate_copper_cooling(duration=1800, noise_level=0.2):
    """生成铜的冷却曲线"""
    params = {
        'initial_temp': 100.0,    # 初始温度 (°C)
        'ambient_temp': 25.0,     # 环境温度 (°C)
        'volume': 1e-5,          # 样品体积 (m³)
        'cooling_time': duration  # 总时间 (s)
    }
    
    # 计算物理量
    m = COPPER['density'] * params['volume']  # 质量 (kg)
    A = 6 * (params['volume']**(1/3))**2      # 表面积 (m²)
    
    # 数值求解（改进时间步长）
    t = np.linspace(0, params['cooling_time'], 1000)
    T = np.zeros_like(t)
    T[0] = params['initial_temp']
    h = 12  # 优化后的对流系数 (W/m²·K)
    
    # 改进的欧拉法
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        k1 = cooling_law(t[i-1], T[i-1], h, A, m,
                        COPPER['specific_heat'],
                        params['ambient_temp'])
        k2 = cooling_law(t[i-1] + dt/2, T[i-1] + k1*dt/2, 
                        h, A, m, COPPER['specific_heat'],
                        params['ambient_temp'])
        T[i] = T[i-1] + k2 * dt
        
    # 添加噪声
    T += np.random.normal(0, noise_level, T.shape)
    return t, T

# ========================
# 铜热分析系统（升级版）
# ========================
class CopperThermalAnalyzer:
    def __init__(self):
        # 生成并加载数据
        self.time, self.T1 = generate_copper_cooling()
        self.T1 = savgol_filter(self.T1, 21, 3)  # 数据平滑
        
        # 计算相关参数
        self.calculate_derivatives()
        self.calculate_t2_profile()
        self.calculate_delta_ratios()
    
    def calculate_derivatives(self):
        """计算温度变化率"""
        self.dt = np.gradient(self.time)
        self.dTdt = np.gradient(self.T1) / self.dt
    
    def calculate_t2_profile(self):
        """计算内部温度T2"""
        # 热传导参数
        thickness = 5e-3  # 样品厚度5mm
        diameter = 0.1    # 接触面直径10cm
        contact_area = np.pi * (diameter/2)**2
        
        # 热流计算
        mass = COPPER['density'] * 1e-5  # 1cm³样品
        self.Q = -mass * COPPER['specific_heat'] * self.dTdt
        
        # T2温度计算
        self.T2 = self.T1 - (self.Q * thickness) / (COPPER['thermal_conductivity'] * contact_area)
    
    def calculate_delta_ratios(self):
        """计算温度差时间比"""
        self.delta_T = self.T1 - self.T2
        self.time_intervals = self.dt
        self.delta_ratios = self.delta_T[1:] / self.time_intervals[1:]
    
    def calculate_timedelta_ratio(self, T1_val: float, T2_val: float):
        """
        根据输入的T1和T2值计算时间差及比值
        返回: {
            't1': T1对应时间, 
            't2': T2对应时间,
            'delta_t': 时间差,
            'delta_T': 温度差,
            'ratio': 比值
        }
        """
        # 验证输入值范围 - 使用统一的温度范围
        if not (38.2 <= T1_val <= 100.0):
            raise ValueError(f"T1值{T1_val}超出数据范围(38.2-100.0°C)")
        if not (38.2 <= T2_val <= 100.0):
            raise ValueError(f"T2值{T2_val}超出数据范围(38.2-100.0°C)")

        # 反向插值（因温度随时间递减）
        t1 = np.interp(T1_val, self.T1[::-1], self.time[::-1])
        t2 = np.interp(T2_val, self.T2[::-1], self.time[::-1])
        
        delta_t = abs(t2 - t1)
        delta_T = T1_val - T2_val
        
        if delta_t < 1e-6:  # 避免除零
            return {'error': '时间差过小，可能导致计算不稳定'}
            
        return {
            't1': t1,
            't2': t2,
            'delta_t': delta_t,
            'delta_T': delta_T,
            'ratio': delta_T / delta_t
        }
    
    def visualize_analysis(self):
        """综合可视化"""
        plt.figure(figsize=(15, 8))
        
        # 温度曲线
        ax1 = plt.subplot(221)
        ax1.plot(self.time, self.T1, 'b', label='表面温度 T1')
        ax1.plot(self.time, self.T2, 'r', label='内部温度 T2')
        ax1.set_title('铜材料冷却曲线')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('温度 (°C)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 温差时间比
        ax2 = plt.subplot(222)
        ax2.plot(self.time[1:], self.delta_ratios, 'g')
        ax2.set_title('温差时间比 (ΔT/Δt)')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('Δ(T1-T2)/Δt (°C/s)')
        ax2.grid(True, alpha=0.3)
        
        # T1-T2关系
        ax3 = plt.subplot(223)
        sc = ax3.scatter(self.T1, self.T2, c=self.time, cmap='viridis')
        plt.colorbar(sc, label='时间 (s)')
        ax3.set_title('T1-T2相位关系')
        ax3.set_xlabel('T1 (°C)')
        ax3.set_ylabel('T2 (°C)')
        
        # 温差分布
        ax4 = plt.subplot(224)
        ax4.plot(self.time, self.delta_T, 'm')
        ax4.set_title('实时温度差 ΔT')
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('T1-T2 (°C)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ========================
# 增强版主程序
# ========================
if __name__ == "__main__":
    analyzer = CopperThermalAnalyzer()
    
    # 打印关键数据示例
    print("实验参数摘要：")
    print(f"温度采样点数量: {len(analyzer.time)}")
    print(f"温度范围 T1: {analyzer.T1.min():.1f}°C - {analyzer.T1.max():.1f}°C")
    print(f"温度范围 T2: {analyzer.T2.min():.1f}°C - {analyzer.T2.max():.1f}°C")
    print(f"最大时间差: {analyzer.time[-1]:.1f}s\n")
    
    # 命令行参数支持
    if len(sys.argv) == 3:
        try:
            T1_input = float(sys.argv[1])
            T2_input = float(sys.argv[2])
            
            result = analyzer.calculate_timedelta_ratio(T1_input, T2_input)
            
            if 'error' in result:
                print(f"计算警告: {result['error']}")
            else:
                print("\n分析结果：")
                print(f"T1={T1_input}°C 发生在 {result['t1']:.2f}s")
                print(f"T2={T2_input}°C 发生在 {result['t2']:.2f}s")
                print(f"时间差 Δt = {result['delta_t']:.2f}s")
                print(f"温度差 ΔT = {result['delta_T']:.2f}°C")
                print(f"特征比值 ΔT/Δt = {result['ratio']:.4f} °C/s\n")
            sys.exit(0)  # 直接退出，不进入交互模式
        except ValueError as e:
            print(f"命令行参数错误: {e}")
            sys.exit(1)
    
    # 交互式分析
    while True:
        try:
            T1_input = float(input("请输入温度T1值（°C）: "))
            T2_input = float(input("请输入温度T2值（°C）: "))
            
            result = analyzer.calculate_timedelta_ratio(T1_input, T2_input)
            
            if 'error' in result:
                print(f"计算警告: {result['error']}")
            else:
                print("\n分析结果：")
                print(f"T1={T1_input}°C 发生在 {result['t1']:.2f}s")
                print(f"T2={T2_input}°C 发生在 {result['t2']:.2f}s")
                print(f"时间差 Δt = {result['delta_t']:.2f}s")
                print(f"温度差 ΔT = {result['delta_T']:.2f}°C")
                print(f"特征比值 ΔT/Δt = {result['ratio']:.4f} °C/s\n")
                
        except ValueError as e:
            if "could not convert string to float" in str(e):
                cmd = input("检测到非数字输入，是否退出？(y/n) ").lower()
                if cmd == 'y':
                    break
            else:
                print(f"输入错误: {e}\n")
        except Exception as e:
            print(f"发生错误: {str(e)}\n")
    
    # 可视化分析
    analyzer.visualize_analysis()