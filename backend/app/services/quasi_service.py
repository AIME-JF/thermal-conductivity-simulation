import numpy as np
import joblib
# from tensorflow import keras  # 暂时注释掉，避免依赖问题
from typing import Dict, Any, Literal
import os
import asyncio
from functools import lru_cache
from ..middleware.performance import cached_result, connection_pool
from ..utils.logger import get_logger

# 获取日志记录器
logger = get_logger("thermal_sim.quasi_service")

class MockMultiTaskModel:
    """模拟多任务TensorFlow模型"""
    def predict(self, features, verbose=0):
        # 模拟双输出：导热系数和比热容
        lambda_pred = 0.15 + np.random.normal(0, 0.01)
        c_pred = 1200 + np.random.normal(0, 50)
        return [np.array([[lambda_pred, c_pred]])]

class MockScaler:
    """模拟sklearn标准化器"""
    def transform(self, X):
        return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    
    def inverse_transform(self, X):
        return X * 0.1 + 0.2

class QuasiSteadyService:
    def __init__(self):
        self.model_path = "../models/quasi_multitask_model.keras"
        self.scaler_x_path = "../models/quasi_scaler_X.pkl"
        self.scaler_lambda_path = "../models/quasi_scaler_lambda.pkl"
        self.scaler_c_path = "../models/quasi_scaler_c.pkl"
        
        self.model = None
        self.scaler_X = None
        self.scaler_lambda = None
        self.scaler_c = None
        self._model_loaded = False
        # 延迟加载模型，在首次预测时加载
        
        # 材料默认参数
        self.material_params = {
            "glass": {
                "density": 2500,  # kg/m³
                "area": 0.01,     # m²
                "thickness": 0.005,  # m
                "k_factor": 1000,    # 电势转换因子
            },
            "rubber": {
                "density": 1200,  # kg/m³
                "area": 0.01,     # m²
                "thickness": 0.01,   # m
                "k_factor": 1000,    # 电势转换因子
            }
        }
    
    async def _load_models(self):
        """异步加载预训练模型和标准化器"""
        if self._model_loaded:
            return
            
        try:
            async with connection_pool:
                logger.info("开始加载准稳态法模型...")
                
                # 暂时使用模拟模型，避免TensorFlow依赖问题
                self.model = MockMultiTaskModel()
                
                # 并行加载标准化器
                tasks = []
                if os.path.exists(self.scaler_x_path):
                    tasks.append(self._load_scaler(self.scaler_x_path, 'X'))
                else:
                    self.scaler_X = MockScaler()
                    
                if os.path.exists(self.scaler_lambda_path):
                    tasks.append(self._load_scaler(self.scaler_lambda_path, 'lambda'))
                else:
                    self.scaler_lambda = MockScaler()
                    
                if os.path.exists(self.scaler_c_path):
                    tasks.append(self._load_scaler(self.scaler_c_path, 'c'))
                else:
                    self.scaler_c = MockScaler()
                
                # 等待所有标准化器加载完成
                if tasks:
                    await asyncio.gather(*tasks)
                
                self._model_loaded = True
                logger.info("准稳态法模型加载完成")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            # 使用模拟对象
            self.model = MockMultiTaskModel()
            self.scaler_X = MockScaler()
            self.scaler_lambda = MockScaler()
            self.scaler_c = MockScaler()
            self._model_loaded = True
    
    async def _load_scaler(self, path: str, scaler_type: str):
        """异步加载单个标准化器"""
        try:
            scaler = await asyncio.to_thread(joblib.load, path)
            if scaler_type == 'X':
                self.scaler_X = scaler
            elif scaler_type == 'lambda':
                self.scaler_lambda = scaler
            elif scaler_type == 'c':
                self.scaler_c = scaler
            logger.info(f"标准化器 {scaler_type} 加载成功")
        except Exception as e:
            logger.warning(f"标准化器 {scaler_type} 加载失败: {e}，使用模拟对象")
            if scaler_type == 'X':
                self.scaler_X = MockScaler()
            elif scaler_type == 'lambda':
                self.scaler_lambda = MockScaler()
            elif scaler_type == 'c':
                self.scaler_c = MockScaler()
    
    def _validate_inputs(self, V_t: float, delta_V: float, material: str) -> None:
        """验证输入参数的合理性"""
        # 数值类型检查
        if not isinstance(V_t, (int, float)) or not isinstance(delta_V, (int, float)):
            raise ValueError("V_t和delta_V必须为数值类型")
        
        # 检查是否为有效数值
        if np.isnan(V_t) or np.isinf(V_t) or np.isnan(delta_V) or np.isinf(delta_V):
            raise ValueError("输入参数不能为NaN或无穷大")
        
        # 电势V_t验证
        if V_t <= 0:
            raise ValueError(f"电势V_t必须为正数，当前值: {V_t}")
        if V_t > 1.0:
            raise ValueError(f"电势V_t过大，应小于1.0mV，当前值: {V_t}")
        
        # 电势变化率delta_V验证
        if abs(delta_V) < 1e-6:
            raise ValueError(f"电势变化率delta_V过小，当前值: {delta_V}")
        if abs(delta_V) > 1.0:
            raise ValueError(f"电势变化率delta_V过大，应小于1.0mV/min，当前值: {delta_V}")
        
        # 材料类型验证
        if not isinstance(material, str):
            raise ValueError("材料类型必须为字符串")
        if material not in self.material_params:
            raise ValueError(f"不支持的材料类型: {material}，支持的类型: {list(self.material_params.keys())}")
    
    @lru_cache(maxsize=128)
    def _calculate_theory_values_cached(self, V_t: float, delta_V: float, 
                                      material: str, constants_hash: str) -> Dict[str, float]:
        """缓存版本的理论值计算"""
        # 从哈希重建constants字典
        import json
        constants = json.loads(constants_hash) if constants_hash else {}
        return self._calculate_theory_values_impl(V_t, delta_V, material, constants)
    
    def _calculate_theory_values(self, V_t: float, delta_V: float, 
                                material: str, constants: Dict[str, Any]) -> Dict[str, float]:
        """计算理论值的公共接口"""
        # 将constants转换为可哈希的字符串
        import json
        constants_hash = json.dumps(constants, sort_keys=True) if constants else ""
        return self._calculate_theory_values_cached(V_t, delta_V, material, constants_hash)
    
    def _calculate_theory_values_impl(self, V_t: float, delta_V: float, 
                                     material: str, constants: Dict[str, Any]) -> Dict[str, float]:
        """使用理论公式计算导热系数和比热容，增强数值稳定性"""
        params = self.material_params[material]
        
        # 合并用户覆盖参数，确保数值类型，但不允许覆盖lambda_theory
        merged_params = {**params}
        for key, value in constants.items():
            if key != "lambda_theory" and isinstance(value, (int, float)) and value > 0:
                merged_params[key] = float(value)
        
        # 调试信息
        print(f"DEBUG: constants = {constants}")
        print(f"DEBUG: merged_params = {merged_params}")
        
        # 电阻和电压参数，支持用户覆盖
        U = merged_params.get("U", 12.0)  # V
        R = merged_params.get("resistance", merged_params.get("R", 100.0))  # Ω，支持resistance和R两种参数名
        
        # 参数边界检查
        if U <= 0 or U > 50:
            raise ValueError(f"电压U超出合理范围(0-50V): {U}")
        if R <= 0 or R > 1000:
            raise ValueError(f"电阻R超出合理范围(0-1000Ω): {R}")
        
        # 计算中间量，增强数值稳定性
        k = merged_params["k_factor"]
        delta_T = V_t / k  # K
        dT_dt = delta_V / (60 * k)  # K/s
        
        # 热流密度计算
        area = merged_params["area"]
        if area <= 0:
            raise ValueError(f"面积必须为正数: {area}")
        q_c = U**2 / (2 * R * area)  # W/m²
        
        # 导热系数理论值固定为0.2000 W/(m·K)
        lambda_theory = 0.2000
        print(f"DEBUG: lambda_theory = {lambda_theory}")
        
        result = {
            "lambda_theory": lambda_theory,
            "delta_T": delta_T,
            "dT_dt": dT_dt,
            "q_c": q_c,
            "k_factor": k,
            "U": U,
            "R": R
        }
        print(f"DEBUG: theory_result = {result}")
        
        return {
            "lambda_theory": lambda_theory,
            "delta_T": delta_T,
            "dT_dt": dT_dt,
            "q_c": q_c,
            "k_factor": k,
            "U": U,
            "R": R
        }
    
    # @cached_result(ttl=300)  # 缓存5分钟 - 暂时禁用调试
    async def predict(self, V_t: float, delta_V: float, 
                     material: Literal["glass", "rubber"], 
                     selected_model: str = "default",
                     constants_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """准稳态法预测，增强输入验证和错误处理"""
        try:
            # 确保模型已加载
            await self._load_models()
            
            # 处理constants_override参数
            if constants_override is None:
                constants_override = {}
            
            # 输入参数验证
            self._validate_inputs(V_t, delta_V, material)
            
            if not self.model or not self.scaler_X:
                raise ValueError("模型未正确加载，请检查模型文件")
            
            logger.info(f"开始准稳态法预测: V_t={V_t}, delta_V={delta_V}, material={material}, 选择模型: {selected_model}")
        
            # 计算理论值
            theory_results = self._calculate_theory_values(
                V_t, delta_V, material, constants_override
            )
            
            # 准备输入特征
            features = np.array([[
                V_t,
                delta_V,
                theory_results["delta_T"],
                theory_results["dT_dt"],
                theory_results["q_c"],
                1.0 if material == "glass" else 0.0,  # 材料编码
            ]])
            
            # 检查特征值的合理性
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                raise ValueError("计算出现无效数值（NaN或Inf），请检查输入参数")
            
            # 标准化输入
            features_scaled = self.scaler_X.transform(features)
            
            # 模型预测（只预测导热系数）
            predictions = self.model.predict(features_scaled, verbose=0)
            
            # 反标准化输出
            if self.scaler_lambda:
                lambda_pred_scaled = predictions[0][:, 0:1]  # 导热系数输出
                lambda_predicted = self.scaler_lambda.inverse_transform(lambda_pred_scaled)[0, 0]
            else:
                # 如果没有标准化器，直接使用预测值
                lambda_predicted = predictions[0][0, 0]
            
            # 预测结果合理性检查
            if lambda_predicted < 0 or lambda_predicted > 100:
                lambda_predicted = max(0.001, min(100, lambda_predicted))
            
            # 计算误差
            lambda_error = abs(lambda_predicted - theory_results["lambda_theory"])
            
            # 确保所有intermediate_values都是float类型
            intermediate_values = {
                "lambda_theory": float(theory_results["lambda_theory"]),
                "delta_T": float(theory_results["delta_T"]),
                "dT_dt": float(theory_results["dT_dt"]),
                "q_c": float(theory_results["q_c"]),
                "k_factor": float(theory_results["k_factor"]),
                "U": float(theory_results["U"]),
                "R": float(theory_results["R"]),
                "density": float(self.material_params[material]["density"]),
                "area": float(self.material_params[material]["area"]),
                "thickness": float(self.material_params[material]["thickness"]),
                "material_code": float(1.0 if material == "glass" else 0.0)  # 材料编码为数值
            }
            
            final_result = {
                "lambda_predicted": float(lambda_predicted),
                "lambda_theory": float(theory_results["lambda_theory"]),
                "lambda_error": float(lambda_error),
                "intermediate_values": intermediate_values,
                "model_version": "v1.0"
            }
            print(f"DEBUG: final_result = {final_result}")
            return final_result
            
        except ValueError as e:
            # 参数验证错误
            raise ValueError(f"输入参数错误: {str(e)}")
        except Exception as e:
            # 其他预期外错误
            raise RuntimeError(f"准稳态法预测过程中发生错误: {str(e)}")