from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class MaterialType(str, Enum):
    """材料类型枚举"""
    GLASS = "glass"
    RUBBER = "rubber"
    COPPER = "copper"
    ALUMINUM = "aluminum"
    STEEL = "steel"

class ExperimentMethod(str, Enum):
    """实验方法枚举"""
    STEADY_STATE = "steady"
    QUASI_STEADY_STATE = "quasi"

# 稳态法相关模型
class SteadyStateRequest(BaseModel):
    """稳态法预测请求"""
    T1: float = Field(..., description="热端表面温度 (°C)", ge=38.2, le=100.0)
    T2: float = Field(..., description="冷端表面温度 (°C)", ge=38.2, le=100.0)
    selectedModel: Optional[str] = Field(default="default", description="选择的模型 (default或模型文件名)")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="可选参数")
    
    class Config:
        json_schema_extra = {
            "example": {
                "T1": 75.0,
                "T2": 52.0,
                "options": {
                    "use_auto_cooling": False,
                    "constantsOverride": {
                        "thickness": 0.01,
                        "area": 0.01,
                        "density": 1200,
                        "specific_heat": 1400
                    }
                }
            }
        }

class SteadyStateResponse(BaseModel):
    """稳态法预测响应"""
    lambda_predicted: float = Field(..., description="预测的导热系数 W/(m·K)")
    T2_corrected: float = Field(..., description="修正后的T2温度 (°C)")
    correction_params: Dict[str, float] = Field(..., description="修正参数 {a, b}")
    intermediate_values: Dict[str, float] = Field(..., description="中间计算值")
    confidence: Optional[float] = Field(None, description="预测置信度")
    
    class Config:
        json_schema_extra = {
            "example": {
                "lambda_predicted": 0.18,
                "T2_corrected": 68.5,
                "correction_params": {"a": 1.02, "b": -1.5},
                "intermediate_values": {
                    "C": 84000.0,
                    "delta_T": 6.5,
                    "dT_dt": 0.05
                },
                "confidence": 0.95
            }
        }

# 冷却仿真相关模型
class CoolingSimulationRequest(BaseModel):
    """冷却仿真请求"""
    duration: Optional[float] = Field(default=300, description="仿真时长 (s)", ge=60, le=3600)
    noise: Optional[float] = Field(default=0.1, description="噪声水平", ge=0, le=1)
    initial_temp: Optional[float] = Field(default=100, description="初始温度 (°C)", ge=50, le=200)
    
class CoolingSimulationResponse(BaseModel):
    """冷却仿真响应"""
    time: List[float] = Field(..., description="时间序列 (s)")
    T1: List[float] = Field(..., description="T1温度序列 (°C)")
    T2: List[float] = Field(..., description="T2温度序列 (°C)")
    dTdt: List[float] = Field(..., description="温度变化率序列 (K/s)")
    deltaRatios: List[float] = Field(..., description="温差时间比序列")

# 准稳态法相关模型
class QuasiSteadyStateRequest(BaseModel):
    """准稳态法预测请求"""
    V_t: float = Field(..., description="电势 V_t (mV)", ge=0.001, le=1.0)
    delta_V: float = Field(..., description="电势变化率 ΔV (mV/min)", ge=0.001, le=1.0)
    material: MaterialType = Field(..., description="材料类型")
    selectedModel: Optional[str] = Field(default="default", description="选择的模型 (default或模型文件名)")
    constantsOverride: Optional[Dict[str, float]] = Field(default_factory=dict, description="常数覆盖")
    
    class Config:
        json_schema_extra = {
            "example": {
                "V_t": 0.014,
                "delta_V": 0.022,
                "material": "glass",
                "constantsOverride": {
                    "U": 12.0,
                    "R": 50.0,
                    "S": 0.01
                }
            }
        }

class QuasiSteadyStateResponse(BaseModel):
    """准稳态法预测响应"""
    lambda_predicted: float = Field(..., description="预测的导热系数 W/(m·K)")
    lambda_theory: float = Field(..., description="理论计算的导热系数 W/(m·K)")
    lambda_error: float = Field(..., description="导热系数预测误差")
    intermediate_values: Dict[str, float] = Field(..., description="中间计算值")
    model_version: str = Field(..., description="模型版本")
    
    class Config:
        json_schema_extra = {
            "example": {
                "lambda_predicted": 0.15,
                "lambda_theory": 0.2000,
                "lambda_error": 0.05,
                "intermediate_values": {
                    "delta_T": 0.028,
                    "dT_dt": 0.00037,
                    "q_c": 0.0288
                },
                "model_version": "v1.0"
            }
        }

# 批量处理相关模型
class BatchPredictionItem(BaseModel):
    """批量预测单项"""
    method: ExperimentMethod = Field(..., description="实验方法")
    data: Dict[str, Any] = Field(..., description="输入数据")
    
class BatchPredictionRequest(BaseModel):
    """批量预测请求"""
    items: List[BatchPredictionItem] = Field(..., description="预测项目列表")
    
class BatchPredictionResult(BaseModel):
    """批量预测结果项"""
    index: int = Field(..., description="项目索引")
    success: bool = Field(..., description="是否成功")
    result: Optional[Dict[str, Any]] = Field(None, description="预测结果")
    error: Optional[str] = Field(None, description="错误信息")
    
class BatchPredictionResponse(BaseModel):
    """批量预测响应"""
    total: int = Field(..., description="总数")
    success_count: int = Field(..., description="成功数量")
    error_count: int = Field(..., description="错误数量")
    results: List[BatchPredictionResult] = Field(..., description="结果列表")

# 模型管理相关
class ModelInfo(BaseModel):
    """模型信息"""
    name: str = Field(..., description="模型名称")
    path: str = Field(..., description="模型路径")
    size: int = Field(..., description="文件大小 (bytes)")
    created_at: str = Field(..., description="创建时间")
    version: str = Field(..., description="版本号")
    
class ModelsResponse(BaseModel):
    """模型列表响应"""
    steady_state_models: List[ModelInfo] = Field(..., description="稳态法模型列表")
    quasi_steady_models: List[ModelInfo] = Field(..., description="准稳态法模型列表")
    scalers: List[ModelInfo] = Field(..., description="标准化器列表")
    
class TrainingRequest(BaseModel):
    """训练请求"""
    method: ExperimentMethod = Field(..., description="实验方法")
    epochs: Optional[int] = Field(default=100, description="训练轮数", ge=10, le=1000)
    batch_size: Optional[int] = Field(default=32, description="批次大小", ge=8, le=128)
    learning_rate: Optional[float] = Field(default=0.001, description="学习率", ge=0.0001, le=0.1)
    
class TrainingResponse(BaseModel):
    """训练响应"""
    task_id: str = Field(..., description="训练任务ID")
    status: str = Field(..., description="训练状态")
    message: str = Field(..., description="状态消息")
    log_path: Optional[str] = Field(None, description="日志文件路径")

# 通用响应模型
class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    
class SuccessResponse(BaseModel):
    """成功响应"""
    message: str = Field(..., description="成功消息")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")