from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
from ..services.steady_service import SteadyStateService
from ..services.cooling_sim import CoolingSimulator
from ..models.schemas import ErrorResponse
from ..utils.logger import get_logger, get_access_logger
import traceback

router = APIRouter()
steady_service = SteadyStateService()
cooling_sim = CoolingSimulator()
logger = get_logger("thermal_sim.steady")
access_logger = get_access_logger()

class SteadyPredictRequest(BaseModel):
    T1: float
    T2: float
    selectedModel: Optional[str] = "default"
    options: Optional[Dict[str, Any]] = None

class CoolingSimRequest(BaseModel):
    duration: Optional[int] = 3600
    noise: Optional[float] = 0.1
    initial_temp: Optional[float] = 100.0
    ambient_temp: Optional[float] = 25.0

class SteadyPredictResponse(BaseModel):
    lambda_predicted: float
    T2_corrected: float
    correction_params: Dict[str, float]
    intermediate_values: Dict[str, float]
    confidence: Optional[float] = None

class CoolingSimResponse(BaseModel):
    time: List[float]
    T1: List[float]
    T2: List[float]
    dTdt: List[float]
    deltaRatios: List[float]

class SteadyStateAnalysisRequest(BaseModel):
    T1: float
    T2: float

class SteadyStateAnalysisResponse(BaseModel):
    analysis_result: str

@router.post("/predict", response_model=Union[SteadyPredictResponse, ErrorResponse])
async def predict_steady_state(request: SteadyPredictRequest, http_request: Request):
    """稳态法导热系数预测"""
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    try:
        logger.info(f"稳态法预测请求 - IP: {client_ip}, 参数: T1={request.T1}, T2={request.T2}")
        
        # 参数验证 - 根据实际物理模型调整
        if request.T1 < 38.2 or request.T1 > 100.0:
            raise ValueError("温度T1必须在38.2-100.0°C范围内")
        if request.T2 < 38.2 or request.T2 > 100.0:
            raise ValueError("温度T2必须在38.2-100.0°C范围内")
        if request.T1 < request.T2:
            raise ValueError("T1应该大于等于T2（热传导方向）")
        
        result = await steady_service.predict(
            T1=request.T1,
            T2=request.T2,
            selected_model=request.selectedModel,
            options=request.options or {}
        )
        
        logger.info(f"稳态法预测成功 - IP: {client_ip}, λ={result['lambda_predicted']:.4f}")
        return result
        
    except ValueError as e:
        error_msg = f"参数验证失败: {str(e)}"
        logger.warning(f"稳态法预测参数错误 - IP: {client_ip}, 错误: {error_msg}")
        raise HTTPException(status_code=422, detail=error_msg)
        
    except FileNotFoundError as e:
        error_msg = "模型文件未找到，请联系管理员"
        logger.error(f"稳态法预测模型文件错误 - IP: {client_ip}, 错误: {str(e)}")
        raise HTTPException(status_code=503, detail=error_msg)
        
    except Exception as e:
        error_msg = f"预测过程发生错误: {str(e)}"
        logger.error(f"稳态法预测未知错误 - IP: {client_ip}, 错误: {error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/simulate-cooling", response_model=Union[CoolingSimResponse, ErrorResponse])
async def simulate_cooling(request: CoolingSimRequest, http_request: Request):
    """铜块冷却仿真"""
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    try:
        logger.info(f"冷却仿真请求 - IP: {client_ip}, 参数: duration={request.duration}, noise={request.noise}")
        
        # 参数验证
        if request.duration and (request.duration < 60 or request.duration > 7200):
            raise ValueError("仿真时长必须在60-7200秒范围内")
        if request.noise and (request.noise < 0 or request.noise > 1):
            raise ValueError("噪声水平必须在0-1范围内")
        
        result = await cooling_sim.simulate(
            duration=request.duration,
            noise=request.noise,
            initial_temp=request.initial_temp,
            ambient_temp=request.ambient_temp
        )
        
        logger.info(f"冷却仿真成功 - IP: {client_ip}, 数据点数: {len(result['time'])}")
        return result
        
    except ValueError as e:
        error_msg = f"参数验证失败: {str(e)}"
        logger.warning(f"冷却仿真参数错误 - IP: {client_ip}, 错误: {error_msg}")
        raise HTTPException(status_code=422, detail=error_msg)
        
    except Exception as e:
        error_msg = f"仿真过程发生错误: {str(e)}"
        logger.error(f"冷却仿真未知错误 - IP: {client_ip}, 错误: {error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/steady-state-analysis", response_model=Union[SteadyStateAnalysisResponse, ErrorResponse])
async def steady_state_analysis(request: SteadyStateAnalysisRequest, http_request: Request):
    """稳态法ΔT Δt分析"""
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    try:
        logger.info(f"稳态法ΔT Δt分析请求 - IP: {client_ip}, 参数: T1={request.T1}, T2={request.T2}")
        
        # 参数验证 - 根据实际物理模型调整
        if request.T1 < 38.2 or request.T1 > 100.0:
            raise ValueError("温度T1必须在38.2-100.0°C范围内")
        if request.T2 < 38.2 or request.T2 > 100.0:
            raise ValueError("温度T2必须在38.2-100.0°C范围内")
        if request.T1 < request.T2:
            raise ValueError("T1应该大于等于T2（热传导方向）")
        
        # 调用冷却仿真分析服务
        result = await cooling_sim.analyze_delta_t_ratio(
            T1=request.T1,
            T2=request.T2
        )
        
        logger.info(f"稳态法ΔT Δt分析成功 - IP: {client_ip}")
        return {"analysis_result": result}
        
    except ValueError as e:
        error_msg = f"参数验证失败: {str(e)}"
        logger.warning(f"稳态法ΔT Δt分析参数错误 - IP: {client_ip}, 错误: {error_msg}")
        raise HTTPException(status_code=422, detail=error_msg)
        
    except Exception as e:
        error_msg = f"分析过程发生错误: {str(e)}"
        logger.error(f"稳态法ΔT Δt分析未知错误 - IP: {client_ip}, 错误: {error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
