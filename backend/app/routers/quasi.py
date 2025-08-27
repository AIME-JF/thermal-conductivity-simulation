from fastapi import APIRouter, HTTPException, Request
from ..services.quasi_service import QuasiSteadyService
from ..models.schemas import QuasiSteadyStateRequest, QuasiSteadyStateResponse, ErrorResponse
from ..utils.logger import get_logger, get_access_logger
import traceback
from typing import Union

router = APIRouter()
quasi_service = QuasiSteadyService()
logger = get_logger("thermal_sim.quasi")
access_logger = get_access_logger()

@router.post("/predict", response_model=Union[QuasiSteadyStateResponse, ErrorResponse])
async def predict_quasi_steady(request: QuasiSteadyStateRequest, http_request: Request):
    """准稳态法导热系数和比热容预测"""
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    try:
        logger.info(f"准稳态法预测请求 - IP: {client_ip}, 参数: V_t={request.V_t}, delta_V={request.delta_V}, material={request.material.value}")
        
        # 参数验证
        if request.V_t <= 0 or request.V_t > 1:
            raise ValueError("电压V_t必须在0-1 mV范围内")
        if request.delta_V <= 0 or request.delta_V > 1:
            raise ValueError("电压变化率delta_V必须在0-1 mV/min范围内")
        
        result = await quasi_service.predict(
            V_t=request.V_t,
            delta_V=request.delta_V,
            material=request.material.value,  # 使用枚举值
            selected_model=request.selectedModel,
            constants_override=request.constantsOverride or {}
        )
        
        logger.info(f"准稳态法预测成功 - IP: {client_ip}, λ={result['lambda_predicted']:.4f}, 理论值={result['lambda_theory']:.4f}")
        return result
        
    except ValueError as e:
        error_msg = f"参数验证失败: {str(e)}"
        logger.warning(f"准稳态法预测参数错误 - IP: {client_ip}, 错误: {error_msg}")
        raise HTTPException(status_code=422, detail=error_msg)
        
    except FileNotFoundError as e:
        error_msg = "模型文件未找到，请联系管理员"
        logger.error(f"准稳态法预测模型文件错误 - IP: {client_ip}, 错误: {str(e)}")
        raise HTTPException(status_code=503, detail=error_msg)
        
    except Exception as e:
        error_msg = f"预测过程发生错误: {str(e)}"
        logger.error(f"准稳态法预测未知错误 - IP: {client_ip}, 错误: {error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)