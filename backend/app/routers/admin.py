from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ..services.admin_service import AdminService
from ..utils.auth import verify_admin_token

router = APIRouter()
admin_service = AdminService()

class ModelInfo(BaseModel):
    name: str
    version: str
    timestamp: str
    file_path: str
    size_mb: float

class TrainingRequest(BaseModel):
    model_type: str
    epochs: Optional[int] = 100
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 0.001
    validation_split: Optional[float] = 0.2

class TrainingResponse(BaseModel):
    task_id: str
    status: str
    message: str
    log_file: Optional[str] = None

@router.get("/models", response_model=List[ModelInfo])
async def list_models(x_admin_token: str = Header(...)):
    """获取模型列表"""
    verify_admin_token(x_admin_token)
    try:
        models = await admin_service.list_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    x_admin_token: str = Header(...)
):
    """开始模型训练"""
    verify_admin_token(x_admin_token)
    try:
        result = await admin_service.start_training(
            model_type=request.model_type,
            config={
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "validation_split": request.validation_split
            }
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training-status/{task_id}")
async def get_training_status(
    task_id: str,
    x_admin_token: str = Header(...)
):
    """获取训练状态"""
    verify_admin_token(x_admin_token)
    try:
        status = await admin_service.get_training_status(task_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training-history")
async def get_training_history(
    x_admin_token: str = Header(...)
):
    """获取训练历史"""
    verify_admin_token(x_admin_token)
    try:
        history = await admin_service.get_training_history()
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))