from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Literal, Optional
from ..services.batch_service import BatchService

router = APIRouter()
batch_service = BatchService()

class BatchPredictRequest(BaseModel):
    method: Literal["steady", "quasi", "cooling"]
    data: List[Dict[str, Any]]
    material: Optional[str] = None

class BatchResultItem(BaseModel):
    row_index: int
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BatchPredictResponse(BaseModel):
    total_rows: int
    success_count: int
    error_count: int
    results: List[BatchResultItem]
    processing_time: float

@router.post("/predict", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    """批量预测"""
    try:
        result = await batch_service.batch_predict(
            method=request.method,
            data=request.data,
            material=request.material
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/upload-csv", response_model=BatchPredictResponse)
async def upload_csv_predict(
    file: UploadFile = File(...),
    method: str = "steady",
    material: Optional[str] = None
):
    """CSV文件上传批量预测"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="只支持CSV文件")
        
        result = await batch_service.process_csv_file(
            file=file,
            method=method,
            material=material
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))