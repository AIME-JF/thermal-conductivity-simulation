import pandas as pd
import time
from typing import List, Dict, Any, Optional
from fastapi import UploadFile
from .steady_service import SteadyStateService
from .quasi_service import QuasiSteadyService
from .cooling_sim import CoolingSimulator
import io

class BatchService:
    def __init__(self):
        self.steady_service = SteadyStateService()
        self.quasi_service = QuasiSteadyService()
        self.cooling_sim = CoolingSimulator()
    
    async def batch_predict(self, method: str, data: List[Dict[str, Any]], 
                           material: Optional[str] = None) -> Dict[str, Any]:
        """批量预测"""
        start_time = time.time()
        results = []
        success_count = 0
        error_count = 0
        
        for i, row_data in enumerate(data):
            try:
                if method == "steady":
                    result = await self.steady_service.predict(
                        T1=row_data["T1"],
                        T2=row_data["T2"],
                        options=row_data.get("options", {})
                    )
                elif method == "quasi":
                    result = await self.quasi_service.predict(
                        V_t=row_data["V_t"],
                        delta_V=row_data["delta_V"],
                        material=material or row_data.get("material", "glass"),
                        constants_override=row_data.get("constantsOverride", {})
                    )
                elif method == "cooling":
                    cooling_data = await self.cooling_sim.simulate(
                        duration=int(row_data["duration"]),
                        noise=float(row_data["noise"])
                    )
                    # 计算冷却特征参数
                    import numpy as np
                    T1_array = np.array(cooling_data["T1"])
                    time_array = np.array(cooling_data["time"])
                    
                    max_temp = float(np.max(T1_array))
                    min_temp = float(np.min(T1_array))
                    avg_cooling_rate = float((T1_array[0] - T1_array[-1]) / (time_array[-1] - time_array[0])) if len(T1_array) > 1 else 0
                    
                    # 计算半冷却时间
                    half_temp = (max_temp + min_temp) / 2
                    half_temp_idx = np.argmin(np.abs(T1_array - half_temp))
                    half_cooling_time = float(time_array[half_temp_idx])
                    
                    result = {
                        "max_temp": max_temp,
                        "min_temp": min_temp,
                        "avg_cooling_rate": avg_cooling_rate,
                        "half_cooling_time": half_cooling_time
                    }
                else:
                    raise ValueError(f"不支持的方法: {method}")
                
                results.append({
                    "row_index": i,
                    "success": True,
                    "result": result,
                    "error": None
                })
                success_count += 1
                
            except Exception as e:
                results.append({
                    "row_index": i,
                    "success": False,
                    "result": None,
                    "error": str(e)
                })
                error_count += 1
        
        processing_time = time.time() - start_time
        
        return {
            "total_rows": len(data),
            "success_count": success_count,
            "error_count": error_count,
            "results": results,
            "processing_time": processing_time
        }
    
    async def process_csv_file(self, file: UploadFile, method: str, 
                              material: Optional[str] = None) -> Dict[str, Any]:
        """处理CSV文件批量预测"""
        try:
            # 读取CSV文件
            content = await file.read()
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            
            # 验证必需的列
            if method == "steady":
                required_cols = ["T1", "T2"]
            elif method == "quasi":
                required_cols = ["V_t", "delta_V"]
                if not material and "material" not in df.columns:
                    raise ValueError("准稳态法需要指定材料类型或在CSV中包含material列")
            elif method == "cooling":
                required_cols = ["duration", "noise"]
            else:
                raise ValueError(f"不支持的方法: {method}")
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV文件缺少必需的列: {missing_cols}")
            
            # 转换为字典列表
            data = df.to_dict('records')
            
            # 批量预测
            result = await self.batch_predict(method, data, material)
            
            return result
            
        except Exception as e:
            raise ValueError(f"CSV文件处理失败: {str(e)}")
    
    def export_results_to_csv(self, results: List[Dict[str, Any]], 
                             method: str) -> str:
        """将结果导出为CSV格式"""
        rows = []
        
        for item in results:
            if item["success"] and item["result"]:
                row = {
                    "row_index": item["row_index"],
                    "success": item["success"]
                }
                
                if method == "steady":
                    result = item["result"]
                    row.update({
                        "lambda_predicted": result["lambda_predicted"],
                        "T2_corrected": result["T2_corrected"],
                        "correction_a": result["correction_params"]["a"],
                        "correction_b": result["correction_params"]["b"]
                    })
                elif method == "quasi":
                    result = item["result"]
                    row.update({
                        "lambda_predicted": result["lambda_predicted"],
                        "c_predicted": result["c_predicted"],
                        "lambda_theory": result["lambda_theory"],
                        "c_theory": result["c_theory"],
                        "lambda_error": result["lambda_error"],
                        "c_error": result["c_error"]
                    })
                
                rows.append(row)
            else:
                rows.append({
                    "row_index": item["row_index"],
                    "success": item["success"],
                    "error": item["error"]
                })
        
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)