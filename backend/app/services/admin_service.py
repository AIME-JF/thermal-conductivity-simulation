import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class AdminService:
    def __init__(self):
        # 使用绝对路径，从backend目录开始
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.models_dir = os.path.join(base_dir, "models")
        self.results_dir = os.path.join(base_dir, "results")
        self.training_tasks = {}  # 存储训练任务状态
        self.training_history_file = os.path.join(self.results_dir, "training_history.json")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """获取模型列表"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith(('.keras', '.pkl')):
                filepath = os.path.join(self.models_dir, filename)
                stat = os.stat(filepath)
                
                models.append({
                    "name": filename,
                    "version": "v1.0",  # 可以从文件名或元数据中提取
                    "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "file_path": filepath,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "size": stat.st_size  # 添加字节大小字段
                })
        
        return sorted(models, key=lambda x: x["timestamp"], reverse=True)
    
    async def start_training(self, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """开始模型训练"""
        task_id = str(uuid.uuid4())
        
        # 创建训练任务记录
        task_info = {
            "task_id": task_id,
            "model_type": model_type,
            "config": config,
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "progress": 0,
            "message": "训练任务已启动",
            "log_file": f"../results/training_log_{task_id}.txt"
        }
        
        self.training_tasks[task_id] = task_info
        
        # 异步启动训练任务
        asyncio.create_task(self._run_training_task(task_id, model_type, config))
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": "训练任务已启动，请稍后查询状态",
            "log_file": task_info["log_file"]
        }
    
    async def _run_training_task(self, task_id: str, model_type: str, config: Dict[str, Any]):
        """运行训练任务（模拟）"""
        try:
            task_info = self.training_tasks[task_id]
            log_file = task_info["log_file"]
            
            # 创建日志文件
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"训练开始: {datetime.now()}\n")
                f.write(f"模型类型: {model_type}\n")
                f.write(f"配置: {json.dumps(config, indent=2)}\n\n")
            
            # 模拟训练过程
            epochs = config.get("epochs", 100)
            for epoch in range(epochs):
                # 更新进度
                progress = int((epoch + 1) / epochs * 100)
                task_info["progress"] = progress
                task_info["message"] = f"训练中... Epoch {epoch + 1}/{epochs}"
                
                # 写入日志
                with open(log_file, 'a', encoding='utf-8') as f:
                    loss = 1.0 - (epoch / epochs) * 0.8 + 0.1 * (0.5 - abs(0.5 - (epoch % 10) / 10))
                    f.write(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f}\n")
                
                # 模拟训练时间
                await asyncio.sleep(0.1)
            
            # 训练完成
            end_time = datetime.now()
            start_time = datetime.fromisoformat(task_info["start_time"])
            duration = int((end_time - start_time).total_seconds())
            
            task_info["status"] = "completed"
            task_info["progress"] = 100
            task_info["message"] = "训练完成"
            task_info["end_time"] = end_time.isoformat()
            task_info["duration"] = duration
            task_info["final_metrics"] = {
                "final_loss": 0.1 + (epochs % 10) * 0.01,
                "final_accuracy": 0.95 - (epochs % 10) * 0.005,
                "epochs_completed": epochs
            }
            
            # 创建模型文件（模拟）
            os.makedirs(self.models_dir, exist_ok=True)
            model_filename = f"{model_type}_model_{task_id[:8]}.keras"
            model_path = os.path.join(self.models_dir, model_filename)
            
            # 创建一个模拟的模型文件
            with open(model_path, 'w', encoding='utf-8') as f:
                f.write(f"# 模拟的{model_type}模型文件\n")
                f.write(f"# 训练时间: {datetime.now()}\n")
                f.write(f"# 配置: {json.dumps(config)}\n")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n训练完成: {datetime.now()}\n")
                f.write(f"模型已保存到: {model_path}\n")
            
            # 保存训练历史
            await self._save_training_history(task_info)
            
        except Exception as e:
            task_info["status"] = "failed"
            task_info["message"] = f"训练失败: {str(e)}"
            task_info["end_time"] = datetime.now().isoformat()
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n训练失败: {str(e)}\n")
    
    async def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """获取训练状态"""
        if task_id not in self.training_tasks:
            raise ValueError(f"训练任务不存在: {task_id}")
        
        return self.training_tasks[task_id]
    
    async def _save_training_history(self, task_info: Dict[str, Any]):
        """保存训练历史"""
        history = []
        
        # 读取现有历史
        if os.path.exists(self.training_history_file):
            try:
                with open(self.training_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []
        
        # 添加新记录
        history.append(task_info)
        
        # 保存历史（只保留最近100条）
        history = history[-100:]
        
        os.makedirs(os.path.dirname(self.training_history_file), exist_ok=True)
        with open(self.training_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    async def get_training_history(self) -> List[Dict[str, Any]]:
        """获取训练历史"""
        if not os.path.exists(self.training_history_file):
            return []
        
        try:
            with open(self.training_history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return sorted(history, key=lambda x: x.get("start_time", ""), reverse=True)
        except:
            return []