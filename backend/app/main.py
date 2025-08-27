from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .routers import steady, quasi, admin, batch
from .middleware.exception_handler import setup_exception_handlers
from .middleware.access_log import AccessLogMiddleware

app = FastAPI(
    title="热传导实验虚拟仿真API",
    description="稳态法和准稳态法导热系数测量实验的后端API",
    version="1.0.0"
)

# 设置全局异常处理器
setup_exception_handlers(app)

# 添加访问日志中间件
app.add_middleware(AccessLogMiddleware)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://localhost:3010", "http://localhost:8000", "http://119.45.35.210:3010", "http://119.45.35.210:8090"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册
app.include_router(steady.router, prefix="/api/steady", tags=["稳态法"])
app.include_router(quasi.router, prefix="/api/quasi", tags=["准稳态法"])
app.include_router(batch.router, prefix="/api/batch", tags=["批量处理"])
app.include_router(admin.router, prefix="/api/admin", tags=["管理员"])

# 静态文件服务
import os
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
app.mount("/results", StaticFiles(directory=results_dir), name="results")

@app.get("/")
async def root():
    return {"message": "热传导实验虚拟仿真API服务正在运行"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}