from fastapi import Request, Response
try:
    from fastapi.middleware.base import BaseHTTPMiddleware
except ImportError:
    from starlette.middleware.base import BaseHTTPMiddleware
import time
from ..utils.logger import get_access_logger

class AccessLogMiddleware(BaseHTTPMiddleware):
    """访问日志中间件"""
    
    def __init__(self, app):
        super().__init__(app)
        self.access_logger = get_access_logger()
    
    async def dispatch(self, request: Request, call_next):
        # 记录请求开始时间
        start_time = time.time()
        
        # 获取客户端IP
        client_ip = request.client.host if request.client else "unknown"
        
        # 获取用户代理
        user_agent = request.headers.get("user-agent", "unknown")
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录访问日志
        self.access_logger.info(
            f"{client_ip} - {request.method} {request.url.path} - "
            f"{response.status_code} - {process_time:.3f}s - "
            f"User-Agent: {user_agent}"
        )
        
        return response