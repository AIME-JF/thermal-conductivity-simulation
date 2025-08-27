import asyncio
import time
import gzip
from typing import Dict, Any, Callable, Optional
from functools import wraps, lru_cache
import threading
import queue
from contextlib import asynccontextmanager
try:
    from fastapi.middleware.base import BaseHTTPMiddleware
    from fastapi import Request, Response
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    BaseHTTPMiddleware = object
    Request = Any
    Response = Any
import json
import hashlib

try:
    from ..utils.logger import get_logger
    logger = get_logger("thermal_sim.performance")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

if FASTAPI_AVAILABLE:
    class PerformanceMiddleware(BaseHTTPMiddleware):
        """性能监控中间件"""
        
        def __init__(self, app):
            super().__init__(app)
            self.logger = logging.getLogger(__name__)
        
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            start_time = time.time()
            
            # 记录请求开始
            self.logger.info(f"开始处理请求: {request.method} {request.url}")
            
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 添加性能头
            response.headers["X-Process-Time"] = str(process_time)
            
            # 记录请求完成
            self.logger.info(f"请求完成: {request.method} {request.url} - {process_time:.4f}s")
            
            return response
else:
    class PerformanceMiddleware:
        """性能监控中间件（模拟版本）"""
        def __init__(self, app):
            self.app = app
            self.logger = logging.getLogger(__name__)

if FASTAPI_AVAILABLE:
    class CacheMiddleware(BaseHTTPMiddleware):
        """缓存中间件"""
        
        def __init__(self, app, cache_size: int = 1000):
            super().__init__(app)
            self.cache = {}
            self.cache_size = cache_size
            self.logger = logging.getLogger(__name__)
        
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            # 只缓存GET请求
            if request.method != "GET":
                return await call_next(request)
            
            # 生成缓存键
            cache_key = f"{request.method}:{request.url}"
            
            # 检查缓存
            if cache_key in self.cache:
                self.logger.info(f"Cache hit: {cache_key}")
                cached_response = self.cache[cache_key]
                return Response(
                    content=cached_response["content"],
                    status_code=cached_response["status_code"],
                    headers=cached_response["headers"]
                )
            
            # 处理请求
            response = await call_next(request)
            
            # 缓存响应（只缓存成功的响应）
            if response.status_code == 200:
                # 限制缓存大小
                if len(self.cache) >= self.cache_size:
                    # 删除最旧的缓存项
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                
                # 读取响应内容
                content = b""
                async for chunk in response.body_iterator:
                    content += chunk
                
                # 缓存响应
                self.cache[cache_key] = {
                    "content": content,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }
                
                self.logger.info(f"Cached response: {cache_key}")
                
                # 返回新的响应对象
                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=response.headers
                )
            
            return response
else:
    class CacheMiddleware:
        """缓存中间件（模拟版本）"""
        def __init__(self, app, cache_size: int = 1000):
            self.app = app
            self.cache = {}
            self.cache_size = cache_size
            self.logger = logging.getLogger(__name__)

if FASTAPI_AVAILABLE:
    class CompressionMiddleware(BaseHTTPMiddleware):
        """响应压缩中间件"""
        
        def __init__(self, app, min_size: int = 1024):
            super().__init__(app)
            self.min_size = min_size
        
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            response = await call_next(request)
            
            # 检查是否支持gzip压缩
            accept_encoding = request.headers.get("accept-encoding", "")
            if "gzip" not in accept_encoding:
                return response
            
            # 读取响应内容
            content = b""
            async for chunk in response.body_iterator:
                content += chunk
            
            # 只压缩足够大的响应
            if len(content) < self.min_size:
                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=response.headers
                )
            
            # 压缩内容
            compressed_content = gzip.compress(content)
            
            # 创建新的响应
            headers = dict(response.headers)
            headers["content-encoding"] = "gzip"
            headers["content-length"] = str(len(compressed_content))
            
            logger.info(f"Compressed response: {len(content)} -> {len(compressed_content)} bytes")
            
            return Response(
                content=compressed_content,
                status_code=response.status_code,
                headers=headers
            )
else:
    class CompressionMiddleware:
        """响应压缩中间件（模拟版本）"""
        def __init__(self, app, min_size: int = 1024):
            self.app = app
            self.min_size = min_size

# 缓存装饰器
def cached_result(ttl: int = 300):
    """结果缓存装饰器"""
    def decorator(func):
        cache = {}
        
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 检查缓存
            if cache_key in cache:
                entry = cache[cache_key]
                if time.time() - entry["timestamp"] < ttl:
                    logger.info(f"Cache hit for {func.__name__}")
                    return entry["result"]
            
            # 执行函数
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # 缓存结果
            cache[cache_key] = {
                "timestamp": time.time(),
                "result": result
            }
            
            logger.info(f"Cached result for {func.__name__}")
            return result
        
        return wrapper
    return decorator

# 批量处理优化
class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, batch_size: int = 10, timeout: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_requests = []
        self.processing = False
    
    async def add_request(self, request_data):
        """添加请求到批次"""
        self.pending_requests.append(request_data)
        
        # 如果达到批次大小或超时，处理批次
        if len(self.pending_requests) >= self.batch_size or not self.processing:
            await self._process_batch()
    
    async def _process_batch(self):
        """处理批次"""
        if self.processing or not self.pending_requests:
            return
        
        self.processing = True
        batch = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]
        
        try:
            # 并行处理批次中的请求
            tasks = [self._process_single_request(req) for req in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info(f"Processed batch of {len(batch)} requests")
            return results
        
        finally:
            self.processing = False
    
    async def _process_single_request(self, request_data):
        """处理单个请求"""
        # 这里实现具体的请求处理逻辑
        await asyncio.sleep(0.1)  # 模拟处理时间
        return {"status": "processed", "data": request_data}

# 连接池优化
class ConnectionPool:
    """连接池管理"""
    
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.active_connections = 0
        self.semaphore = asyncio.Semaphore(max_connections)
    
    async def acquire(self):
        """获取连接"""
        await self.semaphore.acquire()
        self.active_connections += 1
        logger.debug(f"Acquired connection. Active: {self.active_connections}")
    
    async def release(self):
        """释放连接"""
        self.active_connections -= 1
        self.semaphore.release()
        logger.debug(f"Released connection. Active: {self.active_connections}")
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()

# 全局连接池实例
connection_pool = ConnectionPool()