from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
from typing import Union
from ..utils.logger import get_logger, get_access_logger

# 获取日志记录器
logger = get_logger("thermal_sim.exception")
access_logger = get_access_logger()

class ExceptionHandler:
    """统一异常处理器"""
    
    @staticmethod
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP异常处理器"""
        client_ip = request.client.host if request.client else "unknown"
        
        # 记录错误日志和访问日志
        logger.error(f"HTTP异常 - IP: {client_ip}, 路径: {request.url.path}, 状态码: {exc.status_code}, 详情: {exc.detail}")
        access_logger.info(f"{client_ip} - {request.method} {request.url.path} - {exc.status_code} - HTTP异常: {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        )
    
    @staticmethod
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """请求验证异常处理器"""
        client_ip = request.client.host if request.client else "unknown"
        
        # 提取验证错误详情
        error_details = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            error_details.append(f"{field}: {message}")
        
        error_message = "输入参数验证失败: " + "; ".join(error_details)
        
        # 记录错误日志和访问日志
        logger.error(f"参数验证异常 - IP: {client_ip}, 路径: {request.url.path}, 错误: {error_message}")
        access_logger.info(f"{client_ip} - {request.method} {request.url.path} - 422 - 参数验证失败")
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "VALIDATION_ERROR",
                "message": error_message,
                "details": exc.errors(),
                "status_code": 422,
                "path": str(request.url.path)
            }
        )
    
    @staticmethod
    async def general_exception_handler(request: Request, exc: Exception):
        """通用异常处理器"""
        client_ip = request.client.host if request.client else "unknown"
        
        # 记录详细错误日志
        error_traceback = traceback.format_exc()
        logger.error(f"未处理异常 - IP: {client_ip}, 路径: {request.url.path}, 错误: {str(exc)}\n{error_traceback}")
        
        # 根据异常类型返回不同的错误信息
        if isinstance(exc, ValueError):
            status_code = 400
            error_type = "VALUE_ERROR"
            message = f"参数值错误: {str(exc)}"
        elif isinstance(exc, FileNotFoundError):
            status_code = 503
            error_type = "FILE_NOT_FOUND"
            message = "模型文件未找到，请联系管理员"
        elif isinstance(exc, MemoryError):
            status_code = 507
            error_type = "MEMORY_ERROR"
            message = "服务器内存不足，请稍后重试"
        elif isinstance(exc, TimeoutError):
            status_code = 504
            error_type = "TIMEOUT_ERROR"
            message = "请求处理超时，请稍后重试"
        else:
            status_code = 500
            error_type = "INTERNAL_ERROR"
            message = "服务器内部错误，请稍后重试"
        
        # 记录访问日志
        access_logger.info(f"{client_ip} - {request.method} {request.url.path} - {status_code} - 服务器内部错误")
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error_type,
                "message": message,
                "status_code": status_code,
                "path": str(request.url.path)
            }
        )
    
    @staticmethod
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        """Starlette HTTP异常处理器"""
        client_ip = request.client.host if request.client else "unknown"
        
        # 记录错误日志和访问日志
        logger.error(f"Starlette HTTP异常 - IP: {client_ip}, 路径: {request.url.path}, 状态码: {exc.status_code}, 详情: {exc.detail}")
        access_logger.info(f"{client_ip} - {request.method} {request.url.path} - {exc.status_code} - Starlette异常: {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        )

# 便捷函数
def setup_exception_handlers(app):
    """设置异常处理器"""
    app.add_exception_handler(HTTPException, ExceptionHandler.http_exception_handler)
    app.add_exception_handler(RequestValidationError, ExceptionHandler.validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, ExceptionHandler.starlette_exception_handler)
    app.add_exception_handler(Exception, ExceptionHandler.general_exception_handler)
    
    logger.info("异常处理器已设置完成")