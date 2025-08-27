import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys

def setup_logger(name: str = "thermal_sim", level: str = "INFO") -> logging.Logger:
    """设置统一的日志配置"""
    
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger(name)
    
    # 如果logger已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器 - 所有日志
    all_log_file = os.path.join(log_dir, "thermal_sim.log")
    file_handler = RotatingFileHandler(
        all_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 错误日志文件处理器
    error_log_file = os.path.join(log_dir, "error.log")
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # API访问日志文件处理器
    access_log_file = os.path.join(log_dir, "access.log")
    access_handler = RotatingFileHandler(
        access_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    access_handler.setLevel(logging.INFO)
    
    # API访问日志使用简化格式
    access_formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    access_handler.setFormatter(access_formatter)
    
    # 创建专门的访问日志记录器
    access_logger = logging.getLogger(f"{name}.access")
    access_logger.setLevel(logging.INFO)
    access_logger.addHandler(access_handler)
    access_logger.propagate = False  # 防止重复记录
    
    logger.info(f"日志系统初始化完成 - 日志目录: {log_dir}")
    return logger

def get_logger(name: str = "thermal_sim") -> logging.Logger:
    """获取logger实例"""
    return logging.getLogger(name)

def get_access_logger(name: str = "thermal_sim") -> logging.Logger:
    """获取访问日志记录器"""
    return logging.getLogger(f"{name}.access")

# 预配置的logger实例
main_logger = setup_logger()
access_logger = get_access_logger()