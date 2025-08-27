from fastapi import HTTPException
import os
from typing import Optional

# 管理员令牌（生产环境应使用环境变量）
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "admin_secret_token_2024")

def verify_admin_token(token: str) -> bool:
    """验证管理员令牌"""
    if not token:
        raise HTTPException(status_code=401, detail="缺少管理员令牌")
    
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="无效的管理员令牌")
    
    return True

def get_current_admin(token: Optional[str] = None) -> dict:
    """获取当前管理员信息"""
    if verify_admin_token(token):
        return {
            "username": "admin",
            "role": "administrator",
            "permissions": ["read", "write", "train", "manage"]
        }