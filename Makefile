# 热传导实验虚拟仿真Web应用 Makefile

.PHONY: help install dev build clean docker-build docker-up docker-down test

# 默认目标
help:
	@echo "可用命令:"
	@echo "  install      - 安装所有依赖"
	@echo "  dev          - 启动开发环境"
	@echo "  build        - 构建生产版本"
	@echo "  clean        - 清理构建文件"
	@echo "  docker-build - 构建Docker镜像"
	@echo "  docker-up    - 启动Docker服务"
	@echo "  docker-down  - 停止Docker服务"
	@echo "  test         - 运行测试"

# 安装依赖
install:
	@echo "安装前端依赖..."
	cd frontend && npm install
	@echo "安装后端依赖..."
	cd backend && pip install -r requirements.txt
	@echo "依赖安装完成!"

# 开发环境
dev:
	@echo "启动开发环境..."
	@echo "后端服务: http://localhost:8090"
	@echo "前端服务: http://localhost:3010"
	@echo "API文档: http://localhost:8090/docs"
	start cmd /k "cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload"
	start cmd /k "cd frontend && npm run dev"

# 构建生产版本
build:
	@echo "构建前端..."
	cd frontend && npm run build
	@echo "构建完成!"

# 清理文件
clean:
	@echo "清理构建文件..."
	if exist frontend\dist rmdir /s /q frontend\dist
	if exist frontend\node_modules rmdir /s /q frontend\node_modules
	if exist backend\__pycache__ rmdir /s /q backend\__pycache__
	@echo "清理完成!"

# Docker操作
docker-build:
	@echo "构建Docker镜像..."
	docker-compose build

docker-up:
	@echo "启动Docker服务..."
	docker-compose up -d
	@echo "服务已启动:"
	@echo "  前端: http://localhost:3000"
	@echo "  后端: http://localhost:8090"
	@echo "  API文档: http://localhost:8090/docs"

docker-down:
	@echo "停止Docker服务..."
	docker-compose down

# 测试
test:
	@echo "运行测试..."
	cd backend && python -m pytest tests/ -v
	cd frontend && npm run test

# 快速启动（推荐）
quick-start: install build docker-up
	@echo "应用已启动，访问 http://localhost:3000"