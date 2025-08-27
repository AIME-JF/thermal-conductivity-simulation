# 热传导实验虚拟仿真Web应用

基于稳态法和准稳态法的导热系数测量虚拟仿真实验平台，支持在线参数设定、实时曲线显示、神经网络修正和数据导出功能。

## 🌟 功能特性

### 🔬 实验方法
- **稳态法**：铜块冷却实验，支持T1→T2修正和神经网络预测
- **准稳态法**：有机玻璃/橡胶材料导热系数测量
- **冷却仿真**：物理建模的温度变化仿真

### 📊 核心功能
- 🎯 **智能预测**：基于TensorFlow的神经网络模型
- 📈 **实时图表**：基于Recharts的动态温度曲线
- 🔄 **冷却仿真**：物理建模的温度变化模拟
- 📁 **批量处理**：CSV文件上传批量预测
- 📊 **数据导出**：实验结果CSV格式导出
- 🎛️ **模型管理**：管理员模式支持模型训练和管理
- 📱 **响应式设计**：支持桌面和移动设备

### 🎯 技术栈
- **前端**：React 18 + TypeScript + Vite + Tailwind CSS + Recharts
- **后端**：FastAPI + Python 3.10+ + TensorFlow + NumPy + Pandas + Uvicorn
- **部署**：Docker + Docker Compose + Nginx
- **开发工具**：ESLint + PostCSS + Makefile

## 🚀 快速开始

### 📋 环境要求
- Node.js 18+
- Python 3.10+
- Docker（可选，用于容器化部署）

### 💻 本地开发

#### 1. 克隆项目
```bash
git clone <repository-url>
cd thermal-simulation-web
```

#### 2. 使用Makefile快速启动
```bash
# 安装所有依赖
make install

# 快速启动（推荐）
make quick-start
```

#### 3. 手动启动

**启动后端服务：**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8090
```

**启动前端服务：**
```bash
cd frontend
npm install
npm run dev
```

#### 4. 访问应用
- 🌐 **前端应用**：http://localhost:5173
- 📚 **API文档**：http://localhost:8090/docs
- ❤️ **健康检查**：http://localhost:8090/health

### 🐳 Docker部署

#### 一键启动
```bash
# 构建并启动所有服务
docker-compose up -d

# 或使用Makefile
make docker-up
```

#### 访问应用
- 🌐 **前端应用**：http://localhost:3010
- 🔧 **后端API**：http://localhost:8090

## 📁 项目结构

```
.
├── frontend/                 # React前端应用
│   ├── src/
│   │   ├── pages/           # 页面组件
│   │   │   ├── Home.tsx     # 实验选择主页
│   │   │   ├── SteadyState.tsx  # 稳态法实验台
│   │   │   ├── QuasiSteady.tsx  # 准稳态法实验台
│   │   │   └── Admin.tsx    # 管理员控制台
│   │   ├── components/      # 通用组件
│   │   │   ├── ModelSelector.tsx    # 模型选择器
│   │   │   ├── ParameterInput.tsx   # 参数输入组件
│   │   │   ├── ResultDisplay.tsx    # 结果显示组件
│   │   │   └── ChartDisplay.tsx     # 图表显示组件
│   │   ├── lib/            # API客户端和工具
│   │   │   ├── api.ts      # API接口封装
│   │   │   └── utils.ts    # 工具函数
│   │   ├── hooks/          # React自定义Hooks
│   │   └── types/          # TypeScript类型定义
│   ├── public/             # 静态资源
│   ├── Dockerfile          # 前端Docker配置
│   ├── package.json        # 前端依赖配置
│   └── vite.config.ts      # Vite构建配置
├── backend/                  # FastAPI后端服务
│   ├── app/
│   │   ├── routers/        # API路由模块
│   │   │   ├── steady.py   # 稳态法API接口
│   │   │   ├── quasi.py    # 准稳态法API接口
│   │   │   ├── admin.py    # 管理员API接口
│   │   │   └── batch.py    # 批量处理API接口
│   │   ├── services/       # 核心业务逻辑
│   │   │   ├── steady_service.py    # 稳态法算法实现
│   │   │   ├── quasi_service.py     # 准稳态法算法实现
│   │   │   └── cooling_sim.py       # 冷却仿真算法
│   │   ├── schemas/        # Pydantic数据模型
│   │   ├── utils/          # 工具函数和中间件
│   │   │   ├── auth.py     # 身份验证
│   │   │   └── logger.py   # 日志配置
│   │   └── main.py         # FastAPI应用入口
│   ├── models/             # 训练好的神经网络模型
│   │   ├── steady_state_model.keras
│   │   ├── quasi_steady_model.keras
│   │   └── *.pkl          # 数据标准化器
│   ├── results/            # 预测结果存储
│   ├── data/              # 训练数据集
│   ├── Dockerfile          # 后端Docker配置
│   └── requirements.txt    # Python依赖配置
├── nginx/                   # Nginx配置
│   └── nginx.conf          # 反向代理配置
├── docker-compose.yml       # Docker编排配置
├── Makefile                # 开发工具脚本
├── .env.example            # 环境变量示例
└── README.md               # 项目文档
```

## 🔌 API接口文档

### 🌡️ 稳态法相关
- `POST /api/steady/predict` - 稳态法导热系数预测
  - 输入：温度数据、材料参数
  - 输出：导热系数λ、修正参数
- `POST /api/steady/simulate-cooling` - 铜块冷却仿真
  - 输入：初始温度、环境参数
  - 输出：温度-时间曲线数据

### 🔄 准稳态法相关
- `POST /api/quasi/predict` - 准稳态法材料属性预测
  - 输入：温度数据、几何参数
  - 输出：导热系数λ、比热容c

### 📊 批量处理
- `POST /api/batch/predict` - 批量数据预测
  - 输入：CSV文件上传
  - 输出：批量预测结果CSV

### 🛠️ 管理员功能
- `GET /api/admin/models` - 获取可用模型列表
- `POST /api/admin/train` - 启动模型训练任务
- `GET /api/admin/training-status/{task_id}` - 查询训练进度
- `GET /api/admin/training-history` - 获取训练历史记录

### 🔍 系统接口
- `GET /health` - 系统健康检查
- `GET /docs` - Swagger API文档
- `GET /redoc` - ReDoc API文档

📚 **详细API文档**：http://localhost:8090/docs

## 🧮 核心算法

### 🌡️ 稳态法算法
1. **T1→T2修正算法**：
   - 使用线性修正方法优化温度测量误差
   - 采用L-BFGS-B优化算法求解最优修正参数
   - 支持实时参数调整和结果验证

2. **导热系数计算**：
   - 基于傅里叶传热定律的理论计算
   - 结合材料属性和几何参数的综合分析
   - 神经网络端到端预测模型

3. **冷却仿真模型**：
   - 物理建模：组合冷却定律（自然对流+辐射传热）
   - 数值求解：改进欧拉法求解微分方程
   - 数据处理：Savitzky-Golay滤波平滑处理

### 🔄 准稳态法算法
1. **理论公式计算**：
   - 基于非稳态传热学基本方程
   - 考虑材料热物性参数的温度依赖性
   - 多维传热问题的数值解法

2. **神经网络预测**：
   - 多任务学习架构同时预测导热系数λ和比热容c
   - 深度神经网络提取温度变化特征
   - 误差分析和预测置信度评估

### 🤖 机器学习模型
- **TensorFlow/Keras**：深度学习框架
- **模型架构**：全连接神经网络 + 批归一化 + Dropout
- **优化器**：Adam优化算法
- **损失函数**：均方误差（MSE）
- **数据预处理**：StandardScaler标准化

## ⚙️ 配置说明

### 🔐 环境变量配置
创建 `.env` 文件并配置以下变量：

```bash
# 后端配置
BACKEND_PORT=8090
ADMIN_TOKEN=admin_secret_token_2024

# 前端配置
VITE_API_BASE_URL=http://localhost:8090

# 数据库配置（可选）
DATABASE_URL=sqlite:///./thermal_simulation.db

# 日志配置
LOG_LEVEL=INFO
```

### 📁 模型文件结构
```
models/
├── steady_state_model.keras      # 稳态法神经网络模型
├── quasi_steady_model.keras       # 准稳态法神经网络模型
├── steady_scaler_X.pkl           # 稳态法输入特征标准化器
├── steady_scaler_y.pkl           # 稳态法输出标准化器
├── quasi_scaler_X.pkl            # 准稳态法输入特征标准化器
└── quasi_scaler_y.pkl            # 准稳态法输出标准化器
```

### 🔧 Nginx配置
生产环境下的反向代理配置：
```nginx
server {
    listen 80;
    server_name localhost;
    
    # 前端静态文件
    location / {
        proxy_pass http://frontend:3000;
    }
    
    # 后端API
    location /api/ {
        proxy_pass http://backend:8090;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 👨‍💻 开发指南

### 🆕 添加新实验方法
1. **后端开发**：
   ```bash
   # 1. 创建服务类
   touch backend/app/services/new_method_service.py
   
   # 2. 添加API路由
   touch backend/app/routers/new_method.py
   
   # 3. 更新main.py注册路由
   ```

2. **前端开发**：
   ```bash
   # 1. 创建页面组件
   touch frontend/src/pages/NewMethod.tsx
   
   # 2. 添加路由配置
   # 3. 更新导航菜单
   ```

### 🧠 模型训练流程
1. **数据准备**：
   - 准备CSV格式的训练数据
   - 确保数据格式符合API要求
   - 进行数据清洗和预处理

2. **训练执行**：
   - 访问管理员页面：http://localhost:5173/admin
   - 设置管理员令牌：`localStorage.setItem('admin_token', 'admin_secret_token_2024')`
   - 上传训练数据并配置训练参数
   - 监控训练进度和损失曲线

3. **模型部署**：
   - 训练完成后自动保存模型文件
   - 重启后端服务加载新模型
   - 验证模型预测效果

### 🔍 代码质量保证
```bash
# 前端代码检查
cd frontend
npm run lint
npm run type-check

# 后端代码检查
cd backend
python -m flake8 app/
python -m mypy app/

# 运行测试
python -m pytest tests/
```

## 🚨 故障排除

### ❗ 常见问题及解决方案

#### 1. **前端无法连接后端**
```bash
# 检查后端服务状态
curl http://localhost:8090/health

# 检查环境变量配置
cat frontend/.env

# 检查CORS配置
# 确保backend/app/main.py中允许前端域名
```

#### 2. **模型加载失败**
```bash
# 检查模型文件是否存在
ls -la backend/models/

# 检查TensorFlow版本兼容性
pip list | grep tensorflow

# 重新训练模型
# 访问管理员页面重新训练
```

#### 3. **管理员功能无法访问**
```javascript
// 在浏览器控制台设置管理员令牌
localStorage.setItem('admin_token', 'admin_secret_token_2024');

// 检查令牌是否正确设置
console.log(localStorage.getItem('admin_token'));
```

#### 4. **Docker部署问题**
```bash
# 检查容器状态
docker-compose ps

# 查看容器日志
docker-compose logs backend
docker-compose logs frontend

# 重新构建镜像
docker-compose build --no-cache
```

### 📋 日志查看
```bash
# 查看后端日志
docker-compose logs -f backend

# 查看前端日志
docker-compose logs -f frontend

# 查看Nginx日志
docker-compose logs -f nginx

# 查看所有服务日志
docker-compose logs -f
```

### 🔧 性能优化建议
1. **前端优化**：
   - 启用代码分割和懒加载
   - 优化图表渲染性能
   - 使用React.memo减少不必要的重渲染

2. **后端优化**：
   - 使用异步处理长时间运行的任务
   - 实现结果缓存机制
   - 优化神经网络推理速度

3. **部署优化**：
   - 使用CDN加速静态资源
   - 配置Gzip压缩
   - 实现负载均衡

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Wiki: [项目Wiki](https://github.com/your-repo/wiki)

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！