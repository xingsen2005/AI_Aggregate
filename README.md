# AI_Aggregate

## 项目概述

对应大模型平台的 API Keys 请自行准备。

这是一款多 AI 聚合应用，支持用户提交单个问题并同时向多个 AI 模型发起查询，集成各 AI 模型官方 API，实现多 AI 输出结果同步查看，并确保所有 AI 模型能获取其他模型的输出内容。

## 技术栈

### 前端
- Vue 3
- Element Plus
- Axios
- Socket.IO Client
- Vite (构建工具)

### 后端
- Flask
- Flask-SocketIO
- Celery (异步任务处理)
- Redis (消息代理)
- Flask-CORS
- Python unittest (测试框架)

## 项目结构

```
AI Aggregate/
├── frontend/                 # 前端项目
│   ├── src/                  # 前端源码
│   │   ├── main.js           # 入口文件
│   │   └── App.vue           # 主组件
│   ├── package.json          # 前端依赖配置
│   ├── vite.config.js        # Vite构建配置
│   └── index.html            # HTML模板
├── backend/                  # 后端项目
│   ├── app.py                # Flask应用主文件
│   ├── run_backend.py        # 后端启动脚本
│   ├── requirements.txt      # 后端依赖
│   ├── .env                  # 环境变量配置
│   └── tests/                # 测试代码
├── install_frontend_deps.bat # 前端依赖安装脚本(Windows)
├── .gitignore                # Git忽略文件配置
└── README.md                 # 项目说明文档
```

## 安装指南

### 前提条件

- **前端**：Node.js 14+ 和 npm/yarn/pnpm
- **后端**：Python 3.8+ 和 pip
- **可选**：Redis 用于生产环境的Celery消息代理

### 安装步骤

#### 1. 克隆项目

```bash
# 克隆项目到本地
git clone https://github.com/xingsen2005/AI_Aggregate.git
cd AI Aggregate
```

#### 2. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

#### 3. 安装前端依赖

**Windows 用户**：

```bash
# 在项目根目录下运行
install_frontend_deps.bat
```

**其他系统**：

```bash
cd frontend
npm install
```

#### 4. 配置环境变量

在 `backend/.env` 文件中配置所需的环境变量：

```
# AI 模型 API 密钥
DOUBAO_API_KEY=
DEEPSEEK_API_KEY=
OPENAI_API_KEY=
KIMI_API_KEY=
TX_HUNYUAN_API_KEY=
GOOGLE_API_KEY=

# Redis 配置(仅在非模拟模式下需要)
REDIS_URL=redis://localhost:6379/0

# 服务器配置
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5000
```

## 使用指南

### 启动后端服务

```bash
cd backend
python run_backend.py
```

如果 Redis 未安装或不可用，系统将询问是否使用模拟模式启动服务。模拟模式不需要 Redis，适合本地开发和测试。

### 启动前端开发服务器

```bash
cd frontend
npm run dev
```

前端服务将在 `http://localhost:3000` 启动。

### 构建前端生产版本

```bash
cd frontend
npm run build
```

构建后的文件将位于 `frontend/dist` 目录中。

## 测试

### 运行后端测试

```bash
cd backend
python -m unittest discover tests
```

## 故障排除

- **连接问题**：检查后端服务是否正常运行，以及前端 WebSocket 连接 URL 是否正确
- **API密钥错误**：确保 .env 文件中配置了正确的API密钥
- **Redis错误**：如果不需要 Redis，可以使用模拟模式启动服务
- **端口冲突**：如果端口 5000 或 3000 已被占用，可以修改相应的配置文件
