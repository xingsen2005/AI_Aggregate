# AI_Aggregate

## 项目概述

AI_Aggregate 是一款功能强大的多 AI 聚合应用，支持用户提交单个问题并同时向多个顶尖 AI 模型发起查询，集成各 AI 模型官方 API，实现多 AI 输出结果同步查看与对比分析。

### 核心功能

- **多模型并行查询**：同时向多个 AI 模型发送请求，大幅提高信息获取效率
- **模拟模式**：无需配置真实 API 密钥也可体验完整功能，适合开发测试
- **任务批处理**：高效的批处理机制优化请求处理流程
- **智能缓存**：减少重复查询，提高响应速度
- **自动重试机制**：网络波动时自动重试，提高系统稳定性
- **速率限制**：防止 API 调用超限
- **实时结果更新**：通过 WebSocket 实现结果的实时推送
- **JWT 认证**：支持可选的安全认证机制

> 注意：使用真实 AI 模型功能时，需要在 `.env` 文件中配置对应大模型平台的 API Keys

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

## 支持的AI模型

应用支持以下主流AI模型的集成，每种模型都可以独立启用或禁用：

| 模型ID | 模型名称 | 模型描述 | API密钥环境变量 |
|-------|---------|---------|---------------|
| doubao | Doubao | 豆包AI，由字节跳动开发的智能对话模型 | `DOUBAO_API_KEY` |
| deepseek | DeepSeek | 深度求索AI，专注于代码和自然语言处理 | `DEEPSEEK_API_KEY` |
| chatgpt | ChatGPT | OpenAI开发的通用对话模型 | `OPENAI_API_KEY` |
| kimi | Kimi | 月之暗面开发的长文本理解模型 | `KIMI_API_KEY` |
| hunyuan | HunYuan | 腾讯混元大模型，支持中文优化 | `TX_HUNYUAN_API_KEY` |
| gemini | Gemini | Google开发的多模态AI模型 | `GOOGLE_API_KEY` |

## 项目结构

项目采用前后端分离架构，清晰的模块化设计确保了代码的可维护性和可扩展性。

```
AI_Aggregate/
├── frontend/                 # 前端项目目录
│   ├── src/                  # 前端源码
│   │   ├── main.js           # 入口文件
│   │   ├── App.vue           # 主组件
│   │   └── components/       # 子组件
│   ├── package.json          # 前端依赖配置
│   ├── package-lock.json     # 依赖版本锁定
│   ├── vite.config.js        # Vite构建配置
│   └── index.html            # HTML模板
├── backend/                  # 后端项目目录
│   ├── app.py                # Flask应用主文件
│   ├── run_backend.py        # 后端启动脚本
│   ├── model_handlers.py     # AI模型API处理模块
│   ├── task_manager.py       # 任务管理和批处理
│   ├── utils.py              # 工具函数
│   ├── app_context.py        # 应用上下文
│   ├── requirements.txt      # 后端依赖列表
│   ├── .env.template         # 环境变量配置模板
│   └── tests/                # 测试代码
│       ├── test_basic.py     # 基础功能测试
│       └── test_advanced.py  # 高级功能测试
├── install_frontend_deps.bat # 前端依赖安装脚本(Windows)
├── install_and_start_frontend.bat # 一键安装并启动前端
├── start_frontend.ps1        # PowerShell启动前端脚本
├── .gitignore                # Git忽略文件配置
├── .gitattributes            # Git属性配置
├── LICENSE                   # 开源许可证
└── README.md                 # 项目说明文档
```

## 安装指南

### 前提条件

- **前端**：Node.js 14+ 和 npm/yarn/pnpm
- **后端**：Python 3.8+ 和 pip
- **可选**：Redis 用于生产环境的 Celery 消息代理

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

1. 在 `backend` 目录下，复制 `.env.template` 文件并重命名为 `.env`：

```bash
# Windows
copy .env.template .env

# Linux/Mac
cp .env.template .env
```

2. 编辑 `.env` 文件，配置所需的环境变量：

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

# 可选：JWT 认证配置
JWT_SECRET=

# 可选：模拟模式配置
SIMULATION_MODE=false
```

> 注意：如果您不想配置真实 API 密钥，可以在启动后端服务时选择使用模拟模式，系统会自动设置 `SIMULATION_MODE=true`

## 使用指南

### 启动后端服务

```bash
cd backend
python run_backend.py
```

启动过程会自动检查 Redis 是否可用：

- 如果 Redis 可用，服务将以正常模式启动，使用 Redis 作为 Celery 的消息代理
- 如果 Redis 不可用或未安装，系统将自动使用模拟模式启动服务，此时不需要配置真实的 API 密钥

**模拟模式说明**：
- 模拟模式会生成模拟的 AI 回复，用于开发测试和功能演示
- 模拟模式下不需要 Redis 和真实的 API 密钥
- 模拟模式完整保留了应用的所有交互功能

服务启动成功后，后端 API 将在 `http://localhost:5000/api` 可用

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

- **连接问题**：检查后端服务是否正常运行，以及前端 WebSocket 连接 URL 是否正确；
- **API密钥错误**：确保 .env 文件中配置了正确的API密钥；
- **Redis错误**：如果不需要 Redis，可以使用模拟模式启动服务；
- **端口冲突**：如果端口 5000 或 3000 已被占用，可以修改相应的配置文件；
- **速率限制**：如果遇到 API 调用频率限制，可以在 task_manager.py 中调整 RATE_LIMITER 的配置。

## 项目架构与设计

### 后端架构

项目后端采用了多层架构设计：

1. **API层**：处理HTTP和WebSocket请求，位于 `app.py`
2. **任务处理层**：管理异步任务、批处理和并发请求，位于 `task_manager.py`
3. **模型处理层**：封装各AI模型的API调用，位于 `model_handlers.py`
4. **工具层**：提供通用功能和辅助函数，位于 `utils.py`

### 核心功能实现

#### 多模型并行调用

应用使用 `concurrent.futures` 创建线程池，实现对多个AI模型的并行调用，大幅提高处理效率。每个模型请求都会被封装为独立任务提交到线程池执行。

#### 任务批处理机制

`BatchProcessor` 类实现了智能的任务批处理机制，将短时间内相同模型的多个请求合并处理，减少API调用次数并降低服务器负载。

#### 自动重试与错误处理

`model_handlers.py` 中的 `retry_on_failure` 装饰器实现了API调用失败的自动重试功能，使用指数退避策略，有效提高了系统的稳定性。

#### 结果缓存

系统实现了智能缓存机制，对于相同的查询内容，优先返回缓存结果，减少重复的API调用。

## 部署指南

### 开发环境部署

按照前面的安装指南操作即可完成开发环境的部署。

### 生产环境部署

1. **环境准备**：
   - 安装 Redis 服务器
   - 配置所有必要的 API 密钥
   - 设置强 JWT_SECRET

2. **后端配置**：
   ```
   FLASK_ENV=production
   FLASK_DEBUG=False
   SIMULATION_MODE=false
   ```

3. **前端构建**：
   ```bash
   cd frontend
   npm run build
   ```
   构建后的文件将位于 `frontend/dist` 目录

4. **使用Nginx提供前端服务**，并反向代理后端API请求

## 开发规范

1. **代码风格**：遵循PEP 8规范（Python）和ESLint规范（JavaScript）
2. **提交信息**：清晰描述变更内容，遵循语义化版本控制
3. **测试**：在提交代码前运行单元测试，确保功能正常

## 贡献指南

欢迎对项目进行贡献！请遵循以下步骤：

1. Fork 项目仓库
2. 创建新的功能分支
3. 提交你的更改
4. 推送到远程分支
5. 创建 Pull Request

## 许可证

MIT License
