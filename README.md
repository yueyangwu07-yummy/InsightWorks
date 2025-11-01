# FastAPI LangGraph Agent Template

A production-ready FastAPI template for building AI agent applications with LangGraph integration. This template provides a robust foundation for building scalable, secure, and maintainable AI agent services.

## 🌟 Features

- **Production-Ready Architecture**

  - FastAPI for high-performance async API endpoints
  - LangGraph integration for AI agent workflows
  - Langfuse for LLM observability and monitoring
  - Structured logging with environment-specific formatting
  - Rate limiting with configurable rules
  - PostgreSQL for data persistence
  - Docker and Docker Compose support
  - Prometheus metrics and Grafana dashboards for monitoring

- **Security**

  - JWT-based authentication
  - Session management
  - Input sanitization
  - CORS configuration
  - Rate limiting protection

- **Developer Experience**

  - Environment-specific configuration
  - Comprehensive logging system
  - Clear project structure
  - Type hints throughout
  - Easy local development setup

- **Model Evaluation Framework**
  - Automated metric-based evaluation of model outputs
  - Integration with Langfuse for trace analysis
  - Detailed JSON reports with success/failure metrics
  - Interactive command-line interface
  - Customizable evaluation metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- PostgreSQL ([see Database setup](#database-setup))
- Docker and Docker Compose (optional)

### Environment Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd <project-directory>
```

2. Create and activate a virtual environment:

```bash
uv sync
```

3. Copy the example environment file:

```bash
cp .env.example .env.[development|staging|production] # e.g. .env.development
```

4. Update the `.env` file with your configuration (see `.env.example` for reference)

### Database setup

1. Create a PostgreSQL database (e.g Supabase or local PostgreSQL)
2. Update the database connection settings in your `.env` file:

```bash
POSTGRES_HOST=db
POSTGRES_PORT=5432
POSTGRES_DB=cool_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

- You don't have to create the tables manually, the ORM will handle that for you.But if you faced any issues,please run the `schemas.sql` file to create the tables manually.

### Running the Application

#### Local Development

1. Install dependencies:

```bash
uv sync
```

2. Run the application:

```bash
make [dev|staging|production] # e.g. make dev
```

1. Go to Swagger UI:

```bash
http://localhost:8000/docs
```

#### Using Docker

1. Build and run with Docker Compose:

```bash
make docker-build-env ENV=[development|staging|production] # e.g. make docker-build-env ENV=development
make docker-run-env ENV=[development|staging|production] # e.g. make docker-run-env ENV=development
```

2. Access the monitoring stack:

```bash
# Prometheus metrics
http://localhost:9090

# Grafana dashboards
http://localhost:3000
Default credentials:
- Username: admin
- Password: admin
```

The Docker setup includes:

- FastAPI application
- PostgreSQL database
- Prometheus for metrics collection
- Grafana for metrics visualization
- Pre-configured dashboards for:
  - API performance metrics
  - Rate limiting statistics
  - Database performance
  - System resource usage

## 📊 Model Evaluation

The project includes a robust evaluation framework for measuring and tracking model performance over time. The evaluator automatically fetches traces from Langfuse, applies evaluation metrics, and generates detailed reports.

### Running Evaluations

You can run evaluations with different options using the provided Makefile commands:

```bash
# Interactive mode with step-by-step prompts
make eval [ENV=development|staging|production]

# Quick mode with default settings (no prompts)
make eval-quick [ENV=development|staging|production]

# Evaluation without report generation
make eval-no-report [ENV=development|staging|production]
```

### Evaluation Features

- **Interactive CLI**: User-friendly interface with colored output and progress bars
- **Flexible Configuration**: Set default values or customize at runtime
- **Detailed Reports**: JSON reports with comprehensive metrics including:
  - Overall success rate
  - Metric-specific performance
  - Duration and timing information
  - Trace-level success/failure details

### Customizing Metrics

Evaluation metrics are defined in `evals/metrics/prompts/` as markdown files:

1. Create a new markdown file (e.g., `my_metric.md`) in the prompts directory
2. Define the evaluation criteria and scoring logic
3. The evaluator will automatically discover and apply your new metric

### Viewing Reports

Reports are automatically generated in the `evals/reports/` directory with timestamps in the filename:

```
evals/reports/evaluation_report_YYYYMMDD_HHMMSS.json
```

Each report includes:

- High-level statistics (total trace count, success rate, etc.)
- Per-metric performance metrics
- Detailed trace-level information for debugging

## 🔧 Configuration

The application uses a flexible configuration system with environment-specific settings:

- `.env.development`
-

## 数据库与密钥配置
在你的 .env 文件中更新所有必要的配置。

1. 数据库配置
更新数据库连接设置:

你不需要手动创建表，ORM 会自动处理。如果遇到问题，请手动运行 schemas.sql 文件来创建表。

2. Langfuse (可观测性)
添加你的 Langfuse 项目密钥。

你可以登录到你的 LANGFUSE_HOST (例如: https://us.cloud.langfuse.com/) 来查看 Agent 的运行轨迹。

3. Cleanlab (数据质量) - 新增
添加你的 Cleanlab Codex API 密钥和项目 ID。

### 如何获取 CLEANLAB_CODEX_API_KEY:

1. 登录

2. 右上角点击你的头像 → Settings / Account / API Keys

3. 找到 User-level API Key（必须是 User API Key，不是 Project API Key）

点击 Generate New Key（或复制已有的 key）

4. 填入 .env 文件。

### 如何获取 CLEANLAB_PROJECT_ID:

在 Codex 左边栏选择 Projects

找到你创建的项目（如果还没创建就点击 Create Project）

点击进入项目，项目 URL 会是这样：https://codex.cleanlab.ai/projects/abcd1234efg56789

这里的 abcd1234efg56789 就是你的 CLEANLAB_PROJECT_ID。

### 身份验证 & LLM
关于如何获取其他密钥，ChatGPT 获取教程。

## 如何使用 (API 指南)
一份关于如何使用 http://127.0.0.1:8000/docs Swagger UI 的快速指南：

1. 注册: 找到 POST /api/v1/auth/register。点击 "Try it out"，输入你的信息，然后点击 "Execute"。从响应体 (response body) 中复制 access_token。

2. 授权: 点击页面右上角的绿色 "Authorize" 按钮（锁图标）。在 value 字段中输入 Bearer (注意 Bearer 后面有个空格)，然后粘贴你复制的 access_token。点击 "Authorize"。

3. 登录 (可选): 如果你已有账户，可以使用 POST /api/v1/auth/login 登录。

4. 获取 Session: 找到 POST /api/v1/auth/session。点击 "Try it out" 和 "Execute"。从响应体中复制新的 access_token。

5. 再次授权: 再次点击 "Authorize" 按钮（锁图标）。用你刚从 /session 获得的新 access_token 替换旧的 token (确保也包含 Bearer )。

6. 聊天: 你现在已通过身份验证。可以使用聊天端点：

POST /api/v1/chatbot/chat

POST /api/v1/chatbot/chat/stream

## 如何使用Langfuse
需要账号。去这里https://us.cloud.langfuse.com/ 如果你的LANGFUSE_HOST=https://us.cloud.langfuse.com
