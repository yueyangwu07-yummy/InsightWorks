# Setup Guide for FastAPI LangGraph Agent

This guide will help you set up the project quickly and efficiently.

## Prerequisites

- Python 3.13+
- PostgreSQL database (for state management)
- API keys for:
  - OpenAI (or compatible LLM API)
  - Cleanlab Codex (optional, for AI response validation)
  - Langfuse (optional, for monitoring and tracing)

## Quick Start

### Option 1: Using `uv` (Recommended)

`uv` is a fast Python package installer and resolver. Install it first:

```bash
# Install uv
pip install uv

# Clone the repository
git clone <repository-url>
cd fastapi-langgraph-agent-production-ready-template

# Checkout the stable version
git checkout v2.0-complete-integration

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv pip install -e .

# Or use the pinned requirements
uv pip install -r requirements.txt
```

### Option 2: Using `pip`

```bash
# Clone the repository
git clone <repository-url>
cd fastapi-langgraph-agent-production-ready-template

# Checkout the stable version
git checkout v2.0-complete-integration

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Or use the pinned requirements
pip install -r requirements.txt
```

## Environment Configuration

### 1. Create `.env` file

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Configure environment variables

Edit `.env` and set the following variables:

```env
# Basic Configuration
ENVIRONMENT=development
PROJECT_NAME=langgraph-fastapi-template

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
MAX_TOKENS=4096

# PostgreSQL Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Supabase (for user management)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# JWT Configuration
JWT_SECRET_KEY=your_jwt_secret_key_here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Cleanlab Codex (Optional - for AI response validation)
CLEANLAB_CODEX_API_KEY=your_cleanlab_api_key
CLEANLAB_PROJECT_ID=your_cleanlab_project_id

# Langfuse (Optional - for monitoring and tracing)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://us.cloud.langfuse.com

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

### 3. Environment-specific configuration

You can also create `.env.development`, `.env.staging`, or `.env.production` files for environment-specific settings. The system will automatically load the appropriate file based on the `ENVIRONMENT` variable.

## Database Setup

### 1. Create PostgreSQL database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE langgraph_agent;

# Exit psql
\q
```

### 2. Run database migrations (if applicable)

```bash
# Initialize database schema
python -m alembic upgrade head
```

## Running the Application

### Development Mode

```bash
# Run the FastAPI server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# Run with gunicorn (if installed)
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Verify Installation

1. **Health Check**: Visit `http://localhost:8000/health`

2. **API Documentation**: Visit `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc` (ReDoc)

3. **Test Chat**: Use the `/chat` endpoint to test the LLM agent

## Key Features

### ✅ Version 2.0 Features

- **Cleanlab Integration**: AI response validation and quality monitoring
- **Langfuse 3.x Integration**: Complete monitoring and tracing
- **Dual Endpoint Support**: Both `/chat` and `/chat/stream` endpoints
- **Robust Error Handling**: Graceful degradation when optional services are unavailable
- **Comprehensive Logging**: Detailed diagnostic logs for troubleshooting

### Core Components

1. **LangGraph Agent**: Orchestrates multi-step LLM workflows
2. **FastAPI**: Web framework for API endpoints
3. **PostgreSQL**: State management and conversation history
4. **Supabase**: User authentication and management
5. **OpenTelemetry**: Distributed tracing via Langfuse
6. **Cleanlab**: AI response quality validation

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   ```bash
   # Make sure virtual environment is activated
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Reinstall dependencies
   uv pip install -e . --force-reinstall
   ```

2. **Database connection errors**:
   - Verify PostgreSQL is running
   - Check `DATABASE_URL` in `.env`
   - Ensure database exists

3. **Langfuse not tracking**:
   - Check API keys are set correctly
   - Verify network connectivity to Langfuse
   - Check logs for diagnostic messages

4. **Cleanlab validation failing**:
   - Ensure API key and Project ID are set
   - Check Cleanlab project is active
   - Review logs for validation errors

### Getting Help

- Check logs in the console output
- Review `app/core/logging.py` for log configuration
- Check environment-specific logs in the logs directory

## Project Structure

```
fastapi-langgraph-agent-production-ready-template/
├── app/
│   ├── api/               # API endpoints
│   ├── core/              # Core functionality
│   │   ├── config.py      # Configuration
│   │   ├── langgraph/     # LangGraph agent
│   │   ├── logging.py     # Logging setup
│   │   └── prompts.py     # System prompts
│   ├── models/            # Database models
│   ├── schemas/           # Pydantic schemas
│   └── main.py            # FastAPI application
├── pyproject.toml         # Project configuration
├── requirements.txt       # Pinned dependencies
└── .env                   # Environment variables (not in git)
```

## Next Steps

1. Configure your API keys in `.env`
2. Set up your PostgreSQL database
3. Run the application in development mode
4. Explore the API documentation at `/docs`
5. Customize the system prompts in `app/core/prompts/`
6. Add your own tools to `app/core/langgraph/tools.py`

## Version History

- **v2.0-complete-integration**: Complete Langfuse 3.x and Cleanlab integration
- **v1.0-cleanlab-integration**: Initial Cleanlab integration and Langfuse fixes

## License

MIT License

