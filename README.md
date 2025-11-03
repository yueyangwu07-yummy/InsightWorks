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

- **Long-term Memory (Hybrid A+B)**
  - **A: Periodic Summaries** - Automatically generated conversation summaries stored in vector store (every N turns, default: 10)
  - **B: Stable Facts** - User profile data (VIN, preferences, timezone) stored in both PostgreSQL `UserProfile` table and vector store for semantic retrieval
  - Automatic memory classification via LLM to detect memory-worthy content
  - Semantic retrieval with configurable similarity thresholds
  - PII-safe Langfuse observability for all memory operations
  - **Guardrails**: Similarity threshold filtering, rate limiting, token caps, feature flags
  - **See**: [Mermaid Flow Diagram](docs/memory_flow.mermaid.md) | [PlantUML Sequence Diagram](docs/memory_flow.puml) | [Langfuse Dashboard Guide](docs/langfuse_memory_dashboard.md)

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

### Long-term Memory Setup

The memory system uses a hybrid approach with PostgreSQL for structured data and Qdrant for vector storage:

**Qdrant Configuration Options:**

The memory system supports both local and cloud Qdrant deployments:

**Option 1: Local Docker (Development)**
For local development, run Qdrant in Docker without authentication:

```bash
# Start Qdrant container
docker run -p 6333:6333 qdrant/qdrant

# In .env.development
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=none  # No authentication required for local
FEATURE_LONG_TERM_MEMORY=true
FEATURE_MEMORY_CLASSIFIER=true
FEATURE_MEMORY_RETRIEVAL=true
```

**Option 2: Qdrant Cloud (Production)**
For production, use Qdrant Cloud with API key authentication:

```bash
# In .env.production
QDRANT_URL=https://your-cluster.qdrant.io  # Your Qdrant Cloud cluster URL
QDRANT_API_KEY=your_cloud_api_key          # Get from Qdrant Cloud dashboard
FEATURE_LONG_TERM_MEMORY=true
FEATURE_MEMORY_CLASSIFIER=true
FEATURE_MEMORY_RETRIEVAL=true
```

**Key Differences:**
- **Local**: `QDRANT_API_KEY=none` - No authentication, suitable for development
- **Cloud**: `QDRANT_API_KEY=<real_key>` - Secure API key from Qdrant Cloud dashboard, required for production

**Memory Configuration:**
```bash
# Embedding model for semantic search
EMBEDDING_MODEL=text-embedding-3-small

# Retrieval settings
MEM_TOP_K=3                      # Number of memories to retrieve
MEM_MIN_SCORE=0.55               # Minimum similarity threshold (0.0-1.0)

# Feature flags (toggle on/off)
FEATURE_LONG_TERM_MEMORY=true    # Enable memory storage
FEATURE_MEMORY_CLASSIFIER=true   # Enable automatic classification
FEATURE_MEMORY_RETRIEVAL=true    # Enable semantic retrieval
```

**Guardrails:**

The memory system includes multiple safety mechanisms:

- **Similarity Threshold**: Only memories with score >= `MEM_MIN_SCORE` (default: 0.55) are injected into prompts
- **Rate Limiting**: Existing chat endpoint rate limits apply to memory operations
- **Token Cap**: Conversation summaries are limited to `MAX_TOKENS // 2` to prevent prompt bloat
- **Feature Flags**: Each component can be toggled independently via environment variables:
  - `FEATURE_LONG_TERM_MEMORY` - Enable/disable memory storage
  - `FEATURE_MEMORY_CLASSIFIER` - Enable/disable automatic classification
  - `FEATURE_MEMORY_RETRIEVAL` - Enable/disable semantic search
  - Set to `false` or omit to disable specific features

**Documentation:**
- **Flow Diagrams**: [Mermaid Flowchart](docs/memory_flow.mermaid.md) | [PlantUML Sequence](docs/memory_flow.puml)
- **Observability**: [Langfuse Memory Dashboard Guide](docs/langfuse_memory_dashboard.md)

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
