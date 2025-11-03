# Hybrid Long-term Memory System - Implementation Summary

## Overview

We extended the existing FastAPI + LangGraph agent template with a production-grade hybrid long-term memory system. The system combines structured PostgreSQL storage with semantic vector search (Qdrant) to enable persistent, context-aware conversations across sessions.

## Core Architecture: Hybrid A+B Approach

### Component A: Periodic Summaries
- **Purpose**: Compress conversation history to prevent context window bloat
- **Trigger**: Automatic summary generation every N conversation turns (default: 10)
- **Storage**: Vector store (Qdrant) only for semantic retrieval
- **Implementation**: LLM-generated 2-3 sentence summaries stored as embeddings

### Component B: Stable Facts
- **Purpose**: Persist user-specific stable information (VIN, preferences, timezone)
- **Detection**: LLM-based classifier identifies memory-worthy content
- **Storage**: Dual storage approach:
  - **PostgreSQL** (`UserProfile` table): Structured, queryable data
  - **Qdrant**: Embeddings for semantic search during conversations
- **Types**: Vehicle info, user preferences, personal context, timezone, units

## Project Structure

### New Files Created

```
app/
├── core/
│   ├── langfuse_client.py          # Shared Langfuse client (avoids circular imports)
│   ├── memory/
│   │   ├── __init__.py             # Module exports
│   │   ├── classifier.py           # LLM-based memory classification
│   │   └── retrieval.py            # Semantic search with Qdrant
│   └── observability/
│       ├── __init__.py             # Observability exports
│       └── memory_spans.py         # Langfuse instrumentation (PII-safe)
│
├── models/
│   ├── user_profile.py             # UserProfile SQLModel (PostgreSQL)
│   └── memory_event.py             # MemoryEvent SQLModel (audit trail)
│
├── services/
│   └── database.py                 # Extended with UserProfile CRUD
│
├── schemas/
│   └── graph.py                    # Enhanced GraphState with user_profile, conversation_summary, turn_count
│
docs/
├── memory_flow.mermaid.md         # Mermaid flowchart diagram
├── memory_flow.puml               # PlantUML sequence diagram
└── langfuse_memory_dashboard.md   # Observability dashboard guide

alembic/
├── env.py                         # Alembic configuration with SQLModel metadata
└── versions/
    └── a1b2c3d4_add_user_profile_and_memory_event.py  # Migration script
```

### Modified Files

1. **`app/core/langgraph/graph.py`**
   - Enhanced `_chat()` method: Memory classification, storage, and retrieval
   - Enhanced `_build_personalized_prompt()`: Injects retrieved memories into system prompt
   - Added `_generate_summary()`: Periodic conversation summarization
   - Integrated Langfuse memory spans

2. **`app/core/config.py`**
   - Added memory feature flags
   - Added Qdrant configuration
   - Added embedding model configuration
   - Added retrieval parameters (top K, min score)

3. **`app/schemas/graph.py`**
   - Extended `GraphState` with:
     - `user_id`: JWT-extracted user identifier
     - `user_profile`: Dictionary for user context
     - `conversation_summary`: Compressed conversation history
     - `turn_count`: Conversation turn counter
     - `last_summary_turn`: Last summary generation point
     - Helper functions: `increment_turn()`, `should_generate_summary()`, `mark_summary_generated()`

## Key Components

### 1. Memory Classifier (`app/core/memory/classifier.py`)

**Purpose**: Detect if user input contains long-term memory information

**Implementation**:
- Uses LLM (ChatOpenAI) to classify messages
- Returns structured classification:
  - `is_memory`: Boolean flag
  - `memory_type`: Category (vehicle_info, preferences, personal_info, context)
  - `extracted_facts`: Structured key-value pairs
- Feature-gated via `FEATURE_MEMORY_CLASSIFIER`

**Example Output**:
```python
{
    "is_memory": True,
    "memory_type": "vehicle_info",
    "extracted_facts": {
        "vin": "1HGBH41JXMN109186",
        "make": "Honda",
        "model": "Civic"
    }
}
```

### 2. Memory Retrieval (`app/core/memory/retrieval.py`)

**Purpose**: Semantic search for relevant long-term memories

**Implementation**:
- Uses OpenAI embeddings (`text-embedding-3-small` by default)
- Qdrant vector database for similarity search
- Configurable parameters:
  - `MEM_TOP_K`: Number of memories to retrieve (default: 3)
  - `MEM_MIN_SCORE`: Minimum similarity threshold (default: 0.55)
- User-scoped filtering (only retrieves memories for current user)
- Feature-gated via `FEATURE_MEMORY_RETRIEVAL`

**Storage Flow**:
1. Classifier extracts facts from user message
2. Facts stored in PostgreSQL `UserProfile` table
3. Facts embedded and upserted to Qdrant
4. During retrieval, query is embedded and searched in Qdrant
5. Top-K results above threshold are returned

### 3. Observability (`app/core/observability/memory_spans.py`)

**Purpose**: PII-safe instrumentation for memory operations

**Events Tracked**:
- `memory.write`: Memory stored to PostgreSQL + Qdrant
- `memory.update`: Memory updated (future use)
- `memory.retrieve`: Semantic search with scores and latency
- `memory.fallback`: Vector store unavailable or errors

**PII Safety**:
- User IDs: SHA-256 hashed, first 8 chars shown (`u_a1b2c3d4`)
- Content: Truncated to 50 chars preview
- Keys: Only dictionary keys stored, not values
- No sensitive data in traces

### 4. Database Models

**UserProfile** (`app/models/user_profile.py`):
- Primary key: `user_id` (string)
- Fields: `name`, `timezone`, `preferred_units`, `vehicle_vin`, `vehicle_type`, `notes` (JSONB)
- `updated_at`: Timestamp with timezone

**MemoryEvent** (`app/models/memory_event.py`):
- UUID primary key
- Tracks: `user_id`, `type`, `text`, `embedding_id`, `score`, `source_message_id`
- Indexed on `user_id` for fast queries

## Data Flow

### Memory Storage Flow

```
User Message
    ↓
Memory Classifier (LLM)
    ↓
├─→ [Long-term fact] → PostgreSQL UserProfile + Qdrant Upsert
├─→ [Summary/content] → Qdrant Upsert only
└─→ [Not memory] → No action
```

### Memory Retrieval Flow

```
User Message → Build Query (last 3 messages)
    ↓
Semantic Search (Qdrant)
    ↓
Score >= MEM_MIN_SCORE?
    ↓
Yes → Inject into System Prompt
    ↓
LLM Response with Context
```

### Prompt Enhancement

The system prompt is dynamically enhanced with:

1. **User Context Section** (from `user_profile`):
   ```
   === USER CONTEXT ===
   Name: John
   Timezone: America/New_York
   Preferred units: imperial
   Vehicle: 2023 Honda Civic
   ```

2. **Retrieved Memory Section** (from semantic search):
   ```
   === RETRIEVED MEMORY ===
   - [vehicle_info] {"vin": "1HGBH...", "make": "Honda"}
   - [preferences] {"units": "imperial", "timezone": "America/New_York"}
   ```

3. **Conversation Summary** (if available):
   ```
   === CONVERSATION SUMMARY ===
   User is planning a road trip. Discussed vehicle maintenance
   and route planning preferences.
   ```

## Environment Configuration

### Required Environment Variables

```bash
# Feature Flags
FEATURE_LONG_TERM_MEMORY=true
FEATURE_MEMORY_CLASSIFIER=true
FEATURE_MEMORY_RETRIEVAL=true

# Qdrant Configuration
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333          # Local: http://localhost:6333
                                          # Cloud: https://your-cluster.qdrant.io
QDRANT_API_KEY=none                       # Local: "none"
                                          # Cloud: <your_api_key>

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-small

# Retrieval Settings
MEM_TOP_K=3                               # Number of memories to retrieve
MEM_MIN_SCORE=0.55                        # Minimum similarity threshold (0.0-1.0)

# JWT Configuration
JWT_USER_ID_FIELD=sub                    # Field in JWT token containing user ID
```

### Qdrant Setup Options

**Local Development**:
```bash
docker run -p 6333:6333 qdrant/qdrant
# QDRANT_API_KEY=none (no authentication)
```

**Production (Qdrant Cloud)**:
```bash
# Get API key from Qdrant Cloud dashboard
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_cloud_api_key
```

## Guardrails & Safety Mechanisms

1. **Similarity Threshold**: Only memories with score >= `MEM_MIN_SCORE` injected
2. **Rate Limiting**: Existing chat endpoint rate limits apply
3. **Token Cap**: Summaries limited to `MAX_TOKENS // 2`
4. **Feature Flags**: Independent toggle for each component:
   - `FEATURE_LONG_TERM_MEMORY`: Master switch
   - `FEATURE_MEMORY_CLASSIFIER`: Classification on/off
   - `FEATURE_MEMORY_RETRIEVAL`: Semantic search on/off
5. **Error Handling**: Memory operations never fail the main request flow
6. **PII Protection**: No sensitive data in observability traces

## Database Migrations

**Alembic Migration**: `a1b2c3d4_add_user_profile_and_memory_event.py`

- Creates `userprofile` table with user_id as primary key
- Creates `memoryevent` table with UUID primary key
- Creates index on `memoryevent.user_id`
- Includes proper `downgrade()` function

**Usage**:
```bash
alembic upgrade head    # Apply migrations
alembic downgrade -1    # Rollback
```

## Integration Points

### GraphState Integration

The `GraphState` class was enhanced to support memory features:

```python
class GraphState(BaseModel):
    messages: list
    session_id: str
    user_id: Optional[str]              # NEW: From JWT
    user_profile: dict                   # NEW: User context
    conversation_summary: str             # NEW: Compressed history
    turn_count: int                      # NEW: Turn counter
    last_summary_turn: int               # NEW: Summary tracking
```

### LangGraph Agent Integration

The `LangGraphAgent._chat()` method now:
1. Increments `turn_count` at start
2. Classifies user message for memory content
3. Stores extracted facts (PostgreSQL + Qdrant)
4. Builds personalized prompt with retrieved memories
5. Generates conversation summary if needed
6. Emits Langfuse spans for observability

## Observability

### Langfuse Events

All memory operations emit Langfuse events:

- **`memory.write`**: Metadata includes user_id (hashed), memory_type, keys_count
- **`memory.retrieve`**: Metadata includes hit_rate, latency_ms, similarity scores
- **`memory.fallback`**: Metadata includes reason for fallback

### Dashboard Queries

See `docs/langfuse_memory_dashboard.md` for:
- Retrieval hit rate analysis
- Similarity score distributions
- Write frequency by memory type
- Latency tracking
- Fallback reason analysis

## Dependencies Added

```toml
# pyproject.toml
qdrant-client>=1.11.0      # Vector database client
openai>=1.62.0             # Embeddings (already used for LLM)
alembic>=1.14.0            # Database migrations
```

## Testing & Validation

1. **Unit Tests**: GraphState helper functions
2. **Integration Tests**: Memory classification and retrieval
3. **E2E Tests**: Full conversation flow with memory persistence
4. **Error Handling**: Verified memory failures don't break requests

## Documentation

- **Flow Diagrams**: `docs/memory_flow.mermaid.md` (GitHub-renderable)
- **Sequence Diagram**: `docs/memory_flow.puml` (PlantUML)
- **Observability Guide**: `docs/langfuse_memory_dashboard.md`
- **README**: Updated with memory setup instructions

## Key Design Decisions

1. **Hybrid Storage**: PostgreSQL for structured queries, Qdrant for semantic search
2. **LLM Classification**: Uses LLM to detect memory-worthy content (flexible, adapts to new patterns)
3. **Dual Storage for Facts**: Facts stored in both systems for redundancy and different query patterns
4. **Feature Flags**: Granular control for gradual rollout and debugging
5. **PII-Safe Observability**: Never log sensitive data, hash user IDs
6. **Non-Blocking**: Memory operations wrapped in try-except, never fail requests
7. **Semantic Search**: Uses cosine similarity for flexible, context-aware retrieval

## Future Enhancements (Not Implemented)

- Memory update/delete operations
- Memory expiration policies
- User memory export
- Multi-tenant memory isolation (currently user-scoped)
- pgvector as alternative to Qdrant

## Summary

This implementation adds a production-ready long-term memory system that:
- ✅ Automatically detects and stores user facts
- ✅ Provides semantic search for relevant context
- ✅ Compresses long conversations with summaries
- ✅ Observes all operations safely (PII-protected)
- ✅ Can be toggled via feature flags
- ✅ Never breaks the main request flow
- ✅ Supports both local and cloud deployments

The system is designed for scalability, safety, and observability while maintaining backward compatibility with the existing LangGraph agent.

