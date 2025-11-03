# Memory Flow Diagram

This diagram illustrates the long-term memory flow from user message to LLM response.

```mermaid
flowchart TD
    A((User Message)) --> B{Memory Classifier}
    
    B -->|Long-term fact| C[Write to PostgreSQL UserProfile]
    B -->|Summary / high-value content| D[Embed → Vector DB Qdrant]
    B -->|Not a memory| E[No Action]
    
    subgraph VectorStore["Vector Store (Qdrant)"]
        D --> F[Upsert Embeddings]
        F --> G[(Stored Vectors)]
    end
    
    C --> H[(PostgreSQL)]
    
    A --> I[Semantic Retrieval]
    I --> J{score >= threshold?}
    
    J -->|Yes| K[Retrieve Top K Memories]
    J -->|No| L[No Memories Retrieved]
    
    K --> M[Inject into System Prompt]
    L --> M
    
    M --> N[LLM Response with Context]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#e8f5e9
    style D fill:#e8f5e9
    style H fill:#e8f5e9
    style G fill:#f3e5f5
    style M fill:#fff9c4
    style N fill:#e1f5ff
```

## Components

- **Memory Classifier**: LLM-based classification to detect memory-worthy content
- **PostgreSQL UserProfile**: Stores stable facts (VIN, preferences, etc.)
- **Qdrant Vector Store**: Stores embeddings for semantic search
- **Semantic Retrieval**: Top-K search with similarity threshold filtering
- **System Prompt Injection**: Relevant memories added to LLM context

## Decision Points

1. **Classification**: Determines if message contains long-term memory
2. **Threshold Check**: Only memories above `MEM_MIN_SCORE` are injected
3. **Storage Path**: Different storage for facts vs. summaries
