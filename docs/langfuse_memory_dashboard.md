# Langfuse Memory Dashboard Guide

## Overview

This guide explains how to monitor memory operations using Langfuse observability spans. All memory-related events are tracked with PII-safe instrumentation.

## Event Names

The memory system emits the following Langfuse events:

- **`memory.write`** - Memory stored to PostgreSQL and Qdrant
- **`memory.update`** - Existing memory updated
- **`memory.retrieve`** - Semantic search for relevant memories
- **`memory.fallback`** - Memory operation failed or unavailable

## Suggested Dashboard Charts

### 1. Retrieval Hit Rate
**Query:** Filter by `event_name: memory.retrieve`

**Metrics:**
- Hit rate: `matched_count / requested_top_k`
- Average matched count per query
- Requests with zero matches

**Insights:**
- Low hit rates may indicate inadequate memory content
- High hit rates suggest effective memory storage

### 2. Similarity Score Distribution
**Query:** `event_name: memory.retrieve`

**Metrics:**
- Average similarity score (`avg_score`)
- Maximum similarity score (`max_score`)
- Minimum similarity score (`min_score`)

**Insights:**
- Scores below 0.5 indicate weak matches
- Scores above 0.8 suggest high-quality retrieval
- Track score trends over time to assess embedding quality

### 3. Write Frequency
**Query:** `event_name: memory.write`

**Metrics:**
- Writes per hour/day
- Writes by memory type
- Write success rate

**Insights:**
- Monitor write volume to detect memory accumulation
- Different memory types may have different write patterns

### 4. Retrieval Latency
**Query:** `event_name: memory.retrieve`

**Metrics:**
- Average latency (`latency_ms`)
- P50, P95, P99 percentile latencies
- Timeout occurrences

**Insights:**
- High latency may indicate Qdrant performance issues
- Embedding generation typically contributes most to latency

### 5. Fallback Count & Reasons
**Query:** `event_name: memory.fallback`

**Metrics:**
- Fallback frequency
- Most common fallback reasons
- Fallback rate per user

**Insights:**
- `feature_disabled_or_unavailable` - Feature not configured
- `retrieval_error:*` - Qdrant connection or query failures
- High fallback rates may indicate infrastructure issues

## Filtering in Langfuse UI

### By Event Name

In the Langfuse UI:
1. Navigate to **Traces** or **Events**
2. Use filter: `event_name IN ['memory.write', 'memory.retrieve', 'memory.fallback']`
3. Group by `event_name` for aggregate views

### By User ID

**Important:** User IDs are hashed for PII protection.

Format: `u_{first_8_chars_of_hash}`

Example filter:
```
metadata.user_id = u_a1b2c3d4
```

### By Memory Type

Memory types include:
- `vehicle_info` - Vehicle VIN, make, model
- `preferences` - User preferences
- `personal_info` - Personal information
- `context` - Conversation context

Filter example:
```
metadata.memory_type = "vehicle_info"
```

## PII Safety

All instrumentation follows strict PII protection:

1. **User IDs:** Hashed with SHA-256, only first 8 chars shown
2. **Content:** Truncated to 50 chars, preview shown
3. **Keys:** Top-level keys extracted, not values
4. **IDs:** Masked and truncated in previews

No personal data, full content, or sensitive information is stored in Langfuse traces.

## Example Queries

### High-Latency Retrievals
```
event_name = "memory.retrieve" AND metadata.latency_ms > 1000
```

### Low-Quality Matches
```
event_name = "memory.retrieve" AND metadata.avg_score < 0.5
```

### Write Failures
```
event_name = "memory.fallback" AND metadata.reason LIKE "storage_error%"
```

### Daily Memory Activity
```
event_name IN ['memory.write', 'memory.retrieve']
GROUP BY date(created_at)
```

## Troubleshooting

### No Events Appearing
1. Verify feature flags are enabled:
   - `FEATURE_LONG_TERM_MEMORY=true`
   - `FEATURE_MEMORY_RETRIEVAL=true`
2. Check Langfuse configuration
3. Review application logs for instrumentation errors

### Unexpected Fallbacks
1. Check Qdrant connectivity
2. Verify embedding model configuration
3. Review similarity threshold settings

### Poor Retrieval Quality
1. Analyze similarity score distributions
2. Consider adjusting `MEM_MIN_SCORE` threshold
3. Review embedding model performance

## Data Retention

Langfuse event data follows your Langfuse plan's retention policies. For long-term analysis, consider exporting data to your data warehouse.

