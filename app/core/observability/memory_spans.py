"""Langfuse instrumentation for memory operations.

This module provides helper functions to emit Langfuse spans for memory-related
operations, ensuring proper observability without storing PII in traces.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.langfuse_client import get_langfuse_client
from app.core.logging import logger


def _mask_user_id(user_id: Any) -> str:
    """Mask user ID to avoid PII leakage in traces.
    
    Args:
        user_id: User identifier (any type)
        
    Returns:
        Masked user ID string (first 8 chars of hash)
    """
    # Convert to string and hash
    user_str = str(user_id)
    hash_obj = hashlib.sha256(user_str.encode())
    return f"u_{hash_obj.hexdigest()[:8]}"


def _mask_content_preview(content: Any, max_chars: int = 50) -> str:
    """Create a safe preview of content without exposing full PII.
    
    Args:
        content: Content to preview
        max_chars: Maximum characters in preview
        
    Returns:
        Masked preview string
    """
    content_str = str(content)
    if len(content_str) <= max_chars:
        return content_str
    # Show first and last few chars
    preview_len = max_chars // 2
    return f"{content_str[:preview_len]}...{content_str[-preview_len:]}"


def _extract_keys_or_ids(memory_content: Dict[str, Any]) -> List[str]:
    """Extract safe identifiers from memory content.
    
    Args:
        memory_content: Memory content dictionary
        
    Returns:
        List of safe keys or IDs
    """
    keys = []
    if isinstance(memory_content, dict):
        # Extract top-level keys
        keys.extend([f"key:{k}" for k in memory_content.keys()])
        # Extract specific safe identifiers if present
        for key in ["vin", "id", "vehicle_id", "memory_id"]:
            if key in memory_content:
                keys.append(f"id:{_mask_content_preview(memory_content[key])}")
    return keys[:10]  # Limit to 10 keys


def emit_memory_write(
    user_id: Any,
    memory_type: str,
    memory_content: Optional[Dict[str, Any]] = None,
    source_message_id: Optional[str] = None,
) -> None:
    """Emit Langfuse span for memory write operation.
    
    Args:
        user_id: User identifier
        memory_type: Type of memory being written
        memory_content: Memory content (optional, for key extraction)
        source_message_id: Source message ID that triggered this write
    """
    if not settings.FEATURE_LONG_TERM_MEMORY:
        return
        
    try:
        langfuse_client = get_langfuse_client()
        if not langfuse_client:
            return
            
        # Extract safe identifiers
        keys_or_ids = _extract_keys_or_ids(memory_content) if memory_content else []
        
        # Emit event (non-blocking span)
        langfuse_client.event(
            name="memory.write",
            metadata={
                "user_id": _mask_user_id(user_id),
                "memory_type": memory_type,
                "keys_count": len(keys_or_ids),
                "keys_preview": keys_or_ids[:5],  # First 5 keys only
                "source_message_id": source_message_id,
            },
        )
    except Exception as e:
        # Never fail the main flow due to logging issues
        logger.debug(f"Failed to emit memory.write span: {e}")


def emit_memory_update(
    user_id: Any,
    diff: Dict[str, Any],
    source_message_id: Optional[str] = None,
) -> None:
    """Emit Langfuse span for memory update operation.
    
    Args:
        user_id: User identifier
        diff: Dictionary of changes (old -> new)
        source_message_id: Source message ID that triggered this update
    """
    if not settings.FEATURE_LONG_TERM_MEMORY:
        return
        
    try:
        langfuse_client = get_langfuse_client()
        if not langfuse_client:
            return
            
        # Create safe diff preview
        diff_preview = {k: _mask_content_preview(v) for k, v in list(diff.items())[:5]}
        
        langfuse_client.event(
            name="memory.update",
            metadata={
                "user_id": _mask_user_id(user_id),
                "fields_updated": list(diff.keys())[:10],  # Top 10 fields
                "diff_preview": diff_preview,
                "source_message_id": source_message_id,
            },
        )
    except Exception as e:
        logger.debug(f"Failed to emit memory.update span: {e}")


def emit_memory_retrieve(
    user_id: Any,
    matched_ids: List[str],
    scores: List[float],
    top_k: int,
    latency_ms: float,
) -> None:
    """Emit Langfuse span for memory retrieval operation.
    
    Args:
        user_id: User identifier
        matched_ids: List of matched memory IDs
        scores: List of similarity scores
        top_k: Number of memories requested
        latency_ms: Retrieval latency in milliseconds
    """
    if not settings.FEATURE_MEMORY_RETRIEVAL:
        return
        
    try:
        langfuse_client = get_langfuse_client()
        if not langfuse_client:
            return
            
        # Mask matched IDs
        masked_ids = [_mask_content_preview(mid) for mid in matched_ids[:10]]
        
        # Compute statistics
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        
        langfuse_client.event(
            name="memory.retrieve",
            metadata={
                "user_id": _mask_user_id(user_id),
                "matched_count": len(matched_ids),
                "requested_top_k": top_k,
                "hit_rate": len(matched_ids) / top_k if top_k > 0 else 0.0,
                "latency_ms": round(latency_ms, 2),
                "avg_score": round(avg_score, 3),
                "max_score": round(max_score, 3),
                "min_score": round(min_score, 3),
                "matched_ids_preview": masked_ids,
            },
        )
    except Exception as e:
        logger.debug(f"Failed to emit memory.retrieve span: {e}")


def emit_memory_fallback(user_id: Any, reason: str) -> None:
    """Emit Langfuse span for memory fallback operation.
    
    Args:
        user_id: User identifier
        reason: Reason for fallback (e.g., "qdrant_unavailable", "low_score")
    """
    try:
        langfuse_client = get_langfuse_client()
        if not langfuse_client:
            return
            
        langfuse_client.event(
            name="memory.fallback",
            metadata={
                "user_id": _mask_user_id(user_id),
                "reason": reason,
            },
        )
    except Exception as e:
        logger.debug(f"Failed to emit memory.fallback span: {e}")

