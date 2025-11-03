"""Langfuse client singleton for observability."""

from typing import Optional

from app.core.config import settings
from app.core.logging import logger

# Import Langfuse for manual tracking
try:
    from langfuse import Langfuse
    _langfuse_available = True
except ImportError:
    _langfuse_available = False
    Langfuse = None
    logger.warning("langfuse package not available. Tracking will be disabled.")


def get_langfuse_client() -> Optional[Langfuse]:
    """Get or create Langfuse client instance.
    
    Returns:
        Optional[Langfuse]: Langfuse client instance if configured, None otherwise.
    """
    logger.debug(f"[DIAGNOSTIC] get_langfuse_client: _langfuse_available={_langfuse_available}")
    if not _langfuse_available:
        logger.debug("[DIAGNOSTIC] get_langfuse_client: langfuse package not available, returning None")
        return None
    
    logger.debug(f"[DIAGNOSTIC] get_langfuse_client: PUBLIC_KEY exists: {bool(settings.LANGFUSE_PUBLIC_KEY)}, SECRET_KEY exists: {bool(settings.LANGFUSE_SECRET_KEY)}, HOST: {settings.LANGFUSE_HOST}")
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.debug("[DIAGNOSTIC] get_langfuse_client: Missing credentials, returning None")
        return None
    
    try:
        client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        logger.debug(f"[DIAGNOSTIC] get_langfuse_client: Successfully created Langfuse client: {type(client)}")
        return client
    except Exception as e:
        logger.warning(f"Failed to create Langfuse client: {e}", exc_info=True)
        return None

