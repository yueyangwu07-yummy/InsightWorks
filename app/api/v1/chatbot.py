"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
from functools import lru_cache
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse
from app.core.metrics import llm_stream_duration_seconds
from app.api.v1.auth import get_current_session
from app.core.config import settings
from app.core.langgraph.graph import LangGraphAgent
from app.core.limiter import limiter
from app.core.logging import logger
from app.models.session import Session
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Message,
    StreamResponse,
)

try:
    from cleanlab_codex import Client, Project
except ImportError:
    Client = None
    Project = None

router = APIRouter()
agent = LangGraphAgent()


@lru_cache
def get_cleanlab_project():
    """Initialize and return a singleton Cleanlab Project instance.

    Supports two authentication methods:
    1. User-level API key + Project ID (via Client)
    2. Project-level Access Key (via Project.from_access_key)

    Uses CLEANLAB_CODEX_API_KEY and CLEANLAB_PROJECT_ID from settings.
    If CLEANLAB_CODEX_API_KEY is a project access key, it will be used directly.
    Otherwise, it will be treated as a user-level API key and used with Project ID.

    Logs will appear in Cleanlab Platform only if project_id is provided.

    Returns:
        Optional[Project]: The Cleanlab Project instance if configured, None otherwise.
    """
    if Client is None or Project is None:
        logger.warning("cleanlab-codex is not installed. Validation will be skipped.")
        return None

    logger.info("Initializing Cleanlab Project...")
    
    # Add debug logging to diagnose configuration issues
    logger.debug(f"CLEANLAB_CODEX_API_KEY exists: {bool(settings.CLEANLAB_CODEX_API_KEY)}")
    logger.debug(f"CLEANLAB_CODEX_API_KEY length: {len(settings.CLEANLAB_CODEX_API_KEY) if settings.CLEANLAB_CODEX_API_KEY else 0}")
    logger.debug(f"CLEANLAB_PROJECT_ID exists: {bool(settings.CLEANLAB_PROJECT_ID)}")
    logger.debug(f"CLEANLAB_PROJECT_ID length: {len(settings.CLEANLAB_PROJECT_ID) if settings.CLEANLAB_PROJECT_ID else 0}")

    if not settings.CLEANLAB_CODEX_API_KEY:
        logger.warning("CLEANLAB_CODEX_API_KEY not set. Validation will be skipped.")
        return None

    # Try method 1: Project Access Key (if Project ID is not required)
    # If Project.from_access_key works without Project ID, try it first
    try:
        logger.debug("Attempting to use Project Access Key (Project.from_access_key)...")
        project = Project.from_access_key(settings.CLEANLAB_CODEX_API_KEY)
        logger.info("Cleanlab Project initialized successfully using Access Key.")
        return project
    except Exception as access_key_error:
        logger.debug(f"Project.from_access_key failed (this is expected if using user-level API key): {str(access_key_error)[:100]}")

    # Try method 2: User-level API Key + Project ID
    if not settings.CLEANLAB_PROJECT_ID:
        logger.warning("CLEANLAB_PROJECT_ID not set and Access Key method failed. Logs will not appear in Cleanlab Platform.")
        return None

    try:
        logger.debug(f"Attempting to use User-level API Key with Client...")
        logger.debug(f"Creating Client with API key (first 8 chars): {settings.CLEANLAB_CODEX_API_KEY[:8]}...")
        client = Client(api_key=settings.CLEANLAB_CODEX_API_KEY)
        logger.debug(f"Retrieving project with ID (first 8 chars): {settings.CLEANLAB_PROJECT_ID[:8]}...")
        project = client.get_project(settings.CLEANLAB_PROJECT_ID)
        logger.info(f"Cleanlab Project initialized successfully with project_id: {settings.CLEANLAB_PROJECT_ID[:8]}...")
        return project
    except Exception as e:
        error_msg = str(e)
        if "user level API key" in error_msg.lower() or "Cannot get user info" in error_msg:
            logger.error(
                f"Failed to initialize Cleanlab Project: {error_msg}. "
                "If you have a Project Access Key, ensure CLEANLAB_PROJECT_ID is empty or not set. "
                "If you have a User API Key, ensure it's a user-level key, not a project access key."
            )
        else:
            logger.error(f"Failed to initialize Cleanlab Project: {e}", exc_info=True)
        return None


def clear_cleanlab_project_cache():
    """Clear the cache for get_cleanlab_project to force re-initialization.
    
    This is useful when configuration changes or to retry initialization after a failure.
    """
    get_cleanlab_project.cache_clear()
    logger.info("Cleanlab project cache cleared.")



@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Process a chat request using LangGraph.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        session: The current session from the auth token.

    Returns:
        ChatResponse: The processed chat response.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        logger.info(
            "chat_request_received",
            session_id=session.id,
            message_count=len(chat_request.messages),
        )

        # Initialize Cleanlab project (gracefully fail if not configured)
        project = get_cleanlab_project()

        result = await agent.get_response(
            chat_request.messages, session.id, user_id=session.user_id
        )

        logger.info("chat_request_processed", session_id=session.id)

        # After getting response, validate with Cleanlab
        if project and result:
            # Extract user query from chat_request.messages (last user message)
            user_query = ""
            for msg in reversed(chat_request.messages):
                if msg.role == "user":
                    user_query = msg.content
                    break
            
            # Extract assistant response (last assistant message from result)
            full_response = ""
            for msg in reversed(result):
                if msg.role == "assistant":
                    full_response = msg.content
                    break

            if user_query and full_response:
                logger.info("Validating response with Cleanlab...")
                try:
                    # Convert chat messages to the format expected by validate
                    messages = []
                    for msg in chat_request.messages:
                        messages.append({
                            "role": msg.role,
                            "content": msg.content
                        })
                    
                    # Synchronous API call to validate
                    results = project.validate(
                        query=user_query,
                        response=full_response,
                        context="",  # RAG context can be added later
                        messages=messages
                    )
                    logger.info(f"Cleanlab validation results: {results}")
                except Exception as e:
                    logger.error(f"Error during Cleanlab validation: {e}", exc_info=True)
            else:
                logger.warning("Could not extract user query or assistant response for validation.")
        elif not project:
            logger.warning("Cleanlab project not initialized. Skipping validation.")

        return ChatResponse(messages=result)
    except Exception as e:
        logger.error("chat_request_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_stream"][0])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Process a chat request using LangGraph with streaming response.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        session: The current session from the auth token.

    Returns:
        StreamingResponse: A streaming response of the chat completion.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        logger.info(
            "stream_chat_request_received",
            session_id=session.id,
            message_count=len(chat_request.messages),
        )

        async def event_generator():
            """Generate streaming events.

            Yields:
                str: Server-sent events in JSON format.

            Raises:
                Exception: If there's an error during streaming.
            """
            # Initialize Cleanlab project (gracefully fail if not configured)
            project = get_cleanlab_project()

            # Prepare variables to capture final response and user query
            full_response = ""
            user_query = ""
            rag_context = ""  # RAG context can be added later

            try:
                with llm_stream_duration_seconds.labels(model=agent.llm.model_name).time():
                    async for chunk in agent.get_stream_response(
                        chat_request.messages, session.id, user_id=session.user_id
                     ):
                        full_response += chunk  # Accumulate the full response
                        response = StreamResponse(content=chunk, done=False)
                        yield f"data: {json.dumps(response.model_dump())}\n\n"

                # Send final message indicating completion
                final_response = StreamResponse(content="", done=True)
                yield f"data: {json.dumps(final_response.model_dump())}\n\n"

                # After streaming is complete, validate the response
                if project and full_response:
                    # Extract user query from chat_request.messages (last user message)
                    for msg in reversed(chat_request.messages):
                        if msg.role == "user":
                            user_query = msg.content
                            break

                    if user_query:
                        logger.info("Streaming complete. Validating with Cleanlab...")
                        try:
                            # Convert chat messages to the format expected by validate
                            messages = []
                            for msg in chat_request.messages:
                                messages.append({
                                    "role": msg.role,
                                    "content": msg.content
                                })
                            
                            # Synchronous API call (happens after all yields)
                            # The validate method requires: query, response, context, and messages
                            results = project.validate(
                                query=user_query,
                                response=full_response,
                                context=rag_context,
                                messages=messages
                            )
                            logger.info(f"Cleanlab validation results: {results}")
                        except Exception as e:
                            logger.error(f"Error during Cleanlab validation: {e}", exc_info=True)
                    else:
                        logger.warning("Could not extract user query for validation.")
                elif not full_response:
                    logger.warning("No response generated. Skipping Cleanlab validation.")
                elif not project:
                    logger.warning("Cleanlab project not initialized. Skipping validation.")

            except Exception as e:
                logger.error(
                    "stream_chat_request_failed",
                    session_id=session.id,
                    error=str(e),
                    exc_info=True,
                )
                error_response = StreamResponse(content=str(e), done=True)
                yield f"data: {json.dumps(error_response.model_dump())}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(
            "stream_chat_request_failed",
            session_id=session.id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def get_session_messages(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Get all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        ChatResponse: All messages in the session.

    Raises:
        HTTPException: If there's an error retrieving the messages.
    """
    try:
        messages = await agent.get_chat_history(session.id)
        return ChatResponse(messages=messages)
    except Exception as e:
        logger.error("get_messages_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/messages")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def clear_chat_history(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Clear all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        dict: A message indicating the chat history was cleared.
    """
    try:
        await agent.clear_chat_history(session.id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error("clear_chat_history_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
