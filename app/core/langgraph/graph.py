"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

import os
from datetime import datetime
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Literal,
    Optional,
)
from urllib.parse import quote_plus

from asgiref.sync import sync_to_async
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import (
    END,
    StateGraph,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from openai import OpenAIError
from psycopg_pool import AsyncConnectionPool

from app.core.config import (
    Environment,
    settings,
)
from app.core.langgraph.tools import tools
from app.core.logging import logger
from app.core.memory import memory_classifier, memory_retrieval
from app.core.metrics import llm_inference_duration_seconds
from app.core.observability import emit_memory_write
from app.core.prompts import SYSTEM_PROMPT

# Debug log file path (in project root)
# Calculate project root: app/core/langgraph/graph.py -> project root
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_file_dir)))
DEBUG_LOG_FILE = os.path.join(_project_root, "debug_messages.log")


def _write_debug_log(content: str) -> None:
    """Write debug message to a separate log file.
    
    Args:
        content: The debug message content to write.
    """
    try:
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n[{timestamp}] {content}\n")
    except Exception as e:
        # Fallback to logger if file write fails
        logger.debug(f"Failed to write debug log: {e}")

# Import Langfuse client
from app.core.langfuse_client import get_langfuse_client
from app.schemas import (
    GraphState,
    Message,
)
from app.schemas.graph import (
    increment_turn,
    mark_summary_generated,
    should_generate_summary,
)
from app.utils import (
    dump_messages,
    prepare_messages,
)


class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Use environment-specific LLM model
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            api_key=settings.LLM_API_KEY,
            max_tokens=settings.MAX_TOKENS,
            **self._get_model_kwargs(),
        ).bind_tools(tools)
        
        # Store model name for Langfuse tracking (Langfuse 3.x compatibility)
        self.model_name = settings.LLM_MODEL
        
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        # Dictionary to store active Langfuse root spans by session_id
        # Root span contains trace_context needed for child spans/generations
        self._active_root_spans: Dict[str, Any] = {}

        logger.info("llm_initialized", model=settings.LLM_MODEL, environment=settings.ENVIRONMENT.value)

    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get environment-specific model kwargs.

        Returns:
            Dict[str, Any]: Additional model arguments based on environment
        """
        model_kwargs = {}

        # Development - we can use lower speeds for cost savings
        if settings.ENVIRONMENT == Environment.DEVELOPMENT:
            model_kwargs["top_p"] = 0.8

        # Production - use higher quality settings
        elif settings.ENVIRONMENT == Environment.PRODUCTION:
            model_kwargs["top_p"] = 0.95
            model_kwargs["presence_penalty"] = 0.1
            model_kwargs["frequency_penalty"] = 0.1

        return model_kwargs

    def _sanitize_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize tool arguments to remove sensitive information for logging.
        
        Args:
            args: Tool arguments dictionary
            
        Returns:
            Sanitized arguments dictionary safe for logging
        """
        if not isinstance(args, dict):
            return args
        
        sanitized = {}
        sensitive_keys = [
            "api_key", "apikey", "token", "password", "secret", "key",
            "authorization", "auth", "credentials", "access_token", "refresh_token"
        ]
        
        for key, value in args.items():
            key_lower = key.lower()
            # Check if key contains sensitive patterns
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 500:
                # Truncate very long strings
                sanitized[key] = value[:200] + "...[truncated]"
            elif isinstance(value, (dict, list)):
                # Recursively sanitize nested structures (limited depth)
                sanitized[key] = self._sanitize_args(value) if isinstance(value, dict) else "[complex object]"
            else:
                sanitized[key] = value
        
        return sanitized

    def _get_trace_context(self, state: GraphState, config: Optional[RunnableConfig] = None) -> Optional[Any]:
        """Get trace_context for the current session.
        
        This method retrieves trace_context to associate child spans/generations with the root span.
        Tries multiple sources and approaches in order:
        1. Instance variable (_active_root_spans) - extract trace_context from root span
        2. Config metadata - get trace_context directly or from root_span
        3. State metadata - cached trace_context or root_span
        4. Fallback: construct TraceContext manually if trace_id available
        
        Args:
            state: Current graph state
            config: Optional runtime configuration
            
        Returns:
            Langfuse trace_context object, root_span object, or None if unavailable.
            Can be used directly in client.start_span() or client.start_generation().
        """
        langfuse_client = get_langfuse_client()
        if not langfuse_client:
            return None
        
        # Try 1: Instance variable - get trace_context from root span
        root_span = self._active_root_spans.get(state.session_id)
        if root_span:
            try:
                # Approach A: Check for trace_context attribute (preferred)
                if hasattr(root_span, 'trace_context'):
                    trace_context = root_span.trace_context
                    logger.debug(f"[Langfuse] Got trace_context from instance variable root span, session_id: {state.session_id}")
                    return trace_context
                
                # Approach B: Try to construct TraceContext manually if trace_id is available
                if hasattr(root_span, 'trace_id') or hasattr(root_span, 'id'):
                    try:
                        from langfuse.types import TraceContext
                        trace_id = getattr(root_span, 'trace_id', None) or getattr(root_span, 'id', None)
                        observation_id = getattr(root_span, 'observation_id', None) or getattr(root_span, 'id', None)
                        
                        if trace_id:
                            manual_ctx = TraceContext(
                                trace_id=trace_id,
                                observation_id=observation_id,
                            )
                            logger.debug(f"[Langfuse] Constructed TraceContext manually from root span, session_id: {state.session_id}")
                            return manual_ctx
                    except (ImportError, Exception) as e:
                        logger.debug(f"[Langfuse] Could not construct TraceContext manually: {e}")
                
                # Approach C: Return root span object itself (some APIs might accept it)
                logger.debug(f"[Langfuse] Using root span object as trace_context fallback, session_id: {state.session_id}")
                return root_span
                
            except Exception as e:
                logger.warning(f"[Langfuse] Failed to get trace_context from root span: {e}", exc_info=True)
        
        # Try 2: Config metadata (from LangGraph config passed to nodes)
        if config:
            config_metadata = None
            if isinstance(config, dict):
                # Check both metadata and configurable keys
                config_metadata = config.get("metadata") or config.get("configurable", {})
            elif hasattr(config, 'metadata'):
                config_metadata = getattr(config, 'metadata', None)
            
            if config_metadata:
                # Check for trace_context directly
                if isinstance(config_metadata, dict):
                    trace_context = config_metadata.get("trace_context")
                    if trace_context:
                        logger.debug(f"[Langfuse] Got trace_context from config metadata, session_id: {state.session_id}")
                        return trace_context
                    
                    # Check for root_span and extract trace_context
                    root_span = config_metadata.get("root_span")
                    if root_span:
                        try:
                            if hasattr(root_span, 'trace_context'):
                                trace_context = root_span.trace_context
                                logger.debug(f"[Langfuse] Got trace_context from config root_span, session_id: {state.session_id}")
                                return trace_context
                            # Fallback: return root span itself
                            logger.debug(f"[Langfuse] Using root_span from config as trace_context, session_id: {state.session_id}")
                            return root_span
                        except Exception as e:
                            logger.warning(f"[Langfuse] Failed to get trace_context from config root_span: {e}")
        
        # Try 3: State metadata (cached values)
        if hasattr(state, '_lf_trace_context') and state._lf_trace_context:
            logger.debug(f"[Langfuse] Got trace_context from state metadata, session_id: {state.session_id}")
            return state._lf_trace_context
        
        if hasattr(state, '_lf_root_span') and state._lf_root_span:
            try:
                root_span = state._lf_root_span
                if hasattr(root_span, 'trace_context'):
                    trace_context = root_span.trace_context
                    # Cache for future use
                    if not hasattr(state, '_lf_trace_context'):
                        state._lf_trace_context = trace_context
                    logger.debug(f"[Langfuse] Got trace_context from state root_span, session_id: {state.session_id}")
                    return trace_context
                # Fallback: return root span itself
                return root_span
            except Exception as e:
                logger.warning(f"[Langfuse] Failed to get trace_context from state root_span: {e}")
        
        # No trace_context found - log warning but don't fail (main flow continues)
        logger.debug(f"[Langfuse] No trace_context found for session_id: {state.session_id} - child spans will not be linked")
        return None

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool: A connection pool for PostgreSQL database.
        """
        if self._connection_pool is None:
            try:
                # Configure pool size based on environment
                max_size = settings.POSTGRES_POOL_SIZE

                connection_url = (
                    "postgresql://"
                    f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
                )

                self._connection_pool = AsyncConnectionPool(
                    connection_url,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
            except Exception as e:
                logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we might want to degrade gracefully
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
                    return None
                raise e
        return self._connection_pool

    async def _chat(self, state: GraphState, config: Optional[RunnableConfig] = None) -> dict:
        """Process the chat state and generate a response.

        Args:
            state (GraphState): The current state of the conversation.
            config (Optional[RunnableConfig]): Runtime configuration for the runnable.

        Returns:
            dict: Updated state with new messages.
        """
        # Increment turn counter at the start of each chat interaction
        state = increment_turn(state)
        logger.info(
            "chat_turn",
            session_id=state.session_id,
            turn_count=state.turn_count,
            user_name=state.user_profile.get("name", "Unknown") if state.user_profile else "Unknown",
        )

        # ========== DIAGNOSTIC LOGGING - LLM Configuration ==========
        import os
        api_key_exists = bool(os.getenv('OPENAI_API_KEY'))
        api_key_preview = os.getenv('OPENAI_API_KEY', '')[:10] + '...' if os.getenv('OPENAI_API_KEY') else 'NOT SET'
        
        logger.info(f"[Chat-Debug] ========== LLM Configuration Check ==========")
        logger.info(f"[Chat-Debug] OpenAI API Key exists: {api_key_exists}")
        logger.info(f"[Chat-Debug] API Key preview: {api_key_preview}")
        logger.info(f"[Chat-Debug] Model name: {self.model_name}")
        logger.info(f"[Chat-Debug] LLM object type: {type(self.llm)}")
        logger.info(f"[Chat-Debug] LLM object: {self.llm}")
        logger.info(f"[Chat-Debug] LLM has ainvoke: {hasattr(self.llm, 'ainvoke')}")
        logger.info(f"[Chat-Debug] Temperature: {getattr(self, 'temperature', 'N/A')}")
        logger.info(f"[Chat-Debug] Max tokens: {getattr(self, 'max_tokens', 'N/A')}")
        logger.info(f"[Chat-Debug] ==========================================")

        # Get trace_context using unified method
        trace_context = self._get_trace_context(state, config)
        langfuse_client = get_langfuse_client()

        # Build personalized system prompt based on user profile and conversation summary
        # Pass trace_context instead of trace to _build_personalized_prompt
        try:
            personalized_prompt = await self._build_personalized_prompt(state, SYSTEM_PROMPT, trace_context, langfuse_client)
            messages = prepare_messages(state.messages, self.llm, personalized_prompt)
            logger.info(f"[Chat-Debug] Personalized prompt built successfully, length: {len(personalized_prompt)}")
        except Exception as prompt_error:
            logger.error(f"[Chat-Debug] Failed to build personalized prompt: {prompt_error}", exc_info=True)
            # Fallback to base prompt if personalization fails
            messages = prepare_messages(state.messages, self.llm, SYSTEM_PROMPT)
            logger.warning(f"[Chat-Debug] Using base SYSTEM_PROMPT as fallback")

        llm_calls_num = 0

        # Configure retry attempts based on environment
        max_retries = settings.MAX_LLM_CALL_RETRIES

        # Prepare messages for Langfuse tracking
        langfuse_messages = dump_messages(messages)

        logger.info(f"[Langfuse] _chat: trace_context available: {trace_context is not None}, session_id: {state.session_id}, turn: {state.turn_count}")

        # ========== DIAGNOSTIC LOGGING - Messages Preparation ==========
        logger.info(f"[Chat-Debug] ========== Messages Preparation ==========")
        logger.info(f"[Chat-Debug] Total messages count: {len(messages)}")
        logger.info(f"[Chat-Debug] Session ID: {state.session_id}")
        logger.info(f"[Chat-Debug] Turn count: {state.turn_count}")
        
        # Detailed message inspection
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            has_content = hasattr(msg, 'content')
            content_length = len(msg.content) if has_content and msg.content else 0
            
            logger.debug(f"[Chat-Debug] Message {i}: type={msg_type}, has_content={has_content}, content_length={content_length}")
            
            # Check message format
            if not has_content:
                logger.error(f"[Chat-Debug] ❌ Invalid message {i}: Missing 'content' attribute: {msg}")
            
            # Detailed type-specific logging
            if hasattr(msg, 'type'):
                msg_type_attr = getattr(msg, 'type', 'unknown')
                logger.debug(f"[Chat-Debug]   Message {i} type attribute: {msg_type_attr}")
                
                if msg_type_attr == 'human':
                    content_preview = msg.content[:100] + '...' if len(msg.content) > 100 else msg.content
                    logger.debug(f"[Chat-Debug]   HumanMessage: {content_preview}")
                elif msg_type_attr == 'ai':
                    content_preview = msg.content[:100] + '...' if len(msg.content) > 100 else msg.content
                    logger.debug(f"[Chat-Debug]   AIMessage: {content_preview}")
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        logger.debug(f"[Chat-Debug]     - Has {len(msg.tool_calls)} tool calls")
                elif msg_type_attr == 'system':
                    content_preview = msg.content[:100] + '...' if len(msg.content) > 100 else msg.content
                    logger.debug(f"[Chat-Debug]   SystemMessage: {content_preview}")
                elif msg_type_attr == 'tool':
                    content_preview = msg.content[:100] + '...' if len(msg.content) > 100 else msg.content
                    logger.debug(f"[Chat-Debug]   ToolMessage: {content_preview}")
        
        logger.info(f"[Chat-Debug] ==========================================")

        # === DEBUG: Messages to be sent to LLM ===
        debug_lines = [
            "\n" + "=" * 80,
            "🚀 _chat() - Messages prepared to send to LLM:",
            f"Total message count: {len(messages)}",
            f"Session ID: {state.session_id}",
            f"Turn count: {state.turn_count}",
            "=" * 80
        ]
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            debug_lines.append(f"  [{i}] {msg_type}")
            
            if hasattr(msg, 'role'):
                debug_lines.append(f"      role={msg.role}")
            
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                debug_lines.append(f"      tool_calls={len(msg.tool_calls)} calls")
                for j, tc in enumerate(msg.tool_calls[:3]):  # First 3 tool calls
                    tc_id = tc.get('id', 'N/A') if isinstance(tc, dict) else getattr(tc, 'id', 'N/A')
                    tc_name = tc.get('name', 'N/A') if isinstance(tc, dict) else getattr(tc, 'name', 'N/A')
                    debug_lines.append(f"        [{j}] id={tc_id}, name={tc_name}")
                if len(msg.tool_calls) > 3:
                    debug_lines.append(f"        ... and {len(msg.tool_calls) - 3} more tool calls")
            
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                debug_lines.append(f"      tool_call_id={msg.tool_call_id}")
            
            if hasattr(msg, 'name') and msg.name:
                debug_lines.append(f"      name={msg.name}")
            
            if hasattr(msg, 'content'):
                content_str = str(msg.content) if msg.content else ""
                debug_lines.append(f"      content_length={len(content_str)}")
                if content_str:
                    preview = content_str[:100] + "..." if len(content_str) > 100 else content_str
                    debug_lines.append(f"      content_preview={preview}")
        
        debug_lines.append("=" * 80)
        _write_debug_log("\n".join(debug_lines))

        for attempt in range(max_retries):
            logger.info(f"[Chat-Debug] ========== Attempt {attempt + 1}/{max_retries} ==========")
            # Determine if this is likely llm_decide (first call) or llm_answer (after tools)
            # Check if there are ToolMessages in the state - if yes, this is likely the answer call
            has_tool_messages = any(
                isinstance(msg, ToolMessage) or 
                (hasattr(msg, 'type') and getattr(msg, 'type', None) == 'tool')
                for msg in state.messages
            )
            span_name = "llm_answer" if has_tool_messages else "llm_decide"
            
            # Create LLM generation before the call to capture accurate timing
            # Use client.start_generation() with trace_context - this is the correct Langfuse 3.x API
            # IMPORTANT: Langfuse errors should NEVER block LLM calls
            llm_generation = None
            if langfuse_client and trace_context:
                try:
                    logger.debug(f"[Chat-Debug] Creating Langfuse generation for {span_name}...")
                    llm_generation = langfuse_client.start_generation(
                        trace_context=trace_context,  # Key: pass trace_context to link to parent span
                        name=span_name,
                        model=settings.LLM_MODEL,
                        model_parameters={
                            "temperature": settings.DEFAULT_LLM_TEMPERATURE,
                            "max_tokens": settings.MAX_TOKENS,
                            **self._get_model_kwargs(),
                        },
                        input=langfuse_messages,
                        metadata={
                            "session_id": state.session_id,
                            "turn_count": state.turn_count,
                            "attempt": attempt + 1,
                            "has_tool_messages": has_tool_messages,
                        },
                    )
                    logger.info(f"[Langfuse] ✅ Created LLM generation: {span_name}, session_id: {state.session_id}, attempt: {attempt + 1}")
                except Exception as gen_error:
                    # NEVER fail LLM call due to Langfuse errors - this is critical!
                    logger.warning(f"[Langfuse] ❌ Failed to create generation for LLM call: {gen_error}", exc_info=True)
                    logger.warning(f"[Chat-Debug] Continuing without Langfuse tracing...")
                    llm_generation = None
            else:
                if not langfuse_client:
                    logger.debug(f"[Langfuse] Langfuse client not available for LLM generation, session_id: {state.session_id}")
                if not trace_context:
                    logger.debug(f"[Langfuse] trace_context not available for LLM generation, session_id: {state.session_id}")
            
            try:
                # ========== DIAGNOSTIC LOGGING - LLM Invocation ==========
                logger.info(f"[Chat-Debug] Invoking LLM: {self.model_name}")
                logger.debug(f"[Chat-Debug] LLM config: temp={settings.DEFAULT_LLM_TEMPERATURE}, max_tokens={settings.MAX_TOKENS}")
                logger.debug(f"[Chat-Debug] Messages prepared: {len(langfuse_messages)} items")
                
                # Add timeout protection
                import asyncio
                timeout_seconds = 60.0  # 60 second timeout
                
                try:
                    # === DEBUG: Log messages being sent to LLM ===
                    logger.info("=" * 80)
                    logger.info("SENDING TO LLM:")
                    logger.info(f"Total messages: {len(messages)}")
                    for i, msg in enumerate(messages):
                        msg_type = type(msg).__name__
                        role = getattr(msg, 'role', 'N/A')
                        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
                        tool_call_count = len(msg.tool_calls) if has_tool_calls else 0
                        content_preview = str(getattr(msg, 'content', ''))[:100]
                        
                        logger.info(f"  [{i}] {msg_type} | role={role} | tool_calls={tool_call_count} | content={content_preview}")
                        
                        if has_tool_calls:
                            for tc in msg.tool_calls:
                                logger.info(f"      - tool_call_id: {tc.get('id', 'N/A')} | name: {tc.get('name', 'N/A')}")
                    logger.info("=" * 80)
                    
                    with llm_inference_duration_seconds.labels(model=self.llm.model_name).time():
                        logger.info(f"[Chat-Debug] Starting LLM.ainvoke() with {timeout_seconds}s timeout...")
                        # Use asyncio.wait_for to add timeout protection
                        response = await asyncio.wait_for(
                            self.llm.ainvoke(langfuse_messages, config=config),
                            timeout=timeout_seconds
                        )
                        logger.info(f"[Chat-Debug] ✅ LLM response received successfully")
                        
                        # === DEBUG: Log LLM response ===
                        logger.info("=" * 80)
                        logger.info("LLM RESPONSE:")
                        logger.info(f"Response type: {type(response).__name__}")
                        has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
                        logger.info(f"Has tool_calls: {has_tool_calls}")
                        if has_tool_calls:
                            logger.info(f"Number of tool_calls: {len(response.tool_calls)}")
                            for tc in response.tool_calls:
                                logger.info(f"  - tool_call_id: {tc.get('id', 'N/A')} | name: {tc.get('name', 'N/A')}")
                        content_preview = str(getattr(response, 'content', ''))[:200]
                        logger.info(f"Response content preview: {content_preview}")
                        logger.info("=" * 80)
                except asyncio.TimeoutError:
                    logger.error(f"[Chat-Debug] ❌ LLM call timed out after {timeout_seconds}s")
                    raise Exception(f"LLM call timed out after {timeout_seconds} seconds")
                except Exception as llm_error:
                    logger.error(f"[Chat-Debug] ❌ LLM.ainvoke() failed: {type(llm_error).__name__}: {llm_error}")
                    logger.error(f"[Chat-Debug] Error details:", exc_info=True)
                    
                    # Log exception attributes if available
                    if hasattr(llm_error, '__dict__'):
                        logger.error(f"[Chat-Debug] Exception attributes: {llm_error.__dict__}")
                    
                    # Re-raise to be caught by outer exception handler
                    raise
                
                # If we get here, LLM call was successful
                generated_state = {"messages": [response]}
                logger.debug(f"[Chat-Debug] Response type: {type(response)}")
                logger.debug(f"[Chat-Debug] Response has content: {hasattr(response, 'content')}")
                
                # Verify if this actually produced tool_calls (for output metadata)
                has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls

                # Get response content
                response_content = response.content if hasattr(response, 'content') else str(response)
                
                # End LLM generation with appropriate output
                # FIX: Langfuse end() does NOT accept 'output' parameter - use update() first, then end()
                if llm_generation:
                    try:
                        # Prepare output and metadata
                        output_data = {"content": response_content}
                        metadata_data = {}
                        
                        if span_name == "llm_decide" and has_tool_calls:
                            # For llm_decide: indicate which tools were requested
                            tool_names = [tc.get("name", "unknown") for tc in response.tool_calls]
                            metadata_data = {
                                "tool_calls_requested": len(response.tool_calls),
                                "tools": tool_names,
                            }
                        elif span_name == "llm_answer":
                            # For llm_answer: include excerpt
                            excerpt = response_content[:120] + ("..." if len(response_content) > 120 else "")
                            metadata_data = {"excerpt": excerpt}
                        else:
                            # Edge case: llm_decide but no tool calls
                            metadata_data = {"note": "No tool calls requested"}
                        
                        # FIX: Use update() then end() instead of end(output=...)
                        llm_generation.update(
                            output=output_data,
                            metadata=metadata_data,
                        )
                        llm_generation.end()  # No parameters!
                        logger.debug(f"[Langfuse] ✅ Ended generation: {span_name}")
                    except Exception as gen_end_error:
                        # Never fail due to Langfuse generation end errors
                        logger.warning(f"[Langfuse] ❌ Failed to end generation for LLM call: {gen_end_error}", exc_info=True)
                        # Try to at least call end() without parameters
                        try:
                            llm_generation.end()
                        except:
                            pass

                logger.info(
                    "llm_response_generated",
                    session_id=state.session_id,
                    llm_calls_num=llm_calls_num + 1,
                    model=settings.LLM_MODEL,
                    environment=settings.ENVIRONMENT.value,
                    span_name=span_name,
                    has_tool_calls=has_tool_calls,
                )
                llm_calls_num += 1

                # Check if we should generate a conversation summary
                if should_generate_summary(state, threshold=10):
                    try:
                        summary = await self._generate_summary(state)
                        state.conversation_summary = summary
                        state = mark_summary_generated(state)

                        # Include summary updates in returned state
                        generated_state["conversation_summary"] = state.conversation_summary
                        generated_state["last_summary_turn"] = state.last_summary_turn
                        generated_state["turn_count"] = state.turn_count

                        logger.info(
                            "conversation_summary_generated",
                            session_id=state.session_id,
                            turn_count=state.turn_count,
                            summary_length=len(summary),
                        )

                        # Update Langfuse root span metadata with summary information
                        # Note: We can't update trace_context directly, so we update via root span if available
                        root_span = self._active_root_spans.get(state.session_id)
                        if root_span:
                            try:
                                root_span.update(
                                    metadata={
                                        "summary_generated": True,
                                        "summary_turn": state.turn_count,
                                    }
                                )
                            except Exception as trace_error:
                                logger.debug(f"Failed to update trace metadata: {trace_error}")
                    except Exception as summary_error:
                        logger.warning(
                            "summary_generation_failed",
                            error=str(summary_error),
                            session_id=state.session_id,
                            exc_info=True,
                        )
                        # Don't fail the request if summary generation fails

                # Classify and store long-term memory if enabled
                if settings.FEATURE_LONG_TERM_MEMORY and state.user_id:
                    try:
                        # Get the last user message
                        last_user_msg = None
                        for msg in reversed(state.messages):
                            if hasattr(msg, 'type') and msg.type == 'human':
                                last_user_msg = msg.content if hasattr(msg, 'content') else str(msg)
                                break
                        
                        if last_user_msg:
                            # Classify the message
                            classification = await memory_classifier.classify(last_user_msg)
                            
                            if classification.get("is_memory", False):
                                import json
                                
                                memory_type = classification.get("memory_type", "unknown")
                                memory_content = classification.get("extracted_facts", {})
                                
                                # Create Langfuse span for memory write
                                memory_span = None
                                if langfuse_client and trace_context:
                                    try:
                                        memory_span = langfuse_client.start_span(
                                            trace_context=trace_context,
                                            name="memory_write",
                                            metadata={
                                                "memory_type": memory_type,
                                                "content": memory_content,
                                                "session_id": state.session_id,
                                                "user_id": state.user_id,
                                            },
                                        )
                                    except Exception as span_error:
                                        logger.warning(f"Failed to create memory_write span: {span_error}")
                                
                                try:
                                    # Note: langfuse_client and trace_context are already available in this scope
                                    # Store in PostgreSQL
                                    from app.services.database import database_service
                                    await database_service.create_user_profile(
                                        user_id=int(state.user_id),
                                        memory_type=memory_type,
                                        memory_content=json.dumps(memory_content),
                                    )
                                    
                                    # Store in Qdrant
                                    await memory_retrieval.store_memory(
                                        user_id=int(state.user_id),
                                        memory_type=memory_type,
                                        memory_content=memory_content,
                                    )
                                    
                                    logger.info(
                                        "memory_stored",
                                        user_id=state.user_id,
                                        memory_type=memory_type,
                                        session_id=state.session_id,
                                    )
                                    
                                    # End memory span with success
                                    # FIX: Use update() then end() instead of end(output=...)
                                    if memory_span:
                                        try:
                                            memory_span.update(output={"status": "success"})
                                            memory_span.end()  # No parameters!
                                        except Exception as span_end_error:
                                            logger.warning(f"Failed to end memory_write span: {span_end_error}")
                                    
                                    # Emit Langfuse span for memory write (existing helper)
                                    emit_memory_write(
                                        user_id=state.user_id,
                                        memory_type=memory_type,
                                        memory_content=memory_content,
                                        source_message_id=state.session_id,
                                    )
                                except Exception as memory_storage_error:
                                    # End memory span with error
                                    # FIX: Use update() then end() instead of end(output=...)
                                    if memory_span:
                                        try:
                                            memory_span.update(output={"status": "error", "error": str(memory_storage_error)})
                                            memory_span.end()  # No parameters!
                                        except Exception as span_end_error:
                                            logger.warning(f"Failed to end memory_write span on error: {span_end_error}")
                                    raise
                    except Exception as memory_error:
                        logger.warning(
                            "memory_storage_failed",
                            error=str(memory_error),
                            session_id=state.session_id,
                            exc_info=True,
                        )
                        # Don't fail the request if memory storage fails

                # Include turn_count in returned state to persist the increment
                generated_state["turn_count"] = state.turn_count
                
                # === DEBUG: Log _chat() return value ===
                logger.info("=" * 80)
                logger.info("💬 EXITING _chat() method")
                logger.info(f"Returning generated_state with:")
                logger.info(f"  - messages count: {len(generated_state.get('messages', []))}")
                logger.info(f"  - turn_count: {generated_state.get('turn_count', 'N/A')}")
                for i, msg in enumerate(generated_state.get('messages', [])):
                    msg_type = type(msg).__name__
                    has_tc = hasattr(msg, 'tool_calls') and msg.tool_calls
                    logger.info(f"    [{i}] {msg_type} | role={getattr(msg, 'role', 'N/A')} | has_tool_calls={has_tc}")
                logger.info("=" * 80)

                return generated_state
            except (OpenAIError, Exception) as e:
                # End LLM generation with error if it was created
                error_class = type(e).__name__
                error_message = str(e)
                
                logger.error(f"[Chat-Debug] ❌ Attempt {attempt + 1} failed: {error_class}: {error_message}")
                logger.error(f"[Chat-Debug] Error details:", exc_info=True)
                
                # === DEBUG: Log full messages on error ===
                if "tool_call_id" in error_message.lower():
                    logger.error("=" * 80)
                    logger.error("OPENAI ERROR - FULL MESSAGE HISTORY:")
                    logger.error(f"Total messages: {len(messages)}")
                    for i, msg in enumerate(messages):
                        msg_dict = {
                            "index": i,
                            "type": type(msg).__name__,
                            "role": getattr(msg, 'role', 'N/A'),
                            "has_tool_calls": hasattr(msg, 'tool_calls') and bool(msg.tool_calls),
                            "tool_call_count": len(msg.tool_calls) if hasattr(msg, 'tool_calls') and msg.tool_calls else 0,
                            "tool_call_id": getattr(msg, 'tool_call_id', 'N/A'),
                            "name": getattr(msg, 'name', 'N/A'),
                            "content_length": len(str(getattr(msg, 'content', ''))),
                        }
                        logger.error(f"  {msg_dict}")
                        
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                logger.error(f"      tool_call: id={tc.get('id')} name={tc.get('name')}")
                    
                    logger.error("=" * 80)
                
                # Log exception attributes if available
                if hasattr(e, '__dict__'):
                    logger.error(f"[Chat-Debug] Exception attributes: {e.__dict__}")
                
                if llm_generation:
                    try:
                        # FIX: Use update() then end() instead of end(output=...)
                        llm_generation.update(
                            output=None,
                            metadata={
                                "success": False,
                                "error": f"{error_class}: {error_message}",
                                "attempt": attempt + 1,
                            },
                        )
                        llm_generation.end()  # No parameters!
                        logger.debug(f"[Langfuse] Ended generation with error: {error_class}")
                    except Exception as gen_end_error:
                        # Never fail due to Langfuse errors
                        logger.warning(f"[Langfuse] ❌ Failed to end generation with error: {gen_end_error}")
                        # Try to at least call end() without parameters
                        try:
                            llm_generation.end()
                        except:
                            pass

                logger.error(
                    "llm_call_failed",
                    llm_calls_num=llm_calls_num,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                    error_class=error_class,
                    environment=settings.ENVIRONMENT.value,
                )
                llm_calls_num += 1

                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.info(f"[Chat-Debug] Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    
                    # In production, we might want to fall back to a more reliable model
                    if settings.ENVIRONMENT == Environment.PRODUCTION and attempt == max_retries - 2:
                        fallback_model = "gpt-4o"
                        logger.warning(
                            "using_fallback_model", model=fallback_model, environment=settings.ENVIRONMENT.value
                        )
                        self.llm.model_name = fallback_model
                else:
                    # Last attempt failed
                    logger.error(f"[Chat-Debug] ❌ All {max_retries} attempts failed")
                    raise Exception(f"Failed to get a response from the LLM after {max_retries} attempts: {error_class}: {error_message}")

        # Final error tracking - if we get here, all retries failed
        # This should not be reached due to the raise in the except block above, but keep as safety net
        raise Exception(f"Failed to get a response from the LLM after {max_retries} attempts")

    async def _build_personalized_prompt(
        self,
        state: GraphState,
        base_prompt: str,
        trace_context: Optional[Any] = None,
        langfuse_client: Optional[Any] = None,
    ) -> str:
        """Build personalized system prompt based on user profile and retrieved memories.

        Args:
            state: Current graph state
            base_prompt: Base system prompt from SYSTEM_PROMPT
            trace_context: Optional Langfuse trace_context for memory retrieval tracking
            langfuse_client: Optional Langfuse client for creating spans

        Returns:
            Enhanced system prompt with user context and retrieved memories
        """
        enhanced_prompt = base_prompt

        # Add user context if available
        if state.user_profile:
            enhanced_prompt += "\n\n=== USER CONTEXT ==="

            if name := state.user_profile.get("name"):
                enhanced_prompt += f"\nUser's name: {name}"

            if language := state.user_profile.get("language"):
                enhanced_prompt += f"\nPreferred language: {language}"

            if timezone := state.user_profile.get("timezone"):
                enhanced_prompt += f"\nUser's timezone: {timezone}"

            if vehicle_type := state.user_profile.get("vehicle_type"):
                enhanced_prompt += f"\nUser's vehicle: {vehicle_type}"

            if preferred_units := state.user_profile.get("preferred_units"):
                enhanced_prompt += f"\nPreferred units: {preferred_units}"

        # Add retrieved memories if available
        if settings.FEATURE_MEMORY_RETRIEVAL and state.user_id:
            # Build query from recent messages
            query_text = ""
            for msg in state.messages[-3:]:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    if msg.type == 'human':
                        query_text += f" {msg.content}"
            
            # Create Langfuse span for memory retrieval
            retrieval_span = None
            if langfuse_client and trace_context and query_text.strip():
                try:
                    retrieval_span = langfuse_client.start_span(
                        trace_context=trace_context,
                        name="memory_retrieval",
                        metadata={
                            "session_id": state.session_id,
                            "user_id": state.user_id,
                            "query": query_text.strip()[:200],  # Truncate long queries
                        },
                    )
                except Exception as span_error:
                    logger.warning(f"Failed to create memory_retrieval span: {span_error}")
            
            try:
                if query_text.strip():
                    memories = await memory_retrieval.retrieve_memories(
                        user_id=int(state.user_id),
                        query_text=query_text.strip(),
                    )
                    
                    if memories:
                        enhanced_prompt += "\n\n=== RETRIEVED MEMORY ==="
                        for mem in memories:
                            enhanced_prompt += f"\n- [{mem.get('memory_type', 'unknown')}] {mem.get('memory_content', {})}"
                        
                        logger.info(
                            "memories_injected",
                            user_id=state.user_id,
                            count=len(memories),
                        )
                    
                    # End retrieval span with success
                    if retrieval_span:
                        try:
                            retrieval_span.end(
                                output={
                                    "count": len(memories) if memories else 0,
                                    "memories": [
                                        {
                                            "memory_type": mem.get('memory_type'),
                                            "score": mem.get('score'),
                                        }
                                        for mem in (memories or [])[:5]  # Limit to top 5 for output
                                    ]
                                }
                            )
                        except Exception as span_end_error:
                            logger.warning(f"Failed to end memory_retrieval span: {span_end_error}")
            except Exception as e:
                # End retrieval span with error
                # FIX: Use update() then end() instead of end(output=...)
                if retrieval_span:
                    try:
                        retrieval_span.update(output={"error": str(e)}, metadata={"success": False})
                        retrieval_span.end()  # No parameters!
                    except Exception as span_end_error:
                        logger.warning(f"Failed to end memory_retrieval span on error: {span_end_error}")
                logger.warning("memory_injection_failed", error=str(e))

        # Add conversation summary if available
        if state.conversation_summary:
            enhanced_prompt += f"\n\n=== CONVERSATION HISTORY SUMMARY ===\n{state.conversation_summary}"
            enhanced_prompt += f"\n(Summary up to turn {state.last_summary_turn}, current turn: {state.turn_count})"

        return enhanced_prompt

    async def _generate_summary(self, state: GraphState) -> str:
        """Generate a summary of the conversation so far.

        Args:
            state: Current graph state

        Returns:
            Conversation summary string
        """
        # Format recent messages for summarization
        recent_messages = state.messages[-20:]  # Last 20 messages
        formatted_messages = []

        for msg in recent_messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = msg.type if msg.type in ['human', 'ai', 'system'] else 'unknown'
                content = msg.content[:200]  # Truncate long messages
                formatted_messages.append(f"{role}: {content}")

        conversation_text = "\n".join(formatted_messages)

        summary_prompt = f"""Summarize the following conversation in 2-3 sentences. 
Focus on:
- What the user is trying to accomplish
- Key information exchanged
- Current context or state

Conversation:

{conversation_text}

Summary:"""

        # Use a simple LLM call (no tools) for summary
        # Create a temporary LLM instance without tools for summary generation
        summary_llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            api_key=settings.LLM_API_KEY,
            max_tokens=settings.MAX_TOKENS // 2,  # Use fewer tokens for summary
            **self._get_model_kwargs(),
        )

        response = await summary_llm.ainvoke([HumanMessage(content=summary_prompt)])

        return response.content if hasattr(response, 'content') else str(response)

    # Define our tool node
    async def _tool_call(self, state: GraphState, config: Optional[RunnableConfig] = None) -> GraphState:
        """Process tool calls from the last message.
        
        CRITICAL: Must return a ToolMessage for EVERY tool_call in the last message,
        otherwise OpenAI API will reject the request.
        
        Args:
            state: The current agent state containing messages and tool calls.
            config (Optional[RunnableConfig]): Runtime configuration for the runnable.

        Returns:
            Dict with updated messages containing tool responses.
        """
        # === DEBUG: Log tool call entry ===
        logger.info("=" * 80)
        logger.info("🔧 ENTERED _tool_call() method")
        logger.info(f"State messages count BEFORE: {len(state.messages)}")
        logger.info(f"Total messages in state: {len(state.messages)}")
        
        # Get the last message and all tool calls
        messages = state.messages
        last_message = messages[-1]
        
        logger.info(f"Last message type: {type(last_message).__name__}")
        logger.info(f"Last message role: {getattr(last_message, 'role', 'N/A')}")
        
        has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
        logger.info(f"Last message has tool_calls: {has_tool_calls}")
        
        if has_tool_calls:
            logger.info(f"Number of tool_calls to execute: {len(last_message.tool_calls)}")
            for i, tc in enumerate(last_message.tool_calls):
                logger.info(f"  [{i}] tool_call_id: {tc.get('id', 'N/A')} | name: {tc.get('name', 'N/A')} | args: {tc.get('args', {})}")
        else:
            logger.warning("_tool_call() was called but last message has no tool_calls!")
        
        logger.info("=" * 80)
        
        # Get trace_context using unified method
        trace_context = self._get_trace_context(state, config)
        langfuse_client = get_langfuse_client()
        
        # CRITICAL: Check if there are tool calls
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.warning(f"[ToolCall] _tool_call called but no tool_calls found in last message, session_id: {state.session_id}")
            return {"messages": []}
        
        tool_calls = last_message.tool_calls
        
        # ========== EXTENSIVE LOGGING FOR DEBUGGING ==========
        logger.info(f"[ToolCall] ========== TOOL_CALL DEBUG ==========")
        logger.info(f"[ToolCall] Session ID: {state.session_id}")
        logger.info(f"[ToolCall] Number of tool_calls: {len(tool_calls)}")
        logger.info(f"[ToolCall] Tool call IDs: {[tc.get('id', 'NO_ID') for tc in tool_calls]}")
        logger.info(f"[ToolCall] Tool names: {[tc.get('name', 'NO_NAME') for tc in tool_calls]}")
        logger.info(f"[ToolCall] =====================================")
        
        logger.debug(f"[Langfuse] Tool call - trace_context available: {trace_context is not None}, session_id: {state.session_id}")
        
        # Temporary debug probe to verify trace_context attachment (optional, can be removed later)
        if langfuse_client and trace_context:
            try:
                probe_span = langfuse_client.start_span(
                    trace_context=trace_context,
                    name="tool_probe",
                    metadata={
                        "session_id": state.session_id,
                        "turn": state.turn_count,
                    },
                )
                # FIX: Use update() then end() instead of end(output=...)
                probe_span.update(output={"status": "probe_success"})
                probe_span.end()
                logger.debug(f"[Langfuse] Tool probe span created for session: {state.session_id}, turn: {state.turn_count}")
            except Exception as probe_error:
                logger.debug(f"[Langfuse] Tool probe span failed (non-critical): {probe_error}")
        
        # CRITICAL: Initialize list to collect ALL tool messages
        tool_messages = []
        
        # Process EVERY tool call - NEVER skip any
        # CRITICAL: Each tool_call MUST get a ToolMessage response, even if tool fails
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", "unknown")
            
            logger.info(f"[ToolCall] Processing tool: {tool_name}, call_id: {tool_call_id}, session_id: {state.session_id}")
            
            # Create Langfuse span for tool execution with format: tool:<tool_name>
            # Use client.start_span() with trace_context - this is the correct Langfuse 3.x API
            tool_span = None
            if langfuse_client and trace_context:
                try:
                    # Sanitize args for safe logging (remove sensitive data)
                    safe_args = self._sanitize_args(tool_args) if isinstance(tool_args, dict) else tool_args
                    
                    tool_span = langfuse_client.start_span(
                        trace_context=trace_context,  # Key: pass trace_context to link to parent
                        name=f"tool:{tool_name}",
                        input=tool_args,  # Include input for observability
                        metadata={
                            "tool_name": tool_name,
                            "args": safe_args,  # Sanitized args in metadata
                            "session_id": state.session_id,
                            "turn_count": state.turn_count,
                            "tool_call_id": tool_call_id,
                        },
                    )
                    logger.info(f"[Langfuse] Created tool span: {tool_name}, session_id: {state.session_id}")
                except Exception as span_error:
                    # Never fail tool execution due to Langfuse errors
                    logger.warning(f"[Langfuse] Failed to create tool span for {tool_name}: {span_error}", exc_info=True)
                    tool_span = None
            else:
                if not langfuse_client:
                    logger.debug(f"[Langfuse] Langfuse client not available for tool span")
                if not trace_context:
                    logger.debug(f"[Langfuse] trace_context not available for tool span, session_id: {state.session_id}")
            
            tool_success = False
            preview = ""
            tool_result = None
            tool_error = None
            
            try:
                # CRITICAL: Check if tool exists
                if tool_name not in self.tools_by_name:
                    error_msg = f"Tool '{tool_name}' not found. Available tools: {list(self.tools_by_name.keys())}"
                    logger.error(f"[ToolCall] {error_msg}")
                    tool_error = ValueError(error_msg)
                    tool_success = False
                    
                    # CRITICAL: Return error ToolMessage, don't skip
                    tool_messages.append(
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        )
                    )
                    
                    # End Langfuse span with error
                    if tool_span:
                        try:
                            tool_span.update(
                                output={"success": False, "error": error_msg},
                            )
                            tool_span.end()
                        except Exception as span_end_error:
                            logger.warning(f"[Langfuse] Failed to end tool span: {span_end_error}")
                    
                    # Continue to next tool call (don't re-raise, we already created error ToolMessage)
                    continue
                
                tool = self.tools_by_name[tool_name]
                logger.info(f"[ToolCall] Executing tool: {tool_name}, args: {tool_args}")
                
                # Execute the tool with multiple fallback methods
                try:
                    # Try async first (preferred)
                    if hasattr(tool, 'ainvoke'):
                        tool_result = await tool.ainvoke(tool_args, config=config)
                    elif hasattr(tool, '_arun'):
                        # Some tools use _arun with keyword args
                        if isinstance(tool_args, dict):
                            tool_result = await tool._arun(**tool_args)
                        else:
                            tool_result = await tool._arun(tool_args)
                    # Fallback to sync methods
                    elif hasattr(tool, 'invoke'):
                        tool_result = await sync_to_async(tool.invoke)(tool_args)
                    elif hasattr(tool, '_run'):
                        if isinstance(tool_args, dict):
                            tool_result = await sync_to_async(tool._run)(**tool_args)
                        else:
                            tool_result = await sync_to_async(tool._run)(tool_args)
                    else:
                        raise AttributeError(f"Tool {tool_name} has no invoke/run/ainvoke/_arun method")
                    
                    # Ensure result is a string
                    if not isinstance(tool_result, str):
                        tool_result = str(tool_result)
                    
                    tool_success = True
                    
                    # Create safe preview of result (first ~120 chars for observability)
                    preview = tool_result[:120] + ("..." if len(tool_result) > 120 else "")
                    
                    logger.info(f"[ToolCall] ✅ Tool {tool_name} succeeded, result length: {len(tool_result)}")
                    
                    # CRITICAL: Create ToolMessage for success
                    tool_messages.append(
                        ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        )
                    )
                
                except Exception as tool_exec_error:
                    error_msg = f"Error executing tool {tool_name}: {str(tool_exec_error)}"
                    logger.error(f"[ToolCall] ❌ {error_msg}", exc_info=True)
                    tool_error = tool_exec_error
                    tool_success = False
                    
                    # CRITICAL: Create ToolMessage for error (don't re-raise, return error message)
                    tool_messages.append(
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        )
                    )
            
            except Exception as outer_error:
                # Catch-all for any unexpected errors in tool processing
                error_msg = f"Unexpected error processing tool call {tool_call_id}: {str(outer_error)}"
                logger.error(f"[ToolCall] ❌ {error_msg}", exc_info=True)
                tool_error = outer_error
                tool_success = False
                
                # CRITICAL: Still return ToolMessage even for unexpected errors
                tool_messages.append(
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call_id,
                        name=tool_name if 'tool_name' in locals() else "unknown",
                    )
                )
            finally:
                # Always end span in finally block to ensure it's closed
                # This ensures spans are properly closed even if tool execution fails
                if tool_span:
                    try:
                        if tool_success:
                            # Tool executed successfully
                            output_data = {
                                "success": True,
                                "preview": preview,
                                "result_length": len(str(tool_result)) if tool_result else 0,
                            }
                            # FIX: Use update() then end() instead of end(output=...)
                            tool_span.update(output=output_data)
                            tool_span.end()  # No parameters!
                            logger.info(f"[Langfuse] ✅ Ended tool span: {tool_name} (success), session_id: {state.session_id}")
                        else:
                            # Tool execution failed
                            error_class = type(tool_error).__name__ if tool_error else "UnknownError"
                            error_message = str(tool_error) if tool_error else "Unknown error"
                            # FIX: Use update() then end() instead of end(output=...)
                            tool_span.update(
                                output={
                                    "success": False,
                                    "error": f"{error_class}: {error_message}",
                                }
                            )
                            tool_span.end()  # No parameters!
                            logger.info(f"[Langfuse] ✅ Ended tool span: {tool_name} (error: {error_class}), session_id: {state.session_id}")
                    except Exception as span_end_error:
                        # Never fail tool execution due to Langfuse span end errors
                        logger.warning(f"[Langfuse] ❌ Failed to end tool span for {tool_name}: {span_end_error}", exc_info=True)
                        # Try to at least call end() without parameters
                        try:
                            tool_span.end()
                        except:
                            pass
        
        # CRITICAL: Verify we have responses for ALL tool calls
        logger.info(f"[ToolCall] ========== TOOL_CALL VERIFICATION ==========")
        logger.info(f"[ToolCall] Tool calls requested: {len(tool_calls)}")
        logger.info(f"[ToolCall] Tool messages created: {len(tool_messages)}")
        
        if len(tool_messages) != len(tool_calls):
            logger.error(
                f"[ToolCall] ❌ MISMATCH: {len(tool_calls)} tool_calls but {len(tool_messages)} tool_messages"
            )
            
            # Find missing tool_call_ids
            returned_ids = {msg.tool_call_id for msg in tool_messages}
            requested_ids = {tc.get("id", "unknown") for tc in tool_calls}
            missing_ids = requested_ids - returned_ids
            
            logger.error(f"[ToolCall] Missing tool_call_ids: {missing_ids}")
            
            # Add error messages for missing tool calls
            for missing_id in missing_ids:
                logger.error(f"[ToolCall] Adding error message for missing tool_call_id: {missing_id}")
                
                # Find the tool call for this ID
                missing_tool_call = next((tc for tc in tool_calls if tc.get("id") == missing_id), None)
                missing_tool_name = missing_tool_call.get("name", "unknown") if missing_tool_call else "unknown"
                
                tool_messages.append(
                    ToolMessage(
                        content="Error: Tool execution was skipped or failed to generate response",
                        tool_call_id=missing_id,
                        name=missing_tool_name,
                    )
                )
        
        # Final verification
        final_returned_ids = {msg.tool_call_id for msg in tool_messages}
        final_requested_ids = {tc.get("id", "unknown") for tc in tool_calls}
        
        if final_returned_ids != final_requested_ids:
            logger.error(f"[ToolCall] ❌ FINAL MISMATCH:")
            logger.error(f"[ToolCall] Requested IDs: {final_requested_ids}")
            logger.error(f"[ToolCall] Returned IDs: {final_returned_ids}")
            logger.error(f"[ToolCall] Still missing: {final_requested_ids - final_returned_ids}")
        else:
            logger.info(f"[ToolCall] ✅ All tool_call_ids have responses!")
        
        logger.info(f"[ToolCall] Returning {len(tool_messages)} tool messages")
        logger.info(f"[ToolCall] Tool message IDs: {[msg.tool_call_id for msg in tool_messages]}")
        logger.info(f"[ToolCall] ==========================================")
        
        # === DEBUG: Log tool call exit ===
        logger.info("=" * 80)
        logger.info("🔧 EXITING _tool_call() method")
        logger.info(f"Tool messages to return: {len(tool_messages)}")
        
        for i, tm in enumerate(tool_messages):
            logger.info(f"  [{i}] tool_call_id={tm.tool_call_id} name={tm.name} content_length={len(tm.content)}")
        
        # Verify all tool_call_ids have responses
        if has_tool_calls:
            requested_ids = {tc.get('id') for tc in last_message.tool_calls}
            returned_ids = {tm.tool_call_id for tm in tool_messages}
            missing_ids = requested_ids - returned_ids
            
            if missing_ids:
                logger.error(f"❌ MISSING TOOL RESPONSES FOR: {missing_ids}")
            else:
                logger.info("✓ All tool_call_ids have responses")
        
        logger.info(f"Returning dict: {{'messages': {len(tool_messages)} ToolMessages}}")
        logger.info("=" * 80)
        
        # === DEBUG: ToolMessages to be returned ===
        debug_lines = [
            "\n" + "=" * 80,
            "🔧 _tool_call() - ToolMessages to be returned:",
            f"ToolMessage count: {len(tool_messages)}",
            "=" * 80
        ]
        for i, msg in enumerate(tool_messages):
            debug_lines.append(f"  [{i}] {type(msg).__name__}")
            debug_lines.append(f"      tool_call_id={msg.tool_call_id}")
            debug_lines.append(f"      name={msg.name}")
            debug_lines.append(f"      content_length={len(msg.content) if msg.content else 0}")
            if msg.content:
                preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                debug_lines.append(f"      content_preview={preview}")
        debug_lines.append("=" * 80)
        _write_debug_log("\n".join(debug_lines))
        
        # CRITICAL: Return format must be {"messages": [ToolMessage, ...]}
        # LangGraph will use add_messages reducer to append these to state.messages
        return {"messages": tool_messages}

    def _should_continue(self, state: GraphState) -> Literal["end", "continue"]:
        """Determine if the agent should continue or end based on unanswered tool calls.

        Logic:
        - Find the last AIMessage with tool_calls (scanning backwards)
        - Check if all its tool_call_ids have corresponding ToolMessage responses
        - If any tool_call_id lacks a response → continue to tool_call node
        - If all tool_call_ids have responses → end

        Args:
            state: The current agent state containing messages.

        Returns:
            Literal["end", "continue"]: "continue" if there are unanswered tool calls, "end" otherwise.
        """
        messages = state.messages
        
        # === DEBUG: Log decision process ===
        debug_lines = [
            "\n" + "=" * 80,
            "🔀 _should_continue() - Checking for tool calls:",
            f"Total messages in state: {len(messages)}",
            "=" * 80
        ]
        
        # Find the last AIMessage with tool_calls (scanning backwards)
        ai_message_with_tool_calls = None
        ai_message_index = -1
        
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            
            # Check if it's an AIMessage with tool_calls
            # Handle both BaseMessage objects and dict-like objects
            has_tool_calls = False
            if isinstance(msg, AIMessage):
                has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
            elif hasattr(msg, 'tool_calls'):
                has_tool_calls = bool(msg.tool_calls)
            elif isinstance(msg, dict):
                has_tool_calls = msg.get("role") == "assistant" and bool(msg.get("tool_calls"))
            
            if has_tool_calls:
                ai_message_with_tool_calls = msg
                ai_message_index = i
                debug_lines.append(f"\nFound AIMessage with tool_calls at index [{i}]")
                break
        
        if ai_message_with_tool_calls is None:
            debug_lines.append("\nNo AIMessage with tool_calls found")
            debug_lines.append("→ Decision: END (no unanswered tool_calls)")
            debug_lines.append("=" * 80)
            _write_debug_log("\n".join(debug_lines))
            return "end"
        
        # Extract all tool_call_ids from the AIMessage
        tool_call_ids = set()
        
        if isinstance(ai_message_with_tool_calls, AIMessage):
            tool_calls = ai_message_with_tool_calls.tool_calls
        elif hasattr(ai_message_with_tool_calls, 'tool_calls'):
            tool_calls = ai_message_with_tool_calls.tool_calls
        elif isinstance(ai_message_with_tool_calls, dict):
            tool_calls = ai_message_with_tool_calls.get("tool_calls", [])
        else:
            tool_calls = []
        
        for tc in tool_calls:
            if isinstance(tc, dict) and 'id' in tc:
                tool_call_ids.add(tc['id'])
            elif hasattr(tc, 'id'):
                tool_call_ids.add(tc.id)
        
        debug_lines.append(f"Tool calls: {tool_call_ids}")
        debug_lines.append(f"Total tool_calls: {len(tool_calls)}")
        
        # Check if there are corresponding ToolMessages after this AIMessage
        answered_ids = set()
        
        debug_lines.append(f"\nChecking messages after index [{ai_message_index}]:")
        for j in range(ai_message_index + 1, len(messages)):
            next_msg = messages[j]
            msg_type = type(next_msg).__name__
            
            # Extract tool_call_id from different message formats
            tool_call_id = None
            if isinstance(next_msg, ToolMessage):
                tool_call_id = next_msg.tool_call_id if hasattr(next_msg, 'tool_call_id') else None
            elif hasattr(next_msg, 'tool_call_id'):
                tool_call_id = next_msg.tool_call_id
            elif isinstance(next_msg, dict):
                if next_msg.get("role") == "tool":
                    tool_call_id = next_msg.get("tool_call_id")
            
            if tool_call_id:
                if tool_call_id in tool_call_ids:
                    answered_ids.add(tool_call_id)
                    debug_lines.append(f"  [{j}] {msg_type} - Found response for tool_call_id: {tool_call_id}")
                else:
                    debug_lines.append(f"  [{j}] {msg_type} - tool_call_id: {tool_call_id} (not in current AIMessage)")
            else:
                debug_lines.append(f"  [{j}] {msg_type} - No tool_call_id")
        
        # Check if there are unanswered tool_call_ids
        unanswered = tool_call_ids - answered_ids
        
        debug_lines.append(f"\nAnswered tool_call_ids: {answered_ids}")
        
        if unanswered:
            debug_lines.append(f"✅ Unanswered tool_calls found: {unanswered}")
            debug_lines.append(f"→ Decision: CONTINUE to tool_call node")
            debug_lines.append("=" * 80)
            _write_debug_log("\n".join(debug_lines))
            return "continue"
        else:
            debug_lines.append(f"ℹ️ All tool_calls have been answered")
            debug_lines.append("→ Decision: END (no unanswered tool_calls)")
            debug_lines.append("=" * 80)
            _write_debug_log("\n".join(debug_lines))
            return "end"

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        """Create and configure the LangGraph workflow.

        Returns:
            Optional[CompiledStateGraph]: The configured LangGraph instance or None if init fails
        """
        if self._graph is None:
            try:
                logger.info("=" * 80)
                logger.info("CREATING LANGGRAPH WORKFLOW")
                logger.info("=" * 80)
                
                graph_builder = StateGraph(GraphState)
                logger.info("✓ Created StateGraph with GraphState")
                
                graph_builder.add_node("chat", self._chat)
                logger.info("✓ Added node: chat")
                
                graph_builder.add_node("tool_call", self._tool_call)
                logger.info("✓ Added node: tool_call")
                
                graph_builder.add_conditional_edges(
                    "chat",
                    self._should_continue,
                    {"continue": "tool_call", "end": END},
                )
                logger.info("✓ Added conditional edge: chat -> {continue: tool_call, end: END}")
                
                graph_builder.add_edge("tool_call", "chat")
                logger.info("✓ Added edge: tool_call -> chat (loop back to chat)")
                
                graph_builder.set_entry_point("chat")
                logger.info("✓ Set entry point: chat")
                
                # CRITICAL: Do NOT set finish_point on chat if tool calls need to loop
                # graph_builder.set_finish_point("chat")  # REMOVED - this was preventing tool_call -> chat loop

                # Get connection pool (may be None in production if DB unavailable)
                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool)
                    await checkpointer.setup()
                    logger.info("✓ Checkpointer configured (PostgreSQL)")
                else:
                    # In production, proceed without checkpointer if needed
                    checkpointer = None
                    logger.warning("⚠ Checkpointer not available (continuing without persistence)")
                    if settings.ENVIRONMENT != Environment.PRODUCTION:
                        raise Exception("Connection pool initialization failed")

                self._graph = graph_builder.compile(
                    checkpointer=checkpointer, name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value})"
                )
                logger.info("✓ Graph compiled successfully")
                logger.info("=" * 80)

                logger.info(
                    "graph_created",
                    graph_name=f"{settings.PROJECT_NAME} Agent",
                    environment=settings.ENVIRONMENT.value,
                    has_checkpointer=checkpointer is not None,
                )
            except Exception as e:
                logger.error("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we don't want to crash the app
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_graph")
                    return None
                raise e

        return self._graph

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Get a response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for Langfuse tracking.
            user_id (Optional[str]): The user ID for Langfuse tracking.

        Returns:
            list[dict]: The response from the LLM.
        """
        # === DEBUG: Frontend incoming messages ===
        debug_lines = [
            "\n" + "=" * 80,
            "🔵 get_response() - Starting request processing",
            f"session_id: {session_id}",
            "=" * 80,
            "📥 Frontend incoming messages:"
        ]
        for i, msg in enumerate(messages):
            line = f"  [{i}] role={msg.role}, content_preview={msg.content[:50] if msg.content else 'N/A'}..."
            debug_lines.append(line)
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                debug_lines.append(f"      🔧 tool_call_id={msg.tool_call_id}, name={msg.name}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                debug_lines.append(f"      🔨 tool_calls count={len(msg.tool_calls)}")
        debug_lines.append("=" * 80)
        _write_debug_log("\n".join(debug_lines))

        if self._graph is None:
            self._graph = await self.create_graph()

        # Create Langfuse root span for this conversation
        # Use start_span() - it automatically creates a trace
        root_span = None
        trace_context = None
        langfuse_client = get_langfuse_client()
        
        # Always attempt to create Langfuse tracing, but never fail the main flow
        if langfuse_client:
            try:
                logger.info(f"[Langfuse] Creating root span for session: {session_id}")
                # Use start_span() to create root span (automatically creates trace)
                root_span = langfuse_client.start_span(
                    name="Chat Conversation",
                    input={"messages": len(messages)},
                    metadata={
                        "environment": settings.ENVIRONMENT.value,
                        "session_id": session_id,
                        "user_id": user_id,
                        "model": settings.LLM_MODEL,
                    },
                )
                
                # Extract trace_context from root span using multiple approaches
                # Approach 1: Check for trace_context attribute (preferred)
                if hasattr(root_span, 'trace_context'):
                    trace_context = root_span.trace_context
                    logger.info(f"[Langfuse] Got trace_context from root span attribute, type: {type(trace_context).__name__}")
                # Approach 2: Try to construct TraceContext manually
                elif hasattr(root_span, 'trace_id') or hasattr(root_span, 'id'):
                    try:
                        from langfuse.types import TraceContext
                        trace_id = getattr(root_span, 'trace_id', None) or getattr(root_span, 'id', None)
                        observation_id = getattr(root_span, 'observation_id', None) or getattr(root_span, 'id', None)
                        
                        if trace_id:
                            trace_context = TraceContext(
                                trace_id=trace_id,
                                observation_id=observation_id,
                            )
                            logger.info(f"[Langfuse] Constructed TraceContext manually, trace_id: {trace_id}")
                    except (ImportError, AttributeError, Exception) as ctx_error:
                        logger.debug(f"[Langfuse] Could not construct TraceContext manually: {ctx_error}")
                        # Approach 3: Use root span object itself as fallback
                        trace_context = root_span
                        logger.info(f"[Langfuse] Using root span object as trace_context fallback")
                else:
                    # Approach 4: Use root span object directly (some APIs might accept it)
                    trace_context = root_span
                    logger.info(f"[Langfuse] Using root span object directly as trace_context")
                
                logger.info(f"[Langfuse] Root span created successfully, session_id: {session_id}")
            except Exception as span_error:
                # Never fail the main flow due to Langfuse errors
                logger.warning(f"[Langfuse] Failed to create root span: {span_error}", exc_info=True)
                root_span = None
                trace_context = None
        else:
            logger.debug(f"[Langfuse] Langfuse client not available - tracing disabled")

        config = {
            "configurable": {"thread_id": session_id},
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": False,
                "root_span": root_span,  # Pass root span for potential direct access
                "trace_context": trace_context,  # Pass trace_context for child spans/generations
            },
        }

        # Store root span in instance variable for _chat/_tool_call to access
        if root_span:
            self._active_root_spans[session_id] = root_span
            logger.debug(f"[DIAGNOSTIC] Stored root span in instance variable, session_id: {session_id}")
        else:
            logger.debug(f"[DIAGNOSTIC] root_span is None, NOT storing in instance variable, session_id: {session_id}")

        try:
            # === Validate and clean message sequence ===
            debug_lines = [
                "\n" + "=" * 80,
                "🔍 Validating message sequence for incomplete tool calls...",
                f"Original message count: {len(messages)}",
                "=" * 80
            ]
            
            # Convert to dict format for validation
            message_dicts = [msg.model_dump() for msg in messages]
            
            cleaned_messages = []
            i = 0
            removed_count = 0
            
            while i < len(message_dicts):
                msg_dict = message_dicts[i]
                
                # Check if this is an assistant message with tool_calls
                if msg_dict.get("role") == "assistant" and msg_dict.get("tool_calls"):
                    # Get all tool_call_ids
                    expected_ids = {
                        tc.get("id") for tc in msg_dict.get("tool_calls", [])
                        if isinstance(tc, dict) and tc.get("id")
                    }
                    
                    debug_lines.append(f"\nFound assistant message at index [{i}] with tool_calls:")
                    debug_lines.append(f"  Tool call IDs: {expected_ids}")
                    
                    # Check if there are corresponding tool messages after this
                    found_ids = set()
                    j = i + 1
                    temp_tool_messages = []
                    
                    while j < len(message_dicts):
                        next_msg_dict = message_dicts[j]
                        next_role = next_msg_dict.get("role")
                        
                        if next_role == "tool":
                            tool_call_id = next_msg_dict.get("tool_call_id")
                            if tool_call_id and tool_call_id in expected_ids:
                                found_ids.add(tool_call_id)
                                temp_tool_messages.append(next_msg_dict)
                                debug_lines.append(f"  [{j}] Found ToolMessage for tool_call_id: {tool_call_id}")
                                j += 1
                            else:
                                # Tool message for different tool_call_id or no tool_call_id
                                break
                        else:
                            # Non-tool message, stop checking
                            break
                    
                    # Check if all tool_calls have responses
                    missing_ids = expected_ids - found_ids
                    
                    if missing_ids:
                        # Incomplete tool calls, remove this assistant message and its tool messages
                        debug_lines.append(f"  ⚠️ Unanswered tool_call_ids: {missing_ids}")
                        debug_lines.append(f"  Removing incomplete message pair(s)")
                        removed_count += 1
                        i = j  # Skip these messages
                        continue
                    else:
                        # Complete message pair, keep them
                        cleaned_messages.append(msg_dict)
                        cleaned_messages.extend(temp_tool_messages)
                        debug_lines.append(f"  ✅ All tool_calls have responses")
                        i = j
                        continue
                
                # Regular message, keep it
                cleaned_messages.append(msg_dict)
                i += 1
            
            if removed_count > 0:
                debug_lines.append(f"\n⚠️ Removed {removed_count} incomplete message pair(s)")
                debug_lines.append(f"Original: {len(messages)} messages → Cleaned: {len(cleaned_messages)} messages")
                
                # Convert back to Message objects
                messages = []
                for msg_dict in cleaned_messages:
                    # Map role properly (handle "human" -> "user", "ai" -> "assistant")
                    role = msg_dict.get("role", "user")
                    if role == "human":
                        role = "user"
                    elif role == "ai":
                        role = "assistant"
                    
                    # Create Message object
                    message_data = {
                        "role": role,
                        "content": msg_dict.get("content", "")
                    }
                    
                    # Add optional fields
                    if msg_dict.get("tool_call_id"):
                        message_data["tool_call_id"] = msg_dict["tool_call_id"]
                    if msg_dict.get("name"):
                        message_data["name"] = msg_dict["name"]
                    if msg_dict.get("tool_calls"):
                        message_data["tool_calls"] = msg_dict["tool_calls"]
                    
                    try:
                        messages.append(Message(**message_data))
                    except Exception as msg_error:
                        logger.warning(f"Failed to create Message from dict: {msg_error}, dict: {msg_dict}")
            else:
                debug_lines.append("\n✅ Message sequence validation passed - no incomplete tool calls found")
            
            debug_lines.append("=" * 80)
            _write_debug_log("\n".join(debug_lines))
            
            # Log before invoking graph
            _write_debug_log(f"\n🔄 Calling _graph.ainvoke() to process messages...")
            
            response = await self._graph.ainvoke(
                {"messages": dump_messages(messages), "session_id": session_id}, config
            )
            
            # Log after graph invocation, before processing
            _write_debug_log(f"\n✅ _graph.ainvoke() completed, starting to process returned messages (total: {len(response.get('messages', []))})")
            
            result = self.__process_messages(response["messages"])
            
            # Log final result
            _write_debug_log(f"\n✅ get_response() - Processing completed, returning {len(result)} messages")
            _write_debug_log("=" * 80 + "\n")

            # Update root span output after workflow completes
            if root_span:
                try:
                    output_summary = {
                        "messages_count": len(result) if isinstance(result, list) else 1,
                        "success": True,
                    }
                    root_span.update(
                        output=output_summary,
                        metadata={"success": True, "session_id": session_id},
                    )
                    logger.debug(f"[Langfuse] Updated root span output, session_id: {session_id}")
                except Exception as update_error:
                    # Never fail due to Langfuse update errors
                    logger.warning(f"[Langfuse] Failed to update root span: {update_error}")

            return result
        except Exception as e:
            # === DEBUG: Log exception to debug log file ===
            import traceback
            
            error_lines = [
                "\n" + "=" * 80,
                "❌ get_response() - Exception occurred",
                f"Exception type: {type(e).__name__}",
                f"Error message: {str(e)}",
                "=" * 80
            ]
            
            # If exception occurred after _graph.ainvoke(), try to log response info
            if 'response' in locals():
                error_lines.append(f"\n📦 Response from _graph.ainvoke():")
                error_lines.append(f"  Message count: {len(response.get('messages', []))}")
                if response.get('messages'):
                    msg_types = [type(m).__name__ for m in response.get('messages', [])[:5]]
                    error_lines.append(f"  Message types (first 5): {msg_types}")
            
            # Log full exception traceback
            error_lines.append("\n📋 Exception traceback:")
            error_lines.append(traceback.format_exc())
            error_lines.append("=" * 80)
            
            _write_debug_log("\n".join(error_lines))
            
            # Update root span with error (don't fail if this fails)
            if root_span:
                try:
                    root_span.update(
                        output=None,
                        metadata={"success": False, "error": str(e), "session_id": session_id},
                    )
                except Exception as update_error:
                    logger.warning(f"[Langfuse] Failed to update root span with error: {update_error}")
            
            logger.error(f"Error getting response: {str(e)}", exc_info=True)
            raise e
        finally:
            # End root span after all child spans/generations complete
            # Note: Langfuse SDK will flush automatically, but we end the span explicitly
            if session_id in self._active_root_spans:
                span_to_end = self._active_root_spans.pop(session_id)
                if span_to_end:
                    try:
                        # FIX: Use update() then end() instead of end(output=...)
                        span_to_end.update(
                            output={"completed": True},
                            metadata={"session_id": session_id},
                        )
                        span_to_end.end()  # No parameters!
                        logger.info(f"[Langfuse] Ended root span, session_id: {session_id}")
                    except Exception as end_error:
                        # Never fail due to Langfuse end errors
                        logger.warning(f"[Langfuse] Failed to end root span: {end_error}")
            
            # Flush Langfuse to ensure all spans are sent
            if langfuse_client:
                try:
                    if hasattr(langfuse_client, 'flush'):
                        langfuse_client.flush()
                        logger.debug(f"[Langfuse] Flushed Langfuse client, session_id: {session_id}")
                    elif hasattr(langfuse_client, 'shutdown'):
                        langfuse_client.shutdown()
                        logger.debug(f"[Langfuse] Shutdown Langfuse client, session_id: {session_id}")
                except Exception as flush_error:
                    # Never fail due to Langfuse flush errors
                    logger.warning(f"[Langfuse] Failed to flush/shutdown Langfuse: {flush_error}")

    async def get_stream_response(
        self, messages: list[Message], session_id: str, user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response.
        """
        # Create Langfuse root span for this streaming conversation
        # Use start_span() - it automatically creates a trace
        root_span = None
        trace_context = None
        langfuse_client = get_langfuse_client()
        
        # Always attempt to create Langfuse tracing, but never fail the main flow
        if langfuse_client:
            try:
                logger.info(f"[Langfuse] Creating root span for streaming session: {session_id}")
                root_span = langfuse_client.start_span(
                    name="Stream Chat Conversation",
                    input={"messages": len(messages), "stream": True},
                    metadata={
                        "environment": settings.ENVIRONMENT.value,
                        "session_id": session_id,
                        "user_id": user_id,
                        "stream": True,
                        "model": settings.LLM_MODEL,
                    },
                )
                
                # Extract trace_context from root span using multiple approaches (same as get_response)
                if hasattr(root_span, 'trace_context'):
                    trace_context = root_span.trace_context
                    logger.info(f"[Langfuse] Got trace_context from root span attribute, type: {type(trace_context).__name__}")
                elif hasattr(root_span, 'trace_id') or hasattr(root_span, 'id'):
                    try:
                        from langfuse.types import TraceContext
                        trace_id = getattr(root_span, 'trace_id', None) or getattr(root_span, 'id', None)
                        observation_id = getattr(root_span, 'observation_id', None) or getattr(root_span, 'id', None)
                        
                        if trace_id:
                            trace_context = TraceContext(
                                trace_id=trace_id,
                                observation_id=observation_id,
                            )
                            logger.info(f"[Langfuse] Constructed TraceContext manually, trace_id: {trace_id}")
                    except (ImportError, AttributeError, Exception) as ctx_error:
                        logger.debug(f"[Langfuse] Could not construct TraceContext manually: {ctx_error}")
                        trace_context = root_span
                        logger.info(f"[Langfuse] Using root span object as trace_context fallback")
                else:
                    trace_context = root_span
                    logger.info(f"[Langfuse] Using root span object directly as trace_context")
                
                logger.info(f"[Langfuse] Root span created successfully for streaming, session_id: {session_id}")
            except Exception as span_error:
                # Never fail the main flow due to Langfuse errors
                logger.warning(f"[Langfuse] Failed to create root span for stream: {span_error}", exc_info=True)
                root_span = None
                trace_context = None
        else:
            logger.debug(f"[Langfuse] Langfuse client not available - tracing disabled for stream")

        config = {
            "configurable": {"thread_id": session_id},
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": False,
                "root_span": root_span,  # Pass root span for potential direct access
                "trace_context": trace_context,  # Pass trace_context for child spans/generations
            },
        }
        if self._graph is None:
            self._graph = await self.create_graph()

        # Store root span in instance variable for _chat to access
        if root_span:
            self._active_root_spans[session_id] = root_span
            logger.debug(f"[Langfuse] Stored root span in instance variable for stream, session_id: {session_id}")
        else:
            logger.debug(f"[Langfuse] root_span is None, NOT storing in instance variable for stream, session_id: {session_id}")

        full_response = ""
        try:
            async for message_token, _ in self._graph.astream(
                {"messages": dump_messages(messages), "session_id": session_id}, config, stream_mode="messages"
            ):
                try:
                    content = message_token.content
                    full_response += content
                    yield content
                except Exception as token_error:
                    logger.error("Error processing token", error=str(token_error), session_id=session_id)
                    # Continue with next token even if current one fails
                    continue

            # Update root span successfully after streaming completes
            if root_span:
                try:
                    root_span.update(
                        output={"response": full_response[:500], "response_length": len(full_response)},  # Limit output size
                        metadata={"success": True, "response_length": len(full_response), "session_id": session_id},
                    )
                    logger.debug(f"[Langfuse] Updated root span output for stream, session_id: {session_id}")
                except Exception as update_error:
                    # Never fail due to Langfuse update errors
                    logger.warning(f"[Langfuse] Failed to update root span for stream: {update_error}")

        except Exception as stream_error:
            # Update root span with error (don't fail if this fails)
            if root_span:
                try:
                    root_span.update(
                        output=None,
                        metadata={"success": False, "error": str(stream_error), "session_id": session_id},
                    )
                except Exception as update_error:
                    logger.warning(f"[Langfuse] Failed to update root span with error: {update_error}")
            
            logger.error("Error in stream processing", error=str(stream_error), session_id=session_id, exc_info=True)
            raise stream_error
        finally:
            # Clean up root span from instance variable and end it
            if session_id in self._active_root_spans:
                span_to_end = self._active_root_spans.pop(session_id)
                if span_to_end:
                    try:
                        # FIX: Use update() then end() instead of end(output=...)
                        span_to_end.update(
                            output={"stream_completed": True},
                            metadata={"session_id": session_id},
                        )
                        span_to_end.end()  # No parameters!
                        logger.info(f"[Langfuse] Ended root span for stream, session_id: {session_id}")
                    except Exception as end_error:
                        # Never fail due to Langfuse end errors
                        logger.warning(f"[Langfuse] Failed to end root span for stream: {end_error}")
            
            # Flush Langfuse to ensure all spans are sent
            if langfuse_client:
                try:
                    if hasattr(langfuse_client, 'flush'):
                        langfuse_client.flush()
                        logger.debug(f"[Langfuse] Flushed Langfuse client for stream, session_id: {session_id}")
                    elif hasattr(langfuse_client, 'shutdown'):
                        langfuse_client.shutdown()
                        logger.debug(f"[Langfuse] Shutdown Langfuse client for stream, session_id: {session_id}")
                except Exception as flush_error:
                    # Never fail due to Langfuse flush errors
                    logger.warning(f"[Langfuse] Failed to flush/shutdown Langfuse for stream: {flush_error}")

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        if self._graph is None:
            self._graph = await self.create_graph()

        state: StateSnapshot = await sync_to_async(self._graph.get_state)(
            config={"configurable": {"thread_id": session_id}}
        )
        return self.__process_messages(state.values["messages"]) if state.values else []

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        try:
            # === DEBUG: Before conversion ===
            debug_lines = [
                "\n" + "=" * 80,
                "🟢 __process_messages() - Starting message conversion",
                f"Input message count: {len(messages)}",
                "=" * 80,
                "🔄 __process_messages - Input BaseMessage:"
            ]
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                debug_lines.append(f"  [{i}] {msg_type}")
                
                # Add more details for specific message types
                if msg_type == "ToolMessage":
                    debug_lines.append(f"      - tool_call_id: {getattr(msg, 'tool_call_id', 'N/A')}")
                    debug_lines.append(f"      - name: {getattr(msg, 'name', 'N/A')}")
                elif msg_type == "AIMessage":
                    tool_calls = getattr(msg, 'tool_calls', None)
                    if tool_calls:
                        debug_lines.append(f"      - tool_calls: {len(tool_calls)} calls")
                        for tc in tool_calls:
                            tc_id = tc.get('id', 'N/A') if isinstance(tc, dict) else getattr(tc, 'id', 'N/A')
                            tc_name = tc.get('name', 'N/A') if isinstance(tc, dict) else getattr(tc, 'name', 'N/A')
                            debug_lines.append(f"        * id={tc_id}, name={tc_name}")

            openai_style_messages = convert_to_openai_messages(messages)

            # === DEBUG: After conversion ===
            debug_lines.append("\n" + "=" * 80)
            debug_lines.append("📝 Converted to OpenAI format:")
            debug_lines.append(f"Converted message count: {len(openai_style_messages)}")
            debug_lines.append("=" * 80)
            for i, msg in enumerate(openai_style_messages):
                role = msg.get('role')
                has_content = bool(msg.get('content'))
                debug_lines.append(f"  [{i}] role={role}, has_content={has_content}")
                if msg.get('tool_call_id'):
                    debug_lines.append(f"      🔧 tool_call_id={msg.get('tool_call_id')}, name={msg.get('name', 'N/A')}")
                if msg.get('tool_calls'):
                    debug_lines.append(f"      🔨 tool_calls={len(msg.get('tool_calls'))} calls")
                    for tc in msg.get('tool_calls', []):
                        tc_id = tc.get('id', 'N/A')
                        tc_name = tc.get('function', {}).get('name', 'N/A')
                        debug_lines.append(f"        * id={tc_id}, name={tc_name}")

            # keep assistant, user, and tool messages to maintain conversation context
            result = [
                Message(**message)
                for message in openai_style_messages
                if message["role"] in ["assistant", "user", "tool"] and message.get("content")
            ]

            # === DEBUG: After filtering ===
            debug_lines.append("\n" + "=" * 80)
            debug_lines.append("✅ Filtered messages returned to frontend:")
            debug_lines.append(f"Final message count: {len(result)}")
            debug_lines.append("=" * 80)
            for i, msg in enumerate(result):
                debug_lines.append(f"  [{i}] role={msg.role}, content_length={len(msg.content) if msg.content else 0}")
                if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                    debug_lines.append(f"      🔧 tool_call_id={msg.tool_call_id}, name={msg.name}")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    debug_lines.append(f"      🔨 tool_calls count={len(msg.tool_calls)}")
                    for tc in msg.tool_calls:
                        tc_id = tc.get('id', 'N/A')
                        tc_func = tc.get('function', {})
                        tc_name = tc_func.get('name', 'N/A')
                        debug_lines.append(f"        * id={tc_id}, name={tc_name}")
            debug_lines.append("=" * 80)
            debug_lines.append("🟢 __process_messages() - Processing completed")
            debug_lines.append("=" * 80)
            _write_debug_log("\n".join(debug_lines))

            return result
            
        except Exception as e:
            # === DEBUG: Log exception during message processing ===
            import traceback
            
            error_lines = [
                "\n" + "=" * 80,
                "❌ __process_messages() - Exception occurred",
                f"Exception type: {type(e).__name__}",
                f"Error message: {str(e)}",
                f"Input message count: {len(messages) if messages else 0}",
                "=" * 80
            ]
            
            # Log input message types for debugging
            if messages:
                error_lines.append("\n📋 Input message types:")
                for i, msg in enumerate(messages[:10]):  # First 10 messages
                    error_lines.append(f"  [{i}] {type(msg).__name__}")
                if len(messages) > 10:
                    error_lines.append(f"  ... and {len(messages) - 10} more messages")
            
            # Log full exception traceback
            error_lines.append("\n📋 Exception traceback:")
            error_lines.append(traceback.format_exc())
            error_lines.append("=" * 80)
            
            _write_debug_log("\n".join(error_lines))
            
            # Re-raise exception to let upper layer handle it
            raise

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        try:
            # Make sure the pool is initialized in the current event loop
            conn_pool = await self._get_connection_pool()

            # Use a new connection for this specific operation
            async with conn_pool.connection() as conn:
                for table in settings.CHECKPOINT_TABLES:
                    try:
                        await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                        logger.info(f"Cleared {table} for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error clearing {table}", error=str(e))
                        raise

        except Exception as e:
            logger.error("Failed to clear chat history", error=str(e))
            raise
