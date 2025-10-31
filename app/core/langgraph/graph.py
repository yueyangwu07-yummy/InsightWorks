"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

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
    BaseMessage,
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
from app.core.metrics import llm_inference_duration_seconds
from app.core.prompts import SYSTEM_PROMPT

# Import Langfuse for manual tracking
try:
    from langfuse import Langfuse
    _langfuse_available = True
except ImportError:
    _langfuse_available = False
    logger.warning("langfuse package not available. Tracking will be disabled.")


def get_langfuse_client():
    """Get or create Langfuse client instance.
    
    Returns:
        Optional[Langfuse]: Langfuse client instance if configured, None otherwise.
    """
    if not _langfuse_available:
        return None
    
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        return None
    
    try:
        return Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    except Exception as e:
        logger.warning(f"Failed to create Langfuse client: {e}")
        return None
from app.schemas import (
    GraphState,
    Message,
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
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None

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
        messages = prepare_messages(state.messages, self.llm, SYSTEM_PROMPT)

        llm_calls_num = 0

        # Configure retry attempts based on environment
        max_retries = settings.MAX_LLM_CALL_RETRIES

        # Prepare messages for Langfuse tracking
        langfuse_messages = dump_messages(messages)
        langfuse_generation = None

        # Get trace from config (set by get_response/get_stream_response)
        # Store trace in config to pass it through LangGraph nodes
        trace = None
        if config and isinstance(config, dict):
            trace = config.get("_langfuse_trace")
        if trace:
            logger.debug(f"Trace available for generation in _chat, session_id: {state.session_id}")
        else:
            logger.debug(f"No trace available in _chat, session_id: {state.session_id}")

        for attempt in range(max_retries):
            try:
                # Create Langfuse generation for tracking
                # In Langfuse 3.x, generation must be created from a trace, not from client
                if trace:
                    try:
                        langfuse_generation = trace.generation(
                            name=f"LLM Call {llm_calls_num + 1}",
                            model=settings.LLM_MODEL,
                            model_parameters={
                                "temperature": settings.DEFAULT_LLM_TEMPERATURE,
                                "max_tokens": settings.MAX_TOKENS,
                                **self._get_model_kwargs(),
                            },
                            input=langfuse_messages,
                            metadata={
                                "attempt": attempt + 1,
                                "environment": settings.ENVIRONMENT.value,
                            },
                        )
                    except Exception as langfuse_error:
                        logger.debug(f"Failed to create Langfuse generation: {langfuse_error}")
                else:
                    # No trace available, cannot create generation
                    logger.debug("No trace available for generation, skipping Langfuse tracking for this call")

                with llm_inference_duration_seconds.labels(model=self.llm.model_name).time():
                    # Pass config to LLM call to ensure context is preserved
                    response = await self.llm.ainvoke(langfuse_messages, config=config)
                    generated_state = {"messages": [response]}

                # Update Langfuse generation with response
                if langfuse_generation:
                    try:
                        response_content = response.content if hasattr(response, 'content') else str(response)
                        langfuse_generation.end(
                            output=response_content,
                            metadata={"success": True, "llm_calls_num": llm_calls_num + 1},
                        )
                    except Exception as langfuse_error:
                        logger.debug(f"Failed to update Langfuse generation: {langfuse_error}")

                logger.info(
                    "llm_response_generated",
                    session_id=state.session_id,
                    llm_calls_num=llm_calls_num + 1,
                    model=settings.LLM_MODEL,
                    environment=settings.ENVIRONMENT.value,
                )
                return generated_state
            except OpenAIError as e:
                # Update Langfuse generation with error
                if langfuse_generation:
                    try:
                        langfuse_generation.end(
                            output=None,
                            metadata={
                                "success": False,
                                "error": str(e),
                                "attempt": attempt + 1,
                            },
                        )
                    except Exception:
                        pass

                logger.error(
                    "llm_call_failed",
                    llm_calls_num=llm_calls_num,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                    environment=settings.ENVIRONMENT.value,
                )
                llm_calls_num += 1

                # In production, we might want to fall back to a more reliable model
                if settings.ENVIRONMENT == Environment.PRODUCTION and attempt == max_retries - 2:
                    fallback_model = "gpt-4o"
                    logger.warning(
                        "using_fallback_model", model=fallback_model, environment=settings.ENVIRONMENT.value
                    )
                    self.llm.model_name = fallback_model

                continue

        # Final error tracking
        if langfuse_generation:
            try:
                langfuse_generation.end(
                    output=None,
                    metadata={"success": False, "error": "Max retries exceeded"},
                )
            except Exception:
                pass

        raise Exception(f"Failed to get a response from the LLM after {max_retries} attempts")

    # Define our tool node
    async def _tool_call(self, state: GraphState, config: Optional[RunnableConfig] = None) -> GraphState:
        """Process tool calls from the last message.

        Args:
            state: The current agent state containing messages and tool calls.
            config (Optional[RunnableConfig]): Runtime configuration for the runnable.

        Returns:
            Dict with updated messages containing tool responses.
        """
        outputs = []
        for tool_call in state.messages[-1].tool_calls:
            # Pass config to tool call to ensure context is preserved
            tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"], config=config)
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    def _should_continue(self, state: GraphState) -> Literal["end", "continue"]:
        """Determine if the agent should continue or end based on the last message.

        Args:
            state: The current agent state containing messages.

        Returns:
            Literal["end", "continue"]: "end" if there are no tool calls, "continue" otherwise.
        """
        messages = state.messages
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        """Create and configure the LangGraph workflow.

        Returns:
            Optional[CompiledStateGraph]: The configured LangGraph instance or None if init fails
        """
        if self._graph is None:
            try:
                graph_builder = StateGraph(GraphState)
                graph_builder.add_node("chat", self._chat)
                graph_builder.add_node("tool_call", self._tool_call)
                graph_builder.add_conditional_edges(
                    "chat",
                    self._should_continue,
                    {"continue": "tool_call", "end": END},
                )
                graph_builder.add_edge("tool_call", "chat")
                graph_builder.set_entry_point("chat")
                graph_builder.set_finish_point("chat")

                # Get connection pool (may be None in production if DB unavailable)
                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool)
                    await checkpointer.setup()
                else:
                    # In production, proceed without checkpointer if needed
                    checkpointer = None
                    if settings.ENVIRONMENT != Environment.PRODUCTION:
                        raise Exception("Connection pool initialization failed")

                self._graph = graph_builder.compile(
                    checkpointer=checkpointer, name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value})"
                )

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
        if self._graph is None:
            self._graph = await self.create_graph()

        # Create Langfuse trace for this conversation
        trace = None
        langfuse_client = get_langfuse_client()
        if langfuse_client:
            try:
                logger.debug(f"Creating Langfuse trace for get_response, session_id: {session_id}")
                trace = langfuse_client.trace(
                    name="Chat Conversation",
                    session_id=session_id,
                    user_id=user_id,
                    metadata={
                        "environment": settings.ENVIRONMENT.value,
                        "message_count": len(messages),
                    },
                )
                logger.debug(f"Langfuse trace created successfully for get_response, session_id: {session_id}")
            except Exception as trace_error:
                logger.warning(f"Failed to create Langfuse trace: {trace_error}", exc_info=True)

        config = {
            "configurable": {"thread_id": session_id},
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": False,
            },
            # Store trace in config to pass it through LangGraph nodes
            "_langfuse_trace": trace,
        }

        try:
            response = await self._graph.ainvoke(
                {"messages": dump_messages(messages), "session_id": session_id}, config
            )
            result = self.__process_messages(response["messages"])

            # End trace successfully
            if trace:
                try:
                    trace.update(
                        output={"messages": result},
                        metadata={"success": True},
                    )
                except Exception:
                    pass

            return result
        except Exception as e:
            # End trace with error
            if trace:
                try:
                    trace.update(
                        output=None,
                        metadata={"success": False, "error": str(e)},
                    )
                except Exception:
                    pass
            logger.error(f"Error getting response: {str(e)}")
            raise e

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
        # Create Langfuse trace for this streaming conversation
        trace = None
        langfuse_client = get_langfuse_client()
        if langfuse_client:
            try:
                logger.debug(f"Creating Langfuse trace for stream, session_id: {session_id}")
                trace = langfuse_client.trace(
                    name="Stream Chat Conversation",
                    session_id=session_id,
                    user_id=user_id,
                    metadata={
                        "environment": settings.ENVIRONMENT.value,
                        "message_count": len(messages),
                        "stream": True,
                    },
                )
                logger.debug(f"Langfuse trace created successfully, session_id: {session_id}, trace type: {type(trace)}")
            except Exception as trace_error:
                logger.warning(f"Failed to create Langfuse trace: {trace_error}", exc_info=True)
        else:
            logger.debug(f"Langfuse client not available. PUBLIC_KEY exists: {bool(settings.LANGFUSE_PUBLIC_KEY)}, SECRET_KEY exists: {bool(settings.LANGFUSE_SECRET_KEY)}")

        config = {
            "configurable": {"thread_id": session_id},
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": False,
            },
            # Store trace in config to pass it through LangGraph nodes
            "_langfuse_trace": trace,
        }
        if self._graph is None:
            self._graph = await self.create_graph()

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

            # End trace successfully after streaming completes
            if trace:
                try:
                    trace.update(
                        output={"response": full_response},
                        metadata={"success": True, "response_length": len(full_response)},
                    )
                except Exception:
                    pass

        except Exception as stream_error:
            # End trace with error
            if trace:
                try:
                    trace.update(
                        output=None,
                        metadata={"success": False, "error": str(stream_error)},
                    )
                except Exception:
                    pass
            logger.error("Error in stream processing", error=str(stream_error), session_id=session_id)
            raise stream_error

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
        openai_style_messages = convert_to_openai_messages(messages)
        # keep just assistant and user messages
        return [
            Message(**message)
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

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
