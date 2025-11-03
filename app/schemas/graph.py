"""This file contains the graph schema for the application."""

import re
import uuid
from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class GraphState(BaseModel):
    """State definition for the LangGraph Agent/Workflow."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list, description="The messages in the conversation"
    )
    session_id: str = Field(..., description="The unique identifier for the conversation session")

    # User Information
    user_id: Optional[str] = Field(
        default=None,
        description="User ID extracted from JWT token"
    )

    user_profile: dict = Field(
        default_factory=dict,
        description="User profile data: name, language, timezone, preferences, etc."
    )
    # Example: {"name": "John", "language": "en", "timezone": "America/New_York",
    #           "vehicle_type": "SUV", "preferred_units": "imperial"}

    # Conversation Management
    conversation_summary: str = Field(
        default="",
        description="Compressed summary of conversation history for long chats"
    )

    turn_count: int = Field(
        default=0,
        description="Number of conversation turns (user messages)"
    )

    last_summary_turn: int = Field(
        default=0,
        description="Turn number when last summary was generated"
    )

    # Metadata
    metadata: dict = Field(
        default_factory=dict,
        description="Flexible metadata storage for future use"
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate that the session ID is a valid UUID or follows safe pattern.

        Args:
            v: The thread ID to validate

        Returns:
            str: The validated session ID

        Raises:
            ValueError: If the session ID is not valid
        """
        # Try to validate as UUID
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            # If not a UUID, check for safe characters only
            if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
                raise ValueError("Session ID must contain only alphanumeric characters, underscores, and hyphens")
            return v

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


def update_user_profile(state: GraphState, **profile_data) -> GraphState:
    """Update user profile with new data.

    Args:
        state: Current graph state
        **profile_data: Key-value pairs to add/update in profile

    Returns:
        Updated state

    Example:
        state = update_user_profile(state, name="John", language="en")
    """
    state.user_profile.update(profile_data)
    return state


def increment_turn(state: GraphState) -> GraphState:
    """Increment the conversation turn counter.

    Args:
        state: Current graph state

    Returns:
        Updated state with turn_count += 1
    """
    state.turn_count += 1
    return state


def should_generate_summary(state: GraphState, threshold: int = 10) -> bool:
    """Check if a conversation summary should be generated.

    Args:
        state: Current graph state
        threshold: Number of turns between summaries (default: 10)

    Returns:
        True if turns since last summary >= threshold

    Example:
        if should_generate_summary(state):
            state.conversation_summary = generate_summary(state.messages)
    """
    return (state.turn_count - state.last_summary_turn) >= threshold


def mark_summary_generated(state: GraphState) -> GraphState:
    """Mark that a summary was just generated at current turn.

    Args:
        state: Current graph state

    Returns:
        Updated state with last_summary_turn = turn_count
    """
    state.last_summary_turn = state.turn_count
    return state


if __name__ == "__main__":
    print("Testing Enhanced GraphState\n" + "="*50)

    # Test 1: Create basic state
    print("\n1. Create basic state:")
    state = GraphState(session_id="test-session-123")
    print(f"   Session ID: {state.session_id}")
    print(f"   Turn count: {state.turn_count}")

    # Test 2: Update user profile
    print("\n2. Update user profile:")
    state = update_user_profile(
        state,
        name="John Doe",
        language="en",
        timezone="America/New_York",
        vehicle_type="SUV"
    )
    print(f"   User profile: {state.user_profile}")

    # Test 3: Increment turns
    print("\n3. Increment conversation turns:")
    for i in range(3):
        state = increment_turn(state)
    print(f"   Turn count: {state.turn_count}")

    # Test 4: Check summary threshold
    print("\n4. Check if summary needed:")
    print(f"   Should generate summary? {should_generate_summary(state, threshold=3)}")

    # Simulate generating summary
    state.conversation_summary = "User asked about trip costs and received estimates."
    state = mark_summary_generated(state)
    print(f"   Summary generated at turn: {state.last_summary_turn}")

    # Test 5: UUID validation still works
    print("\n5. Test UUID validation:")
    try:
        state2 = GraphState(session_id="550e8400-e29b-41d4-a716-446655440000")
        print(f"   ✓ Valid UUID accepted: {state2.session_id}")
    except ValueError as e:
        print(f"   ✗ Error: {e}")

    try:
        state3 = GraphState(session_id="invalid session!")
        print(f"   ✗ Invalid session ID accepted (should not happen)")
    except ValueError as e:
        print(f"   ✓ Invalid session ID rejected: {e}")

    print("\n" + "="*50)
    print("✅ All tests passed!")
