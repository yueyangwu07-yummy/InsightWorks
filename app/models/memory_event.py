"""This file contains the memory event model for tracking memory operations."""

import uuid
from typing import Optional

from sqlalchemy import JSON, Column, Float
from sqlmodel import Field

from app.models.base import BaseModel


class MemoryEvent(BaseModel, table=True):
    """MemoryEvent model for tracking memory storage and retrieval events.

    Attributes:
        id: UUID primary key
        user_id: User ID associated with the memory event
        type: Type of memory event (profile_fact, conversation_summary, etc.)
        text: The memory content as text
        embedding_id: Associated embedding ID in Qdrant
        score: Similarity score for retrieved memories
        source_message_id: Original message that triggered this memory
        created_at: When the event was created
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: str = Field(index=True, description="User ID associated with the event")
    type: str = Field(description="Type of memory event")
    text: str = Field(description="Memory content as text")
    embedding_id: Optional[str] = Field(default=None, description="Qdrant embedding ID")
    score: Optional[float] = Field(default=None, sa_column=Column(Float), description="Similarity score")
    source_message_id: Optional[str] = Field(default=None, description="Source message ID")
    created_at: Optional[str] = Field(default=None, description="Event creation timestamp")

