"""This file contains the user profile model for long-term memory storage."""

from datetime import datetime, UTC
from typing import Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field

from app.models.base import BaseModel


class UserProfile(BaseModel, table=True):
    """UserProfile model for storing user profile information.

    Attributes:
        user_id: Primary key, user identifier as text
        name: User's name
        timezone: User's timezone preference
        preferred_units: Preferred unit system (metric/imperial)
        vehicle_vin: Vehicle VIN number
        vehicle_type: Type of vehicle
        notes: Additional notes as JSON
        updated_at: Last update timestamp
    """

    user_id: str = Field(primary_key=True, description="User ID as primary key")
    name: Optional[str] = Field(default=None, description="User's name")
    timezone: Optional[str] = Field(default=None, description="User's timezone")
    preferred_units: Optional[str] = Field(default=None, description="Preferred units")
    vehicle_vin: Optional[str] = Field(default=None, description="Vehicle VIN")
    vehicle_type: Optional[str] = Field(default=None, description="Vehicle type")
    notes: Optional[dict] = Field(default=None, sa_column=Column(JSON), description="Additional notes")
    updated_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(UTC), description="Last update time")

