"""
Pydantic schemas for validating structured responses from the LLM.
Ensures strict type-safety, valid enum choices, and expected fields.
"""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field, field_validator


class CategoryEnum(str, Enum):
    """Allowed ticket classification categories."""
    BILLING = "Billing"
    TECHNICAL = "Technical"
    ACCOUNT = "Account"
    REFUND = "Refund"
    SUBSCRIPTION = "Subscription"
    SHIPPING = "Shipping"
    OTHER = "Other"


class SentimentEnum(str, Enum):
    """Allowed sentiment values."""
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


class PriorityEnum(str, Enum):
    """Allowed priority levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class TicketAnalysisResponse(BaseModel):
    """
    Validated structured response schema for a customer ticket analysis.
    """
    category: CategoryEnum = Field(
        ...,
        description="The classified category of the support ticket."
    )
    sentiment: SentimentEnum = Field(
        ...,
        description="The customer's emotional sentiment."
    )
    priority: PriorityEnum = Field(
        ...,
        description="The urgency and operational priority of the ticket."
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Key domain entities, product names, or identifiers extracted from the ticket."
    )
    summary: str = Field(
        ...,
        min_length=5,
        description="A concise 1-2 sentence objective summary of the customer's issue."
    )
    suggested_response: str = Field(
        ...,
        min_length=10,
        description="A polite and helpful draft response ready for a support agent to review."
    )

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        """Handle case-insensitive matching for category values."""
        if isinstance(value, str):
            clean_val = value.strip().capitalize()
            for cat in CategoryEnum:
                if cat.value.lower() == clean_val.lower():
                    return cat.value
        return value

    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, value: str) -> str:
        """Handle case-insensitive matching for sentiment values."""
        if isinstance(value, str):
            clean_val = value.strip().capitalize()
            for sent in SentimentEnum:
                if sent.value.lower() == clean_val.lower():
                    return sent.value
        return value

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: str) -> str:
        """Handle case-insensitive matching for priority values."""
        if isinstance(value, str):
            clean_val = value.strip().capitalize()
            for prio in PriorityEnum:
                if prio.value.lower() == clean_val.lower():
                    return prio.value
        return value

    @field_validator("summary", "suggested_response", mode="before")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        """Trim surrounding whitespace from text responses."""
        if isinstance(value, str):
            return value.strip()
        return value
