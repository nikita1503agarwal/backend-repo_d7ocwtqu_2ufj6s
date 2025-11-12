"""
Database Schemas for AI Resume Refiner

Each Pydantic model represents a MongoDB collection (collection name is lowercase of class name).
"""

from pydantic import BaseModel, Field
from typing import Optional, List

class User(BaseModel):
    uid: str = Field(..., description="Firebase Auth user id")
    email: str = Field(..., description="Email address")
    name: Optional[str] = Field(None, description="Display name")
    photo_url: Optional[str] = Field(None, description="Avatar URL")
    credits: int = Field(0, description="Available credits for downloads")
    referrals_count: int = Field(0, description="Number of successful referrals")
    referred_by: Optional[str] = Field(None, description="UID of the referrer")

class Resume(BaseModel):
    uid: str = Field(..., description="Owner user id")
    original_text: str = Field(..., description="Extracted raw text from uploaded resume")
    target_role: str = Field(..., description="Target job role, e.g., Data Analyst")
    refined_text: Optional[str] = Field(None, description="AI-refined resume text")
    status: str = Field("uploaded", description="uploaded | refined | downloaded")

class Log(BaseModel):
    uid: Optional[str] = Field(None, description="User id if available")
    type: str = Field(..., description="upload | refine | payment | download | referral")
    message: str = Field(..., description="Human-readable message")
    meta: Optional[dict] = Field(default_factory=dict, description="Additional metadata")

class Payment(BaseModel):
    uid: str = Field(..., description="User id")
    provider: str = Field(..., description="razorpay | stripe")
    amount: int = Field(..., description="Amount in smallest currency unit (e.g., paise)")
    currency: str = Field("INR", description="Currency code")
    status: str = Field("created", description="created | paid | failed")
    order_id: Optional[str] = Field(None, description="Provider order id")
    payment_id: Optional[str] = Field(None, description="Provider payment id")
    signature: Optional[str] = Field(None, description="Verification signature if any")

