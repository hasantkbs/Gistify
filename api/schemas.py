from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, List, Dict, Any

class SummaryHistoryBase(BaseModel):
    input_text: str
    summary: str
    title: Optional[str] = None
    model_used: Optional[str] = None
    entities: Optional[List[Dict[str, Any]]] = None # New field

class SummaryHistoryCreate(SummaryHistoryBase):
    pass

class SummaryHistory(SummaryHistoryBase):
    id: int
    user_id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class FinetuneModelBase(BaseModel):
    model_name: str
    base_model: str
    status: str
    model_path: Optional[str] = None

class FinetuneModelCreate(FinetuneModelBase):
    pass

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class FinetuneDatasetBase(BaseModel):
    file_path: str

class FinetuneDatasetCreate(FinetuneDatasetBase):
    pass

    model_config = ConfigDict(from_attributes=True)

class WebhookBase(BaseModel):
    url: str
    event_type: str
    is_active: bool = True

class WebhookCreate(WebhookBase):
    pass

class Webhook(WebhookBase):
    id: int
    user_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class FeedbackBase(BaseModel):
    summary_id: Optional[int] = None
    rating: Optional[int] = None
    comment: Optional[str] = None

class FeedbackCreate(FeedbackBase):
    pass

class Feedback(FeedbackBase):
    id: int
    user_id: int
    model_config = ConfigDict(from_attributes=True)

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    summaries: Optional[List[Any]] = None
    finetune_models: Optional[List[Any]] = None
    finetune_datasets: Optional[List[Any]] = None
    webhooks: Optional[List[Any]] = None # New relationship
    feedback: Optional[List[Any]] = None # New relationship

    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class UsageStats(BaseModel):
    total_gists: int
    gists_last_30_days: int
    daily_gists_last_7_days: dict[str, int]

class MultiSummarizeRequest(BaseModel):
    texts: List[str]

class MultiSummarizeRequest(BaseModel):
    texts: List[str]




