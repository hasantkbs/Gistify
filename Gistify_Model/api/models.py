from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    summaries = relationship("SummaryHistory", back_populates="user")
    finetune_models = relationship("FinetuneModel", back_populates="user")
    finetune_datasets = relationship("FinetuneDataset", back_populates="user")
    webhooks = relationship("Webhook", back_populates="user") # New relationship
    feedback = relationship("Feedback", back_populates="user") # New relationship

class SummaryHistory(Base):
    __tablename__ = "summary_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    input_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    title = Column(String, nullable=True) # New field
    model_used = Column(String, nullable=True) # New field
    entities = Column(Text, nullable=True) # New field for storing JSON string of entities
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="summaries")

class FinetuneModel(Base):
    __tablename__ = "finetune_models"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_name = Column(String, nullable=False)
    base_model = Column(String, nullable=False)
    status = Column(String, default="pending") # e.g., pending, training, completed, failed
    model_path = Column(String, nullable=True) # Path to saved model files
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="finetune_models")

class FinetuneDataset(Base):
    __tablename__ = "finetune_datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="finetune_datasets")

class Webhook(Base):
    __tablename__ = "webhooks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    url = Column(String, nullable=False)
    event_type = Column(String, nullable=False) # e.g., "summary_completed", "finetune_completed"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="webhooks")

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    summary_id = Column(Integer, ForeignKey("summary_history.id"), nullable=True) # Optional: link to a specific summary
    rating = Column(Integer, nullable=True) # e.g., 1-5
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="feedback")
    summary = relationship("SummaryHistory") # One-to-one or one-to-many, depending on design