from sqlalchemy.orm import Session
from . import models, schemas, auth
import json

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_summary_history(db: Session, user: models.User, summary_data: schemas.SummaryHistoryCreate):
    # Convert entities to JSON string if present
    entities_json = json.dumps(summary_data.entities) if summary_data.entities else None
    
    db_summary = models.SummaryHistory(
        user_id=user.id,
        input_text=summary_data.input_text,
        summary=summary_data.summary,
        title=summary_data.title,
        model_used=summary_data.model_used,
        entities=entities_json # Save as JSON string
    )
    db.add(db_summary)
    db.commit()
    db.refresh(db_summary)
    return db_summary

def get_user_summaries(db: Session, user: models.User):
    summaries = db.query(models.SummaryHistory).filter(models.SummaryHistory.user_id == user.id).all()
    # Convert entities JSON string back to list of dicts
    for summary in summaries:
        if summary.entities:
            summary.entities = json.loads(summary.entities)
    return summaries

def create_finetune_dataset(db: Session, user_id: int, file_path: str):
    db_dataset = models.FinetuneDataset(user_id=user_id, file_path=file_path)
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

def get_finetune_datasets(db: Session, user_id: int):
    return db.query(models.FinetuneDataset).filter(models.FinetuneDataset.user_id == user_id).all()

def create_finetune_model(db: Session, user_id: int, model_data: schemas.FinetuneModelCreate):
    db_model = models.FinetuneModel(
        user_id=user_id,
        model_name=model_data.model_name,
        base_model=model_data.base_model,
        status=model_data.status,
        model_path=model_data.model_path
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

def get_finetune_models(db: Session, user_id: int):
    return db.query(models.FinetuneModel).filter(models.FinetuneModel.user_id == user_id).all()

def get_finetune_model_by_id(db: Session, model_id: int):
    return db.query(models.FinetuneModel).filter(models.FinetuneModel.id == model_id).first()

def update_finetune_model_status(db: Session, model_id: int, status: str, model_path: str = None):
    db_model = db.query(models.FinetuneModel).filter(models.FinetuneModel.id == model_id).first()
    if db_model:
        db_model.status = status
        if model_path:
            db_model.model_path = model_path
        db.commit()
        db.refresh(db_model)
    return db_model

def create_webhook(db: Session, user_id: int, webhook: schemas.WebhookCreate):
    db_webhook = models.Webhook(
        user_id=user_id,
        url=webhook.url,
        event_type=webhook.event_type,
        is_active=webhook.is_active
    )
    db.add(db_webhook)
    db.commit()
    db.refresh(db_webhook)
    return db_webhook

def get_webhooks(db: Session, user_id: int):
    return db.query(models.Webhook).filter(models.Webhook.user_id == user_id).all()

def get_webhook_by_id(db: Session, webhook_id: int):
    return db.query(models.Webhook).filter(models.Webhook.id == webhook_id).first()

def delete_webhook(db: Session, webhook_id: int):
    db_webhook = db.query(models.Webhook).filter(models.Webhook.id == webhook_id).first()
    if db_webhook:
        db.delete(db_webhook)
        db.commit()
        return True
    return False

def create_feedback(db: Session, user_id: int, feedback: schemas.FeedbackCreate):
    db_feedback = models.Feedback(
        user_id=user_id,
        summary_id=feedback.summary_id,
        rating=feedback.rating,
        comment=feedback.comment
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback

def get_feedback(db: Session, user_id: int):
    return db.query(models.Feedback).filter(models.Feedback.user_id == user_id).all()
