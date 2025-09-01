"""
Database models for MozhiGPT

This module defines database schemas for training data management,
conversation storage, and model versioning.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from typing import Optional

Base = declarative_base()


class TamilText(Base):
    """Table for storing Tamil training texts."""
    __tablename__ = "tamil_texts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    source = Column(String(255), default="Thamizhmalai")
    language = Column(String(10), default="ta")
    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<TamilText(id={self.id}, source='{self.source}', length={len(self.text) if self.text else 0})>"


class Conversation(Base):
    """Table for storing chat conversations."""
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    model_version = Column(String(255), default="tamil-gpt-v1")
    response_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, session='{self.session_id}')>"


class ModelVersion(Base):
    """Table for tracking model versions and training progress."""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    base_model = Column(String(255), nullable=False)
    training_samples = Column(Integer, default=0)
    training_epochs = Column(Integer, default=0)
    performance_score = Column(Float)
    model_path = Column(Text, nullable=False)
    config_json = Column(Text)  # Store training configuration
    status = Column(String(50), default="training")  # training, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    def __repr__(self):
        return f"<ModelVersion(name='{self.name}', status='{self.status}')>"


class TrainingMetrics(Base):
    """Table for storing training metrics and logs."""
    __tablename__ = "training_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version_id = Column(Integer, ForeignKey('model_versions.id'))
    epoch = Column(Integer, nullable=False)
    step = Column(Integer, nullable=False)
    train_loss = Column(Float)
    eval_loss = Column(Float)
    learning_rate = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    model_version = relationship("ModelVersion", backref="metrics")
    
    def __repr__(self):
        return f"<TrainingMetrics(epoch={self.epoch}, step={self.step}, loss={self.train_loss})>"


class UserFeedback(Base):
    """Table for storing user feedback on AI responses."""
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id'))
    rating = Column(Integer)  # 1-5 star rating
    feedback_text = Column(Text)
    feedback_type = Column(String(50))  # positive, negative, suggestion
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", backref="feedback")
    
    def __repr__(self):
        return f"<UserFeedback(rating={self.rating}, type='{self.feedback_type}')>"


# Database configuration
class DatabaseConfig:
    """Database configuration class."""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "mozhigpt",
        username: str = "postgres",
        password: str = "password"
    ):
        if database_url:
            self.database_url = database_url
        else:
            self.database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
    
    def drop_all_tables(self):
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)


# Database utility functions
def get_database():
    """Get database instance (dependency injection)."""
    import os
    
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return DatabaseConfig(database_url=database_url)
    else:
        return DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "mozhigpt"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password")
        )


def init_database():
    """Initialize database with tables."""
    db = get_database()
    db.create_tables()
    print("✅ Database tables created successfully!")
    return db

