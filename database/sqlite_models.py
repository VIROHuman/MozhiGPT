"""
SQLite Database models for MozhiGPT (No PostgreSQL required)

This is a simplified version using SQLite for easy local development.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional
import json

class SQLiteDatabase:
    """Simple SQLite database for MozhiGPT."""
    
    def __init__(self, db_path: str = "mozhigpt.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tamil_texts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tamil_texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source TEXT DEFAULT 'Thamizhmalai',
                language TEXT DEFAULT 'ta',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT 0
            )
        """)
        
        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                model_version TEXT DEFAULT 'tamil-gpt-v1',
                response_time_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create model_versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                base_model TEXT NOT NULL,
                training_samples INTEGER DEFAULT 0,
                training_epochs INTEGER DEFAULT 0,
                performance_score REAL,
                model_path TEXT NOT NULL,
                config_json TEXT,
                status TEXT DEFAULT 'training',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        # Create training_metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version_id INTEGER,
                epoch INTEGER NOT NULL,
                step INTEGER NOT NULL,
                train_loss REAL,
                eval_loss REAL,
                learning_rate REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_version_id) REFERENCES model_versions (id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tamil_texts_processed ON tamil_texts(processed)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)")
        
        # Insert sample data
        cursor.execute("""
            INSERT OR IGNORE INTO model_versions (name, base_model, model_path, status) 
            VALUES ('tamil-gpt-test', 'microsoft/DialoGPT-small', './models/tamil-gpt', 'testing')
        """)
        
        conn.commit()
        conn.close()
        print("✅ SQLite database initialized successfully!")
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def insert_tamil_text(self, text: str, source: str = "Thamizhmalai"):
        """Insert Tamil text for training."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tamil_texts (text, source) VALUES (?, ?)",
            (text, source)
        )
        conn.commit()
        conn.close()
    
    def get_tamil_texts(self, limit: int = 1000, offset: int = 0, unprocessed_only: bool = True):
        """Get Tamil texts for training."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if unprocessed_only:
            cursor.execute(
                "SELECT id, text, source FROM tamil_texts WHERE processed = 0 LIMIT ? OFFSET ?",
                (limit, offset)
            )
        else:
            cursor.execute(
                "SELECT id, text, source FROM tamil_texts LIMIT ? OFFSET ?",
                (limit, offset)
            )
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def mark_texts_processed(self, text_ids: list):
        """Mark texts as processed."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE tamil_texts SET processed = 1 WHERE id = ?",
            [(text_id,) for text_id in text_ids]
        )
        conn.commit()
        conn.close()
    
    def save_conversation(self, conversation_id: str, session_id: str, 
                         user_message: str, ai_response: str, 
                         model_version: str = "tamil-gpt-v1", 
                         response_time_ms: Optional[int] = None):
        """Save conversation to database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO conversations 
            (id, session_id, user_message, ai_response, model_version, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (conversation_id, session_id, user_message, ai_response, model_version, response_time_ms))
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, session_id: str, limit: int = 10):
        """Get conversation history for a session."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_message, ai_response, created_at 
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (session_id, limit))
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_stats(self):
        """Get database statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM tamil_texts")
        total_texts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tamil_texts WHERE processed = 1")
        processed_texts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_versions")
        total_models = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_texts": total_texts,
            "processed_texts": processed_texts,
            "total_conversations": total_conversations,
            "total_models": total_models
        }


def init_sqlite_database(db_path: str = "mozhigpt.db"):
    """Initialize SQLite database."""
    return SQLiteDatabase(db_path)

