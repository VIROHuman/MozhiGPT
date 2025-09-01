-- MozhiGPT Database Setup Script
-- Run this in Beekeeper Studio after connecting to your PostgreSQL database

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Table for storing Tamil training texts
CREATE TABLE tamil_texts (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    source VARCHAR(255) DEFAULT 'Thamizhmalai',
    language VARCHAR(10) DEFAULT 'ta',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE
);

-- Create index for faster queries
CREATE INDEX idx_tamil_texts_processed ON tamil_texts(processed);
CREATE INDEX idx_tamil_texts_source ON tamil_texts(source);

-- 2. Table for storing chat conversations
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    model_version VARCHAR(255) DEFAULT 'tamil-gpt-v1',
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for session lookups
CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_conversations_created ON conversations(created_at);

-- 3. Table for tracking model versions and training progress
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    base_model VARCHAR(255) NOT NULL,
    training_samples INTEGER DEFAULT 0,
    training_epochs INTEGER DEFAULT 0,
    performance_score FLOAT,
    model_path TEXT NOT NULL,
    config_json TEXT,
    status VARCHAR(50) DEFAULT 'training',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- 4. Table for storing training metrics and logs
CREATE TABLE training_metrics (
    id SERIAL PRIMARY KEY,
    model_version_id INTEGER REFERENCES model_versions(id),
    epoch INTEGER NOT NULL,
    step INTEGER NOT NULL,
    train_loss FLOAT,
    eval_loss FLOAT,
    learning_rate FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for metrics queries
CREATE INDEX idx_training_metrics_model ON training_metrics(model_version_id);
CREATE INDEX idx_training_metrics_epoch ON training_metrics(epoch);

-- 5. Table for storing user feedback on AI responses
CREATE TABLE user_feedback (
    id SERIAL PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    feedback_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for feedback queries
CREATE INDEX idx_user_feedback_conversation ON user_feedback(conversation_id);
CREATE INDEX idx_user_feedback_rating ON user_feedback(rating);

-- Create sample data for testing
INSERT INTO model_versions (name, base_model, model_path, status) VALUES 
('tamil-gpt-test', 'microsoft/DialoGPT-small', './models/tamil-gpt', 'testing');

-- Verify tables were created
SELECT 
    table_name, 
    column_name, 
    data_type 
FROM information_schema.columns 
WHERE table_schema = 'public' 
ORDER BY table_name, ordinal_position;

-- Show table summary
SELECT 
    schemaname,
    tablename,
    tableowner
FROM pg_tables 
WHERE schemaname = 'public';

-- Success message
SELECT 'MozhiGPT database setup completed successfully!' as status;

