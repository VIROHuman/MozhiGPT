#!/usr/bin/env python3
"""
Database-Enhanced Tamil GPT Training

This script uses database storage for efficient handling of the massive
Tamil dataset and training progress tracking.
"""

import os
import logging
from datetime import datetime
from typing import Iterator, Dict, Any
import json

from database import init_database, TamilText, ModelVersion, TrainingMetrics
from train import TamilModelTrainer, TamilTrainingConfig
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseTamilTrainer(TamilModelTrainer):
    """Enhanced trainer with database support."""
    
    def __init__(self, config: TamilTrainingConfig, database_session: Session):
        super().__init__(config)
        self.db_session = database_session
        self.model_version = None
    
    def setup_database_training(self) -> None:
        """Setup database for training tracking."""
        logger.info("Setting up database training tracking...")
        
        # Create model version record
        self.model_version = ModelVersion(
            name=f"tamil-gpt-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            base_model=self.config.base_model_name,
            training_samples=self.config.max_samples or 73925706,  # Full dataset size
            training_epochs=self.config.num_train_epochs,
            model_path=self.config.output_dir,
            config_json=json.dumps(self.config.__dict__, default=str),
            status="training"
        )
        
        self.db_session.add(self.model_version)
        self.db_session.commit()
        
        logger.info(f"Created model version: {self.model_version.name}")
    
    def load_tamil_texts_from_db(self, batch_size: int = 1000) -> Iterator[Dict[str, Any]]:
        """Load Tamil texts from database in batches."""
        logger.info("Loading Tamil texts from database...")
        
        offset = 0
        while True:
            # Query batch of texts
            texts = self.db_session.query(TamilText).filter(
                TamilText.processed == False
            ).offset(offset).limit(batch_size).all()
            
            if not texts:
                break
            
            # Convert to training format
            for text_record in texts:
                yield {
                    "text": text_record.text,
                    "source": text_record.source,
                    "id": text_record.id
                }
                
                # Mark as processed
                text_record.processed = True
            
            # Commit processed status
            self.db_session.commit()
            offset += batch_size
            
            if offset % 10000 == 0:
                logger.info(f"Processed {offset} texts from database...")
    
    def log_training_metrics(self, epoch: int, step: int, train_loss: float, 
                           eval_loss: float = None, learning_rate: float = None) -> None:
        """Log training metrics to database."""
        if self.model_version:
            metrics = TrainingMetrics(
                model_version_id=self.model_version.id,
                epoch=epoch,
                step=step,
                train_loss=train_loss,
                eval_loss=eval_loss,
                learning_rate=learning_rate
            )
            self.db_session.add(metrics)
            self.db_session.commit()
    
    def complete_training(self, final_performance: float = None) -> None:
        """Mark training as completed in database."""
        if self.model_version:
            self.model_version.status = "completed"
            self.model_version.completed_at = datetime.utcnow()
            if final_performance:
                self.model_version.performance_score = final_performance
            
            self.db_session.commit()
            logger.info(f"Training completed for model: {self.model_version.name}")


def populate_database_with_tamil_texts(db_session: Session, limit: int = None) -> None:
    """Populate database with Tamil texts from the dataset file."""
    logger.info("Populating database with Tamil texts...")
    
    # Check if already populated
    existing_count = db_session.query(TamilText).count()
    if existing_count > 0:
        logger.info(f"Database already has {existing_count} texts. Skipping population.")
        return
    
    # Read from the downloaded dataset file
    dataset_file = "./data/Thamizhmalai/train.jsonl"
    if not os.path.exists(dataset_file):
        logger.error(f"Dataset file not found: {dataset_file}")
        return
    
    logger.info(f"Reading Tamil texts from {dataset_file}...")
    batch_size = 1000
    batch = []
    processed = 0
    
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if limit and processed >= limit:
                    break
                
                try:
                    data = json.loads(line.strip())
                    text_content = data.get('text', '').strip()
                    
                    if text_content and len(text_content) > 10:  # Filter very short texts
                        batch.append(TamilText(
                            text=text_content,
                            source="Thamizhmalai"
                        ))
                        processed += 1
                    
                    # Insert in batches
                    if len(batch) >= batch_size:
                        db_session.bulk_save_objects(batch)
                        db_session.commit()
                        batch = []
                        
                        if processed % 10000 == 0:
                            logger.info(f"Inserted {processed} Tamil texts into database...")
                
                except json.JSONDecodeError:
                    continue
        
        # Insert remaining batch
        if batch:
            db_session.bulk_save_objects(batch)
            db_session.commit()
        
        logger.info(f"✅ Successfully inserted {processed} Tamil texts into database!")
        
    except Exception as e:
        logger.error(f"Error populating database: {e}")
        db_session.rollback()
        raise


def main():
    """Main function for database-enhanced training."""
    logger.info("🗄️ Starting Database-Enhanced Tamil GPT Training")
    
    # Initialize database
    logger.info("Initializing database...")
    db = init_database()
    db_session = db.get_session()
    
    try:
        # Populate database with Tamil texts (if not already done)
        populate_database_with_tamil_texts(db_session, limit=50000)  # Limit for testing
        
        # Create training configuration
        config = TamilTrainingConfig(
            base_model_name="microsoft/DialoGPT-small",  # Start with small model
            output_dir="./models/tamil-gpt-db",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_samples=10000,  # Start with subset
            use_custom_tokenizer=True
        )
        
        # Initialize database trainer
        trainer = DatabaseTamilTrainer(config, db_session)
        
        # Setup database tracking
        trainer.setup_database_training()
        
        # Run training pipeline
        logger.info("🚂 Starting database-enhanced training...")
        trainer.setup_tokenizer()
        trainer.setup_model()
        trainer.load_and_prepare_data()
        trainer.train()
        
        # Complete training
        trainer.complete_training(final_performance=0.95)  # Placeholder score
        
        logger.info("✅ Database-enhanced training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        db_session.close()


if __name__ == "__main__":
    main()

