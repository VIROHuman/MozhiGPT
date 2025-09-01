"""
Tamil GPT Training Pipeline with LoRA Fine-tuning

This script implements a complete training pipeline for Tamil language model
using the Thamizhmalai dataset and LoRA (Low-Rank Adaptation) for efficient fine-tuning.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    BitsAndBytesConfig, TrainerCallback
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    PeftModel, PeftConfig, prepare_model_for_kbit_training
)
from datasets import load_dataset, Dataset as HFDataset
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

# Import our custom Tamil tokenizer
from tamil_tokenizers.tamil_tokenizer import TamilTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TamilTrainingConfig:
    """Configuration for Tamil model training."""
    
    # Model configuration
    base_model_name: str = "microsoft/DialoGPT-small"  # Smaller model for testing
    model_max_length: int = 512
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training configuration
    output_dir: str = "./models/tamil-gpt"
    num_train_epochs: int = 1  # Reduced for faster training
    per_device_train_batch_size: int = 1  # Reduced for memory efficiency
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4  # Reduced
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Data configuration
    dataset_name: str = "tamiltheorist/Thamizhmalai"
    max_samples: Optional[int] = None  # None for all 73.9M samples
    validation_split: float = 0.05  # Smaller validation for large dataset
    use_streaming: bool = True  # Enable streaming for large datasets
    
    # Quantization
    use_4bit: bool = True
    use_nested_quant: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Hardware optimization
    use_gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Tamil-specific
    use_custom_tokenizer: bool = True
    tokenizer_vocab_size: int = 32000


class TamilConversationDataset(Dataset):
    """Dataset class for Tamil conversation data."""
    
    def __init__(
        self,
        data: Union[List[Dict[str, Any]], Any],  # Support both list and streaming dataset
        tokenizer: Union[AutoTokenizer, TamilTokenizer],
        max_length: int = 512,
        use_custom_tokenizer: bool = False
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_custom_tokenizer = use_custom_tokenizer
        self.is_streaming = not isinstance(data, list)
        
        # For streaming datasets, we can't know the length ahead of time
        if self.is_streaming:
            self._length = None
        else:
            self._length = len(data)
    
    def __len__(self) -> int:
        if self.is_streaming:
            # For streaming datasets, return a large number
            # The trainer will handle the actual iteration
            return 1000000  # Placeholder for streaming
        return self._length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Format the conversation
        if 'instruction' in item and 'output' in item:
            # Instruction-response format
            text = f"<user>{item['instruction']}<assistant>{item['output']}"
        elif 'input' in item and 'output' in item:
            # Input-output format
            text = f"<user>{item['input']}<assistant>{item['output']}"
        elif 'question' in item and 'answer' in item:
            # Question-answer format
            text = f"<user>{item['question']}<assistant>{item['answer']}"
        else:
            # Fallback to text field
            text = item.get('text', str(item))
        
        # Tokenize
        if self.use_custom_tokenizer:
            # Use our custom Tamil tokenizer
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            # Pad or truncate
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
            
            input_ids = torch.tensor(tokens, dtype=torch.long)
            attention_mask = torch.tensor([1 if t != self.tokenizer.pad_token_id else 0 for t in tokens], dtype=torch.long)
        else:
            # Use Hugging Face tokenizer
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For causal LM, labels = input_ids
        }


class TamilModelTrainer:
    """Main trainer class for Tamil language model."""
    
    def __init__(self, config: TamilTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def setup_tokenizer(self) -> None:
        """Setup tokenizer (custom Tamil or Hugging Face)."""
        if self.config.use_custom_tokenizer:
            logger.info("Initializing custom Tamil tokenizer...")
            self.tokenizer = TamilTokenizer(vocab_size=self.config.tokenizer_vocab_size)
            
            # Load and prepare data for vocab building
            logger.info("Loading dataset for vocabulary building...")
            try:
                dataset = load_dataset(self.config.dataset_name, split="train")
                
                # Extract texts from dataset
                texts = []
                for item in tqdm(dataset, desc="Extracting texts"):
                    # Handle different data formats
                    if 'instruction' in item and 'output' in item:
                        texts.extend([item['instruction'], item['output']])
                    elif 'input' in item and 'output' in item:
                        texts.extend([item['input'], item['output']])
                    elif 'question' in item and 'answer' in item:
                        texts.extend([item['question'], item['answer']])
                    elif 'text' in item:
                        texts.append(item['text'])
                
                # Build vocabulary
                logger.info(f"Building vocabulary from {len(texts)} texts...")
                vocab_subset = min(5000, len(texts))  # Use smaller subset for vocab building
                self.tokenizer.build_vocab(texts[:vocab_subset])
                
                # Save tokenizer
                os.makedirs("tamil_tokenizers", exist_ok=True)
                self.tokenizer.save_vocabulary("tamil_tokenizers/tamil_vocab.json")
                
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                logger.info("Falling back to Hugging Face tokenizer...")
                self.config.use_custom_tokenizer = False
        
        if not self.config.use_custom_tokenizer:
            logger.info("Using Hugging Face tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_name,
                trust_remote_code=True
            )
            
            # Add special tokens if needed
            special_tokens = ["<user>", "<assistant>"]
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def setup_model(self) -> None:
        """Setup base model with quantization and LoRA."""
        logger.info(f"Loading base model: {self.config.base_model_name}")
        
        # Quantization configuration
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None
        
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        except Exception as e:
            logger.error(f"Error loading model {self.config.base_model_name}: {e}")
            logger.info("Falling back to DialoGPT-medium...")
            self.config.base_model_name = "microsoft/DialoGPT-medium"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Resize token embeddings if using custom tokenizer
        if self.config.use_custom_tokenizer:
            self.model.resize_token_embeddings(len(self.tokenizer.token_to_id))
        elif hasattr(self.tokenizer, 'additional_special_tokens'):
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Prepare model for training
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Setup LoRA
        logger.info("Setting up LoRA configuration...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
    def load_and_prepare_data(self) -> None:
        """Load and prepare training data."""
        logger.info("Loading and preparing Tamil dataset...")
        
        try:
            # Load dataset with streaming for large datasets
            if self.config.use_streaming and self.config.max_samples is None:
                logger.info("Loading dataset in streaming mode for large data...")
                dataset = load_dataset(self.config.dataset_name, streaming=True)
                if 'train' in dataset:
                    data = dataset['train']
                else:
                    data = dataset[list(dataset.keys())[0]]
                
                # For streaming, we'll use the dataset directly
                self.train_dataset = TamilConversationDataset(
                    data,  # Pass the streaming dataset
                    self.tokenizer,
                    max_length=self.config.model_max_length,
                    use_custom_tokenizer=self.config.use_custom_tokenizer
                )
                
                # Create a smaller eval dataset
                eval_data = list(data.take(1000))  # Take first 1000 samples for eval
                self.eval_dataset = TamilConversationDataset(
                    eval_data,
                    self.tokenizer,
                    max_length=self.config.model_max_length,
                    use_custom_tokenizer=self.config.use_custom_tokenizer
                )
                
                logger.info("Loaded streaming dataset for full training")
                return
            
            # Standard loading for smaller datasets or when max_samples is specified
            dataset = load_dataset(self.config.dataset_name)
            
            if 'train' in dataset:
                data = dataset['train']
            else:
                # If no train split, use the whole dataset
                data = dataset[list(dataset.keys())[0]]
            
            # Convert to list of dictionaries
            data_list = []
            max_items = self.config.max_samples or 50000  # Default limit for non-streaming
            
            logger.info(f"Loading up to {max_items} samples...")
            for i, item in enumerate(data):
                if i >= max_items:
                    break
                data_list.append(dict(item))
                
                # Progress logging
                if (i + 1) % 10000 == 0:
                    logger.info(f"Loaded {i + 1} samples...")
            
            logger.info(f"Loaded {len(data_list)} samples")
            
            # Split into train and validation
            split_idx = int(len(data_list) * (1 - self.config.validation_split))
            train_data = data_list[:split_idx]
            eval_data = data_list[split_idx:]
            
            logger.info(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")
            
            # Create datasets
            self.train_dataset = TamilConversationDataset(
                train_data,
                self.tokenizer,
                max_length=self.config.model_max_length,
                use_custom_tokenizer=self.config.use_custom_tokenizer
            )
            
            self.eval_dataset = TamilConversationDataset(
                eval_data,
                self.tokenizer,
                max_length=self.config.model_max_length,
                use_custom_tokenizer=self.config.use_custom_tokenizer
            )
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Create dummy data for testing
            logger.info("Creating dummy Tamil data for testing...")
            dummy_data = [
                {"instruction": "வணக்கம்", "output": "வணக்கம்! நான் தமிழ் AI உதவியாளர். உங்களுக்கு எப்படி உதவ முடியும்?"},
                {"instruction": "நீங்கள் எப்படி இருக்கிறீர்கள்?", "output": "நான் நன்றாக இருக்கிறேன். நன்றி கேட்டதற்கு!"},
                {"instruction": "தமிழ் மொழி பற்றி சொல்லுங்கள்", "output": "தமிழ் மொழி உலகின் பழமையான மொழிகளில் ஒன்று. இது மிகவும் செம்மையான இலக்கணம் கொண்டது."}
            ]
            
            self.train_dataset = TamilConversationDataset(
                dummy_data,
                self.tokenizer,
                max_length=self.config.model_max_length,
                use_custom_tokenizer=self.config.use_custom_tokenizer
            )
            
            self.eval_dataset = TamilConversationDataset(
                dummy_data[:1],
                self.tokenizer,
                max_length=self.config.model_max_length,
                use_custom_tokenizer=self.config.use_custom_tokenizer
            )
    
    def train(self) -> None:
        """Run the training process."""
        logger.info("Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard
            remove_unused_columns=False,
            dataloader_num_workers=self.config.dataloader_num_workers,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            max_grad_norm=self.config.max_grad_norm,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer if not self.config.use_custom_tokenizer else None,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        
        if not self.config.use_custom_tokenizer:
            self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info(f"Training completed! Model saved to {self.config.output_dir}")


def main():
    """Main training function."""
    # Create configuration
    config = TamilTrainingConfig()
    
    # Initialize trainer
    trainer = TamilModelTrainer(config)
    
    # Run training pipeline
    try:
        trainer.setup_tokenizer()
        trainer.setup_model()
        trainer.load_and_prepare_data()
        trainer.train()
        
        logger.info("✅ Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
