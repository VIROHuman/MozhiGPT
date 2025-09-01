#!/usr/bin/env python3
"""
Full-scale Tamil GPT Training Script

This script is configured to train on the complete Thamizhmalai dataset
with optimizations for large-scale training.
"""

import logging
from train import TamilModelTrainer, TamilTrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_full_training_config():
    """Create configuration for full-scale training."""
    return TamilTrainingConfig(
        # Model configuration - Use a larger, more capable base model
        base_model_name="microsoft/DialoGPT-medium",  # Can upgrade to "unsloth/llama-3-8b-bnb-4bit"
        model_max_length=512,
        
        # LoRA configuration - Optimized for large-scale training
        lora_r=32,  # Increased rank for better performance
        lora_alpha=64,
        lora_dropout=0.1,
        
        # Training configuration - Production settings
        output_dir="./models/tamil-gpt-full",
        num_train_epochs=3,  # Full training epochs
        per_device_train_batch_size=2,  # Adjust based on your GPU memory
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16
        learning_rate=1e-4,  # Lower learning rate for stability
        weight_decay=0.01,
        warmup_steps=1000,  # More warmup for large dataset
        logging_steps=100,
        save_steps=5000,  # Save less frequently for large dataset
        eval_steps=5000,
        max_grad_norm=1.0,
        
        # Data configuration - Full dataset
        dataset_name="tamiltheorist/Thamizhmalai",
        max_samples=None,  # Use ALL 73.9M samples
        validation_split=0.02,  # Small validation set (2% = ~1.5M samples)
        use_streaming=True,  # Enable streaming for memory efficiency
        
        # Hardware optimization - Production settings
        use_4bit=True,  # Enable quantization for memory efficiency
        use_nested_quant=False,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        use_gradient_checkpointing=True,
        dataloader_num_workers=4,
        
        # Tamil-specific
        use_custom_tokenizer=True,
        tokenizer_vocab_size=50000,  # Larger vocab for better Tamil coverage
    )


def main():
    """Main training function for full-scale Tamil GPT."""
    logger.info("🚀 Starting FULL-SCALE Tamil GPT Training")
    logger.info("📊 Dataset: Thamizhmalai (73.9M samples, 22.8GB)")
    logger.info("🔥 Training Mode: Complete dataset with streaming")
    
    # Create full training configuration
    config = create_full_training_config()
    
    # Display training info
    print("\n" + "="*60)
    print("🇹🇦 MOZHIGPT FULL TRAINING CONFIGURATION")
    print("="*60)
    print(f"📦 Base Model: {config.base_model_name}")
    print(f"📊 Dataset: {config.dataset_name}")
    print(f"🔢 Max Samples: ALL (73.9M samples)")
    print(f"💾 Model Output: {config.output_dir}")
    print(f"🔧 LoRA Rank: {config.lora_r}")
    print(f"📈 Epochs: {config.num_train_epochs}")
    print(f"🎯 Batch Size: {config.per_device_train_batch_size}")
    print(f"📡 Streaming: {config.use_streaming}")
    print(f"🧠 Quantization: {config.use_4bit}")
    print("="*60)
    
    # Confirm before starting
    print("\n⚠️  WARNING: This will train on the COMPLETE dataset!")
    print("⏱️  Expected training time: Several hours to days (depending on hardware)")
    print("💽  Ensure you have sufficient disk space and memory")
    print("🔌  Recommend using a GPU for faster training")
    
    response = input("\n🤔 Continue with full training? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Training cancelled by user")
        return
    
    # Initialize trainer
    trainer = TamilModelTrainer(config)
    
    # Run full training pipeline
    try:
        logger.info("🔧 Setting up tokenizer...")
        trainer.setup_tokenizer()
        
        logger.info("🤖 Setting up model with LoRA...")
        trainer.setup_model()
        
        logger.info("📚 Loading Tamil dataset...")
        trainer.load_and_prepare_data()
        
        logger.info("🚂 Starting training...")
        trainer.train()
        
        print("\n" + "🎉"*20)
        print("✅ FULL TRAINING COMPLETED SUCCESSFULLY!")
        print("🎊 MozhiGPT trained on 73.9M Tamil samples!")
        print(f"💾 Model saved to: {config.output_dir}")
        print("🌐 Your API can now use the fully trained model!")
        print("🎉"*20)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"\n💡 If training failed due to memory issues:")
        print("1. Reduce batch_size in the config")
        print("2. Enable gradient checkpointing")
        print("3. Use a smaller base model")
        print("4. Set max_samples to a smaller number (e.g., 100000)")
        raise


if __name__ == "__main__":
    main()

