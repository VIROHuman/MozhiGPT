#!/usr/bin/env python3
"""
Quick start script for MozhiGPT server.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_model_exists(model_path: str = "./models/tamil-gpt") -> bool:
    """Check if trained model exists."""
    model_dir = Path(model_path)
    
    # Check for either LoRA adapter or full model
    has_adapter = (model_dir / "adapter_config.json").exists()
    has_model = (model_dir / "config.json").exists()
    
    return has_adapter or has_model


def check_tokenizer_exists(tokenizer_path: str = "./tamil_tokenizers/tamil_vocab.json") -> bool:
    """Check if custom tokenizer vocabulary exists."""
    return Path(tokenizer_path).exists()


def main():
    """Main function to start MozhiGPT server."""
    parser = argparse.ArgumentParser(description="Start MozhiGPT server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--model-path", default="./models/tamil-gpt", help="Path to model")
    parser.add_argument("--skip-checks", action="store_true", help="Skip model checks")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting MozhiGPT server...")
    
    # Check if model exists (unless skipped)
    if not args.skip_checks:
        if not check_model_exists(args.model_path):
            logger.warning("⚠️ Trained model not found. The server will start but may not work properly.")
            logger.info("💡 To train the model, run: python train.py")
            
            # Ask user if they want to continue
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                logger.info("Exiting...")
                sys.exit(1)
        
        if not check_tokenizer_exists():
            logger.warning("⚠️ Custom tokenizer vocabulary not found. Using default tokenizer.")
    
    # Set environment variables
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    
    # Start server
    try:
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", args.host,
            "--port", str(args.port)
        ]
        
        if args.reload:
            cmd.append("--reload")
        
        logger.info(f"🌐 Starting server at http://{args.host}:{args.port}")
        logger.info("📚 API docs available at http://{args.host}:{args.port}/docs")
        logger.info("🔧 Press Ctrl+C to stop the server")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
