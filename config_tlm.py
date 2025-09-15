#!/usr/bin/env python3
"""
Configuration for Tamil Language Model (TLM) 1.0
"""

# Model Configuration
MODEL_CONFIG = {
    "base_model": "microsoft/DialoGPT-medium",
    "device": "cuda",  # Will auto-detect if CUDA is available
    "torch_dtype": "float16",  # Use float16 for A100 optimization
    "device_map": "auto"  # Auto device mapping for multi-GPU
}

# Generation Parameters
GENERATION_CONFIG = {
    "default_max_length": 120,
    "default_temperature": 0.7,
    "poetry_max_length": 150,
    "poetry_temperature": 0.8,
    "story_max_length": 200,
    "story_temperature": 0.8,
    "translation_max_length": 100,
    "translation_temperature": 0.5,
    "explanation_max_length": 150,
    "explanation_temperature": 0.6
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info"
}

# Tamil-specific Configuration
TAMIL_CONFIG = {
    "default_theme": "அன்பு",
    "default_topic": "கதை",
    "conversation_prefix": "Tamil conversation:",
    "poetry_prefix": "Tamil poetry about",
    "story_prefix": "Tamil story about",
    "translation_prefix": "Translate to Tamil:",
    "explanation_prefix": "Explain Tamil concept"
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
