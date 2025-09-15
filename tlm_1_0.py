#!/usr/bin/env python3
"""
Tamil Language Model (TLM) 1.0
A simple, effective Tamil language model using existing pre-trained models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
import json
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TamilLanguageModel:
    """Tamil Language Model (TLM) 1.0 - Simple and effective Tamil AI"""
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """Initialize TLM 1.0 with an existing model."""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        logger.info(f"🚀 Initializing Tamil Language Model (TLM) 1.0")
        logger.info(f"📱 Using device: {self.device}")
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"📥 Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("✅ TLM 1.0 loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate a response to the given prompt."""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "மன்னிக்கவும், நான் பதில் உருவாக்க முடியவில்லை."
    
    def chat(self, message: str) -> str:
        """Simple chat interface for Tamil conversations."""
        # Add Tamil context to improve responses
        tamil_prompt = f"Tamil conversation: {message}"
        response = self.generate_response(tamil_prompt, max_length=120, temperature=0.7)
        return response
    
    def generate_poetry(self, theme: str = "அன்பு") -> str:
        """Generate Tamil poetry on a given theme."""
        poetry_prompt = f"Tamil poetry about {theme}:"
        response = self.generate_response(poetry_prompt, max_length=150, temperature=0.8)
        return response
    
    def generate_story(self, topic: str = "கதை") -> str:
        """Generate a Tamil story on a given topic."""
        story_prompt = f"Tamil story about {topic}:"
        response = self.generate_response(story_prompt, max_length=200, temperature=0.8)
        return response
    
    def translate_to_tamil(self, english_text: str) -> str:
        """Translate English text to Tamil."""
        translate_prompt = f"Translate to Tamil: {english_text}"
        response = self.generate_response(translate_prompt, max_length=100, temperature=0.5)
        return response
    
    def explain_tamil_concept(self, concept: str) -> str:
        """Explain a Tamil concept or word."""
        explain_prompt = f"Explain Tamil concept {concept}:"
        response = self.generate_response(explain_prompt, max_length=150, temperature=0.6)
        return response

def main():
    """Main function to test TLM 1.0"""
    print("🇹🇦 Tamil Language Model (TLM) 1.0")
    print("=" * 50)
    
    # Initialize TLM
    tlm = TamilLanguageModel()
    
    # Test conversation
    print("\n💬 Testing Tamil Conversation:")
    print("-" * 30)
    response = tlm.chat("வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?")
    print(f"User: வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?")
    print(f"TLM: {response}")
    
    # Test poetry generation
    print("\n🎭 Testing Tamil Poetry Generation:")
    print("-" * 30)
    poetry = tlm.generate_poetry("அன்பு")
    print(f"Poetry about அன்பு:")
    print(f"{poetry}")
    
    # Test story generation
    print("\n📚 Testing Tamil Story Generation:")
    print("-" * 30)
    story = tlm.generate_story("பழைய காலம்")
    print(f"Story about பழைய காலம்:")
    print(f"{story}")
    
    print("\n✅ TLM 1.0 testing completed!")

if __name__ == "__main__":
    main()
