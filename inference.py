"""
Tamil GPT Inference Pipeline

This module provides inference capabilities for the Tamil language model,
including chat history management, streaming responses, and Tamil text generation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Generator, Any
from dataclasses import dataclass, field
import warnings

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    GenerationConfig, TextStreamer
)
from peft import PeftModel, PeftConfig
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
class InferenceConfig:
    """Configuration for Tamil model inference."""
    
    # Model paths
    model_path: str = "./models/tamil-gpt"
    tokenizer_path: Optional[str] = None  # If None, uses model_path
    use_custom_tokenizer: bool = True
    custom_tokenizer_vocab_path: str = "./tamil_tokenizers/tamil_vocab.json"
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Chat parameters
    max_history_length: int = 10  # Number of previous exchanges to remember
    context_window: int = 512    # Maximum tokens in context
    
    # Device settings
    device: str = "auto"  # "auto", "cpu", "cuda"
    torch_dtype: str = "float16"  # "float16", "float32", "bfloat16"
    
    # Performance
    use_cache: bool = True
    batch_size: int = 1


class ChatHistory:
    """Manages conversation history for contextual responses."""
    
    def __init__(self, max_length: int = 10):
        self.max_length = max_length
        self.history: List[Dict[str, str]] = []
    
    def add_exchange(self, user_message: str, assistant_response: str) -> None:
        """Add a user-assistant exchange to history."""
        self.history.append({
            "user": user_message,
            "assistant": assistant_response
        })
        
        # Keep only the last max_length exchanges
        if len(self.history) > self.max_length:
            self.history = self.history[-self.max_length:]
    
    def get_context(self, include_system_prompt: bool = True) -> str:
        """Get formatted conversation context."""
        context_parts = []
        
        if include_system_prompt:
            system_prompt = (
                "நீங்கள் ஒரு உதவிகரமான தமிழ் AI உதவியாளர். "
                "தமிழில் இயற்கையான, பண்பான, மற்றும் துல்லியமான பதில்களை அளிக்கவும்."
            )
            context_parts.append(f"<system>{system_prompt}")
        
        # Add conversation history
        for exchange in self.history:
            context_parts.append(f"<user>{exchange['user']}")
            context_parts.append(f"<assistant>{exchange['assistant']}")
        
        return "".join(context_parts)
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.history.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary for serialization."""
        return {
            "max_length": self.max_length,
            "history": self.history
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load history from dictionary."""
        self.max_length = data.get("max_length", 10)
        self.history = data.get("history", [])


class TamilTextStreamer:
    """Custom text streamer for Tamil text generation."""
    
    def __init__(self, tokenizer: Union[AutoTokenizer, TamilTokenizer], skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.token_cache = []
        self.print_len = 0
    
    def put(self, value: torch.Tensor) -> str:
        """Process new tokens and return the decodable text."""
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
        
        if value.dim() == 0:
            value = value.unsqueeze(0)
        
        self.token_cache.extend(value.tolist())
        
        # Decode the cached tokens
        if hasattr(self.tokenizer, 'decode'):
            # Hugging Face tokenizer
            text = self.tokenizer.decode(self.token_cache, skip_special_tokens=self.skip_special_tokens)
        else:
            # Custom Tamil tokenizer
            text = self.tokenizer.decode(self.token_cache, skip_special_tokens=self.skip_special_tokens)
        
        # Return only the new part
        printable_text = text[self.print_len:]
        self.print_len = len(text)
        
        return printable_text
    
    def end(self) -> str:
        """Called when generation ends."""
        # Return any remaining text
        if hasattr(self.tokenizer, 'decode'):
            text = self.tokenizer.decode(self.token_cache, skip_special_tokens=self.skip_special_tokens)
        else:
            text = self.tokenizer.decode(self.token_cache, skip_special_tokens=self.skip_special_tokens)
        
        printable_text = text[self.print_len:]
        self.token_cache = []
        self.print_len = 0
        return printable_text


class TamilGPTInference:
    """Main inference class for Tamil GPT model."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = self._get_device()
        self.torch_dtype = self._get_torch_dtype()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        # Chat management
        self.chat_history = ChatHistory(max_length=config.max_history_length)
        
        logger.info(f"Initialized Tamil GPT Inference on {self.device}")
    
    def _get_device(self) -> torch.device:
        """Determine the appropriate device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")  # Apple Silicon
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get the appropriate torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        return dtype_map.get(self.config.torch_dtype, torch.float16)
    
    def load_model(self) -> None:
        """Load the trained Tamil model and tokenizer."""
        logger.info("Loading Tamil GPT model...")
        
        # Load tokenizer
        if self.config.use_custom_tokenizer:
            logger.info("Loading custom Tamil tokenizer...")
            self.tokenizer = TamilTokenizer()
            
            if os.path.exists(self.config.custom_tokenizer_vocab_path):
                self.tokenizer.load_vocabulary(self.config.custom_tokenizer_vocab_path)
            else:
                logger.warning(f"Custom tokenizer vocab not found at {self.config.custom_tokenizer_vocab_path}")
                logger.info("Falling back to Hugging Face tokenizer...")
                self.config.use_custom_tokenizer = False
        
        if not self.config.use_custom_tokenizer:
            tokenizer_path = self.config.tokenizer_path or self.config.model_path
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logger.error(f"Error loading tokenizer: {e}")
                raise
        
        # Load model
        try:
            if os.path.exists(os.path.join(self.config.model_path, "adapter_config.json")):
                # Load LoRA adapter
                logger.info("Loading LoRA adapter...")
                peft_config = PeftConfig.from_pretrained(self.config.model_path)
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    peft_config.base_model_name_or_path,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device.type == "cuda" else None,
                    trust_remote_code=True
                )
                
                # Load adapter
                self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
                
            else:
                # Load full model
                logger.info("Loading full model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device.type == "cuda" else None,
                    trust_remote_code=True
                )
            
            if self.device.type != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to base model for testing...")
            
            # Fallback to a base model
            base_model_name = "microsoft/DialoGPT-medium"
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device.type == "cuda" else None,
            )
            
            if not self.config.use_custom_tokenizer:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if self.device.type != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
        
        # Setup generation config
        self._setup_generation_config()
        
        logger.info("Model loaded successfully!")
    
    def _setup_generation_config(self) -> None:
        """Setup generation configuration."""
        # Determine token IDs
        if self.config.use_custom_tokenizer:
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
        
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=self.config.do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=self.config.use_cache,
        )
    
    def _prepare_input(self, user_message: str, include_history: bool = True) -> str:
        """Prepare input text with conversation context."""
        if include_history:
            context = self.chat_history.get_context()
            full_input = f"{context}<user>{user_message}<assistant>"
        else:
            full_input = f"<user>{user_message}<assistant>"
        
        return full_input
    
    def _tokenize_input(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text."""
        if self.config.use_custom_tokenizer:
            # Use custom tokenizer
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Truncate if too long
            if len(token_ids) > self.config.context_window - self.config.max_new_tokens:
                token_ids = token_ids[-(self.config.context_window - self.config.max_new_tokens):]
            
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)
            
        else:
            # Use Hugging Face tokenizer
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.context_window - self.config.max_new_tokens,
                padding=False
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def generate_response(
        self,
        user_message: str,
        include_history: bool = True,
        update_history: bool = True
    ) -> str:
        """Generate a response to user message."""
        
        # Prepare input
        input_text = self._prepare_input(user_message, include_history)
        inputs = self._tokenize_input(input_text)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode response
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        
        if self.config.use_custom_tokenizer:
            response = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up response
        response = self._clean_response(response)
        
        # Update chat history
        if update_history:
            self.chat_history.add_exchange(user_message, response)
        
        return response
    
    def generate_response_stream(
        self,
        user_message: str,
        include_history: bool = True,
        update_history: bool = True
    ) -> Generator[str, None, None]:
        """Generate streaming response to user message."""
        
        # Prepare input
        input_text = self._prepare_input(user_message, include_history)
        inputs = self._tokenize_input(input_text)
        
        # Setup streamer
        streamer = TamilTextStreamer(self.tokenizer, skip_special_tokens=True)
        
        # Generate with streaming
        full_response = ""
        
        with torch.no_grad():
            input_length = inputs["input_ids"].shape[1]
            
            for _ in range(self.config.max_new_tokens):
                # Generate next token
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                
                # Apply temperature and sampling
                if self.config.do_sample:
                    logits = logits / self.config.temperature
                    
                    # Top-k filtering
                    if self.config.top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, self.config.top_k)
                        logits = torch.full_like(logits, float('-inf'))
                        logits.scatter_(0, top_k_indices, top_k_logits)
                    
                    # Top-p (nucleus) filtering
                    if self.config.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > self.config.top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = False
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits.scatter_(0, indices_to_remove, float('-inf'))
                    
                    # Sample next token
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    # Greedy sampling
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Check for end of sequence
                if self.config.use_custom_tokenizer:
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                else:
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                # Add token to streamer and get new text
                new_text = streamer.put(next_token)
                if new_text:
                    yield new_text
                    full_response += new_text
                
                # Update inputs for next iteration
                next_token = next_token.unsqueeze(0)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
                inputs["attention_mask"] = torch.cat([
                    inputs["attention_mask"],
                    torch.ones((1, 1), device=self.device)
                ], dim=1)
        
        # Get any remaining text
        remaining_text = streamer.end()
        if remaining_text:
            yield remaining_text
            full_response += remaining_text
        
        # Update chat history
        if update_history:
            full_response = self._clean_response(full_response)
            self.chat_history.add_exchange(user_message, full_response)
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response."""
        # Remove special tokens
        for token in ["<user>", "<assistant>", "<system>", "<bos>", "<eos>", "<pad>"]:
            response = response.replace(token, "")
        
        # Clean up whitespace
        response = response.strip()
        
        # Remove repetitive patterns (simple heuristic)
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip() and (not cleaned_lines or line.strip() != cleaned_lines[-1].strip()):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.chat_history.clear()
        logger.info("Conversation history reset")
    
    def save_conversation(self, filepath: str) -> None:
        """Save conversation history to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.chat_history.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Conversation saved to {filepath}")
    
    def load_conversation(self, filepath: str) -> None:
        """Load conversation history from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.chat_history.from_dict(data)
        logger.info(f"Conversation loaded from {filepath}")


def main():
    """Demo of the Tamil GPT inference."""
    # Configuration
    config = InferenceConfig(
        model_path="./models/tamil-gpt",
        use_custom_tokenizer=True,
        max_new_tokens=128,
        temperature=0.7
    )
    
    # Initialize inference
    tamil_gpt = TamilGPTInference(config)
    
    try:
        # Load model
        tamil_gpt.load_model()
        
        # Interactive chat loop
        print("🇹🇦 Tamil GPT Chat Bot")
        print("தமிழ் GPT உரையாடல் பொத் ")
        print("Type 'quit' to exit / வெளியேற 'quit' என்று தட்டச்சு செய்யவும்")
        print("-" * 50)
        
        while True:
            # Get user input
            user_input = input("\n👤 நீங்கள்: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("🤖 MozhiGPT: வணக்கம்! மீண்டும் சந்திப்போம்!")
                break
            
            if not user_input:
                continue
            
            # Generate response
            print("🤖 MozhiGPT: ", end="", flush=True)
            
            response = ""
            for token in tamil_gpt.generate_response_stream(user_input):
                print(token, end="", flush=True)
                response += token
            
            print()  # New line after response
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        print(f"Error: {e}")
        print("Make sure you have trained the model first by running: python train.py")


if __name__ == "__main__":
    main()
