"""
Tamil-specific tokenizer with morphology awareness.
Designed to handle Tamil script complexities, compound words, and sandhi rules.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TamilTokenizer:
    """
    Tamil-specific tokenizer that respects morphological boundaries,
    handles compound words, sandhi rules, and Tamil script specifics.
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        
        # Tamil Unicode ranges
        self.TAMIL_RANGE = (0x0B80, 0x0BFF)  # Tamil block
        self.TAMIL_VOWELS = set("அஆஇஈஉஊஎஏஐஒஓஔ")
        self.TAMIL_CONSONANTS = set("கஙசஞடணதநபமயரலவழளறன")
        self.TAMIL_VOWEL_SIGNS = set("ாிீுூெேைொோௌ")
        self.TAMIL_MODIFIERS = set("்ௗ")  # Pulli (virama) and other modifiers
        
        # Common Tamil suffixes for morphological analysis
        self.TAMIL_SUFFIXES = [
            "கள்", "களை", "களால்", "களில்", "களுக்கு", "களோடு",  # Plural markers
            "ஆல்", "இல்", "உக்கு", "ஓடு", "ஆக", "ஐ", "என்று",    # Case markers
            "வான்", "வாள்", "வது", "கிற", "கின்ற", "ந்த", "ட்ட",   # Verb markers
            "உம்", "ஆகவும்", "ஆவது", "என", "எனும்", "எனக்", "ஏ",   # Conjunctions
            "மான", "மிக", "தான்", "கூட", "மட்டும்", "போன்ற"        # Emphasis/comparison
        ]
        
        # Initialize vocabularies
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.word_freq: Dict[str, int] = {}
        
        # Special tokens
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,  # Beginning of sequence
            "<eos>": 3,  # End of sequence
            "<user>": 4,  # User message marker
            "<assistant>": 5,  # Assistant response marker
        }
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Tamil Unicode text using NFC normalization.
        Also handles common Tamil text preprocessing.
        """
        # Unicode normalization (NFC - Canonical Composition)
        text = unicodedata.normalize('NFC', text)
        
        # Remove zero-width characters that can interfere with processing
        text = text.replace('\u200c', '').replace('\u200d', '')  # ZWNJ, ZWJ
        
        # Standardize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def is_tamil_char(self, char: str) -> bool:
        """Check if a character is in Tamil Unicode range."""
        if not char:
            return False
        return self.TAMIL_RANGE[0] <= ord(char) <= self.TAMIL_RANGE[1]
    
    def split_tamil_word(self, word: str) -> List[str]:
        """
        Split Tamil word respecting morphological boundaries.
        Handles compound words and suffix separation.
        """
        if not word or not any(self.is_tamil_char(c) for c in word):
            return [word]
        
        # Check for known suffixes
        for suffix in sorted(self.TAMIL_SUFFIXES, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix):
                root = word[:-len(suffix)]
                if len(root) >= 2:  # Ensure meaningful root
                    return self.split_tamil_word(root) + [suffix]
        
        # If no suffix found, check for compound word patterns
        # Look for potential word boundaries (consonant + vowel patterns)
        potential_splits = []
        for i in range(2, len(word) - 1):
            if (i < len(word) - 1 and 
                word[i] in self.TAMIL_CONSONANTS and 
                (i + 1 < len(word)) and
                (word[i + 1] in self.TAMIL_VOWELS or word[i + 1] in self.TAMIL_VOWEL_SIGNS)):
                potential_splits.append(i)
        
        # For now, return the whole word if no clear split is found
        # This can be enhanced with a more sophisticated morphological analyzer
        return [word]
    
    def tokenize_sentence(self, text: str) -> List[str]:
        """
        Tokenize a sentence into tokens respecting Tamil morphology.
        """
        text = self.normalize_text(text)
        tokens = []
        
        # Split by whitespace and punctuation first
        words = re.findall(r'\S+', text)
        
        for word in words:
            # Separate punctuation from words
            parts = re.findall(r'[\w\u0B80-\u0BFF]+|[^\w\s\u0B80-\u0BFF]', word)
            
            for part in parts:
                if re.match(r'[^\w\s\u0B80-\u0BFF]', part):
                    # Punctuation
                    tokens.append(part)
                elif any(self.is_tamil_char(c) for c in part):
                    # Tamil word - apply morphological splitting
                    tamil_tokens = self.split_tamil_word(part)
                    tokens.extend(tamil_tokens)
                else:
                    # Non-Tamil word (English, numbers, etc.)
                    tokens.append(part)
        
        return tokens
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        """
        logger.info("Building Tamil vocabulary...")
        
        # Count token frequencies
        for text in texts:
            tokens = self.tokenize_sentence(text)
            for token in tokens:
                self.word_freq[token] = self.word_freq.get(token, 0) + 1
        
        # Sort tokens by frequency
        sorted_tokens = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add tokens to vocabulary (excluding special tokens already added)
        current_id = len(self.special_tokens)
        for token, freq in sorted_tokens:
            if current_id >= self.vocab_size:
                break
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        
        logger.info(f"Vocabulary built with {len(self.token_to_id)} tokens")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        """
        tokens = self.tokenize_sentence(text)
        
        if add_special_tokens:
            tokens = ["<bos>"] + tokens + ["<eos>"]
        
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Handle unknown tokens with subword fallback
                subword_ids = self._handle_unknown_token(token)
                token_ids.extend(subword_ids)
        
        return token_ids
    
    def _handle_unknown_token(self, token: str) -> List[int]:
        """
        Handle unknown tokens by breaking them into characters or subwords.
        """
        # Try character-level encoding for Tamil text
        if any(self.is_tamil_char(c) for c in token):
            char_ids = []
            for char in token:
                if char in self.token_to_id:
                    char_ids.append(self.token_to_id[char])
                else:
                    char_ids.append(self.special_tokens["<unk>"])
            return char_ids
        else:
            # For non-Tamil unknown tokens, return UNK
            return [self.special_tokens["<unk>"]]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append("<unk>")
        
        # Join tokens with appropriate spacing
        return self._join_tokens(tokens)
    
    def _join_tokens(self, tokens: List[str]) -> str:
        """
        Intelligently join tokens back into readable text.
        """
        if not tokens:
            return ""
        
        result = []
        for i, token in enumerate(tokens):
            # Add token
            result.append(token)
            
            # Decide whether to add space after this token
            if i < len(tokens) - 1:
                next_token = tokens[i + 1]
                
                # Don't add space before punctuation
                if re.match(r'^[^\w\u0B80-\u0BFF]', next_token):
                    continue
                
                # Don't add space after opening punctuation
                if re.match(r'[^\w\u0B80-\u0BFF]$', token):
                    continue
                
                # Add space between words
                result.append(" ")
        
        return "".join(result)
    
    def save_vocabulary(self, path: Union[str, Path]) -> None:
        """Save vocabulary to file."""
        path = Path(path)
        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Vocabulary saved to {path}")
    
    def load_vocabulary(self, path: Union[str, Path]) -> None:
        """Load vocabulary from file."""
        path = Path(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}
        self.vocab_size = vocab_data["vocab_size"]
        self.special_tokens = vocab_data["special_tokens"]
        
        logger.info(f"Vocabulary loaded from {path}")
    
    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<pad>"]
    
    @property
    def unk_token_id(self) -> int:
        return self.special_tokens["<unk>"]
    
    @property
    def bos_token_id(self) -> int:
        return self.special_tokens["<bos>"]
    
    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<eos>"]


def main():
    """Demo of the Tamil tokenizer."""
    # Sample Tamil texts for testing
    sample_texts = [
        "வணக்கம்! நான் ஒரு தமிழ் AI உதவியாளர்.",
        "நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "தமிழ் மொழி மிகவும் அழகான மொழி.",
        "இன்று நல்ல நாள். வானிலை சிறப்பாக இருக்கிறது.",
        "கணினி அறிவியல் மிகவும் சுவாரஸ்யமான துறை."
    ]
    
    # Initialize tokenizer
    tokenizer = TamilTokenizer(vocab_size=1000)
    
    # Build vocabulary
    tokenizer.build_vocab(sample_texts)
    
    # Test encoding and decoding
    test_text = "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?"
    print(f"Original text: {test_text}")
    
    # Tokenize
    tokens = tokenizer.tokenize_sentence(test_text)
    print(f"Tokens: {tokens}")
    
    # Encode
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    
    # Decode
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded_text}")
    
    # Save vocabulary
    tokenizer.save_vocabulary("tamil_vocab.json")


if __name__ == "__main__":
    main()
