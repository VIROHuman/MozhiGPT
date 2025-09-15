# 🇹🇦 Tamil Language Model (TLM) 1.0

A simple, effective Tamil language model built on top of existing pre-trained models, optimized for A100 GPU.

## ✨ Features

- **Tamil Conversations**: Natural Tamil chat and dialogue
- **Poetry Generation**: Create beautiful Tamil poetry on any theme
- **Story Generation**: Generate Tamil stories and narratives
- **Translation**: English to Tamil translation
- **Concept Explanation**: Explain Tamil concepts and words
- **FastAPI Server**: RESTful API for easy integration
- **A100 Optimized**: Optimized for NVIDIA A100 GPU

## 🚀 Quick Start

### 1. Install Requirements

```bash
pip install -r requirements_tlm.txt
```

### 2. Start TLM 1.0

```bash
python start_tlm.py
```

Choose option 1 to start the API server or option 2 for direct testing.

### 3. Test the Model

```bash
python test_tlm.py
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Tamil Chat
```bash
POST /chat
{
    "message": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
    "max_length": 120,
    "temperature": 0.7
}
```

### Poetry Generation
```bash
POST /poetry
{
    "theme": "அன்பு",
    "max_length": 150,
    "temperature": 0.8
}
```

### Story Generation
```bash
POST /story
{
    "topic": "பழைய காலம்",
    "max_length": 200,
    "temperature": 0.8
}
```

### Translation
```bash
POST /translate
{
    "text": "Hello, how are you?",
    "max_length": 100,
    "temperature": 0.5
}
```

### Concept Explanation
```bash
POST /explain
{
    "concept": "தமிழ் மொழி",
    "max_length": 150,
    "temperature": 0.6
}
```

## 🔧 Usage Examples

### Direct Python Usage

```python
from tlm_1_0 import TamilLanguageModel

# Initialize TLM
tlm = TamilLanguageModel()

# Chat in Tamil
response = tlm.chat("வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?")
print(response)

# Generate poetry
poetry = tlm.generate_poetry("அன்பு")
print(poetry)

# Generate story
story = tlm.generate_story("பழைய காலம்")
print(story)
```

### API Usage

```python
import requests

# Chat with TLM
response = requests.post("http://localhost:8000/chat", json={
    "message": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?"
})
print(response.json()["response"])
```

## 🏗️ Architecture

- **Base Model**: Microsoft DialoGPT-medium
- **Framework**: PyTorch + Transformers
- **API**: FastAPI
- **Optimization**: A100 GPU optimized with mixed precision

## 📁 Project Structure

```
MozhiGPT/
├── tlm_1_0.py              # Main TLM class
├── api_server.py           # FastAPI server
├── test_tlm.py            # Test script
├── start_tlm.py           # Startup script
├── requirements_tlm.txt   # Requirements
└── README_TLM.md         # This file
```

## 🎯 Use Cases

- **Tamil Chatbots**: Build conversational AI in Tamil
- **Content Creation**: Generate Tamil poetry, stories, and content
- **Language Learning**: Help with Tamil language learning
- **Translation**: English to Tamil translation
- **Cultural Preservation**: Generate Tamil cultural content

## 🔧 Configuration

The model can be configured by modifying the `TamilLanguageModel` class:

```python
# Change base model
tlm = TamilLanguageModel(model_name="microsoft/DialoGPT-large")

# Adjust generation parameters
response = tlm.generate_response(
    prompt="Your prompt",
    max_length=200,
    temperature=0.8
)
```

## 🚀 Performance

- **GPU Memory**: ~2-4GB VRAM
- **Response Time**: <2 seconds per request
- **Model Size**: ~345M parameters
- **Optimized for**: NVIDIA A100 GPU

## 📝 License

This project is part of MozhiGPT and follows the same license terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📞 Support

For support and questions, please open an issue in the repository.

---

**Tamil Language Model (TLM) 1.0** - Making Tamil AI accessible and effective! 🇹🇦
