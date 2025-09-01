# 🇹🇦 MozhiGPT - Tamil-First ChatGPT

**MozhiGPT** is a Tamil-first conversational AI system built for natural, culturally accurate, and high-quality Tamil language interactions. Inspired by the Tamil AI community and designed to solve the problem of incoherent Tamil AI responses.

## ✨ Features

### 🔹 Core Capabilities
- **Tamil-First Design**: Optimized for Tamil language understanding and generation
- **Custom Tokenizer**: Morphology-aware tokenization respecting Tamil word boundaries
- **LoRA Fine-tuning**: Efficient training on Mac M1 and other hardware
- **Streaming Responses**: Real-time chat experience
- **Chat History**: Contextual conversations with memory
- **RESTful API**: Easy integration with frontends
- **WebSocket Support**: Real-time bidirectional communication

### 🔹 Technical Stack
- **Base Models**: LLaMA 3, Mistral, or Qwen (configurable)
- **Fine-tuning**: LoRA/PEFT for efficient training
- **Dataset**: Thamizhmalai + additional Tamil corpora
- **Backend**: FastAPI with async support
- **Deployment**: Docker containerization
- **Hardware**: Optimized for both GPU and CPU inference

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 8GB+ RAM (16GB+ recommended)
- CUDA GPU (optional, but recommended)
- Docker (for containerized deployment)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MozhiGPT.git
cd MozhiGPT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp env.example .env

# Edit configuration (optional)
nano .env
```

### 3. Training the Model

```bash
# Train Tamil GPT model
python train.py

# This will:
# - Download Thamizhmalai dataset
# - Build Tamil vocabulary
# - Fine-tune model with LoRA
# - Save model to ./models/tamil-gpt/
```

### 4. Start the API Server

```bash
# Start the FastAPI server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Or use the direct script
python api/main.py
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Chat endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?"}'
```

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Build and run
docker build -t mozhigpt .
docker run -p 8000:8000 mozhigpt

# Or use docker-compose
docker-compose up mozhigpt-api
```

### Development Mode

```bash
# Start in development mode with hot reload
docker-compose --profile dev up mozhigpt-dev
```

### Production Deployment

```bash
# Start production stack
docker-compose up -d mozhigpt-api

# With Redis and Nginx (optional)
docker-compose --profile redis --profile nginx up -d
```

## 📡 API Reference

### REST Endpoints

#### Chat
```http
POST /chat
Content-Type: application/json

{
  "message": "வணக்கம்!",
  "conversation_id": "optional-uuid",
  "include_history": true,
  "stream": false,
  "max_tokens": 256,
  "temperature": 0.7
}
```

#### Streaming Chat
```http
POST /chat/stream
Content-Type: application/json

{
  "message": "தமிழ் மொழி பற்றி சொல்லுங்கள்"
}
```

#### Health Check
```http
GET /health
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/chat/ws');

ws.send(JSON.stringify({
  message: "வணக்கம்!",
  conversation_id: "optional-uuid"
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.type, data.data);
};
```

## 🛠️ Development

### Project Structure

```
MozhiGPT/
├── api/                    # FastAPI backend
│   └── main.py            # API server
├── tamil_tokenizers/      # Custom Tamil tokenizer
│   └── tamil_tokenizer.py
├── models/                # Trained models
│   └── tamil-gpt/        # LoRA fine-tuned model
├── data/                  # Training data
│   └── Thamizhmalai/     # Dataset storage
├── train.py              # Training pipeline
├── inference.py          # Inference engine
├── requirements.txt      # Dependencies
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Multi-service setup
└── README.md           # This file
```

### Custom Tokenizer

The Tamil tokenizer is designed to handle:
- **Unicode Normalization**: NFC normalization for Tamil text
- **Morphological Awareness**: Respects Tamil word boundaries
- **Suffix Handling**: Recognizes common Tamil suffixes
- **Compound Words**: Handles Tamil compound word structures
- **Sandhi Rules**: Basic support for sandhi transformations

```python
from tamil_tokenizers.tamil_tokenizer import TamilTokenizer

tokenizer = TamilTokenizer(vocab_size=32000)
tokenizer.build_vocab(tamil_texts)

# Tokenize
tokens = tokenizer.tokenize_sentence("வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?")
token_ids = tokenizer.encode("வணக்கம்!")
text = tokenizer.decode(token_ids)
```

### Training Configuration

Key training parameters in `train.py`:

```python
@dataclass
class TamilTrainingConfig:
    base_model_name: str = "microsoft/DialoGPT-medium"
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
```

### Inference Configuration

Key inference parameters in `inference.py`:

```python
@dataclass
class InferenceConfig:
    model_path: str = "./models/tamil-gpt"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    use_custom_tokenizer: bool = True
```

## 🌐 Hosting Options

### Local Development
```bash
python api/main.py
# Access at http://localhost:8000
```

### Docker + Cloud VM
```bash
# On any cloud VM (AWS EC2, GCP Compute, Azure VM)
docker run -p 8000:8000 mozhigpt
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mozhigpt
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mozhigpt
  template:
    metadata:
      labels:
        app: mozhigpt
    spec:
      containers:
      - name: mozhigpt
        image: mozhigpt:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEVICE
          value: "cpu"
```

### Cloud Platforms

#### AWS ECS/Fargate
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag mozhigpt:latest <account>.dkr.ecr.us-east-1.amazonaws.com/mozhigpt:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/mozhigpt:latest
```

#### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy mozhigpt \
  --image gcr.io/PROJECT_ID/mozhigpt \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

#### Vercel (API Routes)
```javascript
// api/chat.js
import { TamilGPTInference } from '../inference.py';

export default async function handler(req, res) {
  const response = await tamil_gpt.generate_response(req.body.message);
  res.json({ response });
}
```

## 🔧 Advanced Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./models/tamil-gpt` | Path to trained model |
| `USE_CUSTOM_TOKENIZER` | `true` | Use custom Tamil tokenizer |
| `MAX_NEW_TOKENS` | `256` | Maximum tokens to generate |
| `TEMPERATURE` | `0.7` | Generation temperature |
| `DEVICE` | `auto` | Device: auto, cpu, cuda, mps |
| `PORT` | `8000` | API server port |

### Performance Tuning

#### For CPU Inference
```python
config = InferenceConfig(
    device="cpu",
    torch_dtype="float32",
    max_new_tokens=128,
    batch_size=1
)
```

#### For GPU Inference
```python
config = InferenceConfig(
    device="cuda",
    torch_dtype="float16",
    max_new_tokens=256,
    batch_size=4
)
```

#### For Apple Silicon (M1/M2)
```python
config = InferenceConfig(
    device="mps",
    torch_dtype="float16",
    max_new_tokens=256
)
```

## 🎯 Future Enhancements

### Phase 2: RAG Integration
- [ ] Tamil knowledge base integration
- [ ] Vector database for semantic search
- [ ] Document Q&A capabilities

### Phase 3: Multimodal
- [ ] Speech-to-text (Whisper fine-tuned for Tamil)
- [ ] Text-to-speech (Coqui TTS/Festival)
- [ ] Image understanding with Tamil descriptions

### Phase 4: Community Features
- [ ] Bias correction with community feedback
- [ ] Fine-tuning on domain-specific data
- [ ] Tamil dialect support
- [ ] Cultural context awareness

## 🤝 Contributing

We welcome contributions to MozhiGPT! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Areas for Contribution
- [ ] Better Tamil morphological analysis
- [ ] Additional Tamil datasets
- [ ] Performance optimizations
- [ ] Frontend integrations
- [ ] Documentation improvements
- [ ] Test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Tamil AI Community**: AI Tamil Nadu, Tamil Virtual Academy
- **Thamizhmalai Dataset**: [tamiltheorist/Thamizhmalai](https://huggingface.co/datasets/tamiltheorist/Thamizhmalai)
- **Hugging Face**: For the transformers library and model hub
- **Tamil Computing Community**: For language processing insights

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/MozhiGPT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/MozhiGPT/discussions)
- **Tamil AI Community**: Join our Tamil AI groups for support

---

**MozhiGPT** - Empowering Tamil conversations with AI! 🇹🇦✨
