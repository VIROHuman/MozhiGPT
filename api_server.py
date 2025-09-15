#!/usr/bin/env python3
"""
TLM 1.0 API Server
FastAPI server for Tamil Language Model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import uvicorn
import logging

from tlm_1_0 import TamilLanguageModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize TLM
tlm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize TLM on startup"""
    global tlm
    try:
        logger.info("🚀 Starting TLM 1.0 API Server...")
        tlm = TamilLanguageModel()
        logger.info("✅ TLM 1.0 API Server ready!")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize TLM: {e}")
        raise

# Initialize FastAPI app
app = FastAPI(
    title="Tamil Language Model (TLM) 1.0 API",
    description="A simple and effective Tamil language model API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    max_length: Optional[int] = 120
    temperature: Optional[float] = 0.7

class PoetryRequest(BaseModel):
    theme: str = "அன்பு"
    max_length: Optional[int] = 150
    temperature: Optional[float] = 0.8

class StoryRequest(BaseModel):
    topic: str = "கதை"
    max_length: Optional[int] = 200
    temperature: Optional[float] = 0.8

class TranslateRequest(BaseModel):
    text: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.5

class ExplainRequest(BaseModel):
    concept: str
    max_length: Optional[int] = 150
    temperature: Optional[float] = 0.6

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Tamil Language Model (TLM) 1.0 API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": tlm is not None}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with TLM in Tamil"""
    try:
        response = tlm.chat(request.message)
        return {
            "message": request.message,
            "response": response,
            "model": "TLM 1.0"
        }
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/poetry")
async def generate_poetry(request: PoetryRequest):
    """Generate Tamil poetry"""
    try:
        poetry = tlm.generate_poetry(request.theme)
        return {
            "theme": request.theme,
            "poetry": poetry,
            "model": "TLM 1.0"
        }
    except Exception as e:
        logger.error(f"Error generating poetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/story")
async def generate_story(request: StoryRequest):
    """Generate Tamil story"""
    try:
        story = tlm.generate_story(request.topic)
        return {
            "topic": request.topic,
            "story": story,
            "model": "TLM 1.0"
        }
    except Exception as e:
        logger.error(f"Error generating story: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate_to_tamil(request: TranslateRequest):
    """Translate English to Tamil"""
    try:
        translation = tlm.translate_to_tamil(request.text)
        return {
            "original": request.text,
            "translation": translation,
            "model": "TLM 1.0"
        }
    except Exception as e:
        logger.error(f"Error translating: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain_concept(request: ExplainRequest):
    """Explain Tamil concept"""
    try:
        explanation = tlm.explain_tamil_concept(request.concept)
        return {
            "concept": request.concept,
            "explanation": explanation,
            "model": "TLM 1.0"
        }
    except Exception as e:
        logger.error(f"Error explaining concept: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
