"""
MozhiGPT FastAPI Backend

This module provides REST API and WebSocket endpoints for the Tamil ChatGPT application.
Supports both standard chat responses and streaming responses.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import aiofiles

# Import our inference module
from inference import TamilGPTInference, InferenceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global inference instance
tamil_gpt: Optional[TamilGPTInference] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting MozhiGPT API...")
    await load_model()
    logger.info("✅ MozhiGPT API started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MozhiGPT API...")


app = FastAPI(
    title="MozhiGPT API",
    description="Tamil-first ChatGPT API with natural language understanding",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security (optional)
security = HTTPBearer(auto_error=False)


# Pydantic models
class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message in Tamil or English")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    include_history: bool = Field(True, description="Include conversation history")
    stream: bool = Field(False, description="Enable streaming response")
    
    # Generation parameters (optional overrides)
    max_tokens: Optional[int] = Field(None, ge=1, le=512, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.1, le=2.0, description="Generation temperature")
    top_p: Optional[float] = Field(None, ge=0.1, le=1.0, description="Top-p sampling parameter")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Generated response")
    conversation_id: str = Field(..., description="Conversation ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    model_info: Dict[str, Any] = Field(default_factory=dict, description="Model metadata")


class ConversationHistory(BaseModel):
    """Conversation history model."""
    conversation_id: str = Field(..., description="Conversation ID")
    messages: List[ChatMessage] = Field(default_factory=list, description="Chat messages")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    code: int = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now)


# Database support for conversation storage
try:
    from database import get_database, Conversation
    USE_DATABASE = True
    db = get_database()
except ImportError:
    USE_DATABASE = False
    # Fallback to in-memory storage
    conversations: Dict[str, ConversationHistory] = {}


async def load_model() -> None:
    """Load the Tamil GPT model."""
    global tamil_gpt
    
    try:
        # Configuration
        config = InferenceConfig(
            model_path=os.getenv("MODEL_PATH", "./models/tamil-gpt"),
            use_custom_tokenizer=os.getenv("USE_CUSTOM_TOKENIZER", "true").lower() == "true",
            custom_tokenizer_vocab_path=os.getenv("TOKENIZER_VOCAB_PATH", "./tamil_tokenizers/tamil_vocab.json"),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "256")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            top_k=int(os.getenv("TOP_K", "50")),
            top_p=float(os.getenv("TOP_P", "0.9")),
            device=os.getenv("DEVICE", "auto"),
        )
        
        # Initialize inference
        tamil_gpt = TamilGPTInference(config)
        tamil_gpt.load_model()
        
        logger.info("Tamil GPT model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't raise exception to allow API to start in development
        tamil_gpt = None


def get_model() -> TamilGPTInference:
    """Dependency to get the loaded model."""
    if tamil_gpt is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    return tamil_gpt


def generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    from uuid import uuid4
    return str(uuid4())


def get_or_create_conversation(conversation_id: Optional[str]) -> ConversationHistory:
    """Get existing conversation or create new one."""
    if conversation_id and conversation_id in conversations:
        return conversations[conversation_id]
    
    # Create new conversation
    new_id = conversation_id or generate_conversation_id()
    conversation = ConversationHistory(conversation_id=new_id)
    conversations[new_id] = conversation
    return conversation


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with welcome message."""
    return {
        "message": "வணக்கம்! Welcome to MozhiGPT API",
        "description": "Tamil-first ChatGPT API for natural conversations",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if tamil_gpt else "model_not_loaded",
        model_loaded=tamil_gpt is not None,
        version="1.0.0"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    model: TamilGPTInference = Depends(get_model),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    Generate a chat response.
    
    This endpoint accepts a user message and returns a generated response
    in Tamil, maintaining conversation context if a conversation_id is provided.
    """
    try:
        # Get or create conversation
        conversation = get_or_create_conversation(request.conversation_id)
        
        # Update model parameters if provided
        if request.max_tokens:
            model.config.max_new_tokens = request.max_tokens
        if request.temperature:
            model.config.temperature = request.temperature
        if request.top_p:
            model.config.top_p = request.top_p
        
        # Generate response
        response = model.generate_response(
            user_message=request.message,
            include_history=request.include_history,
            update_history=True
        )
        
        # Update conversation history
        conversation.messages.extend([
            ChatMessage(role="user", content=request.message),
            ChatMessage(role="assistant", content=response)
        ])
        conversation.updated_at = datetime.now()
        
        # Limit conversation history
        if len(conversation.messages) > 20:  # Keep last 20 messages
            conversation.messages = conversation.messages[-20:]
        
        return ChatResponse(
            response=response,
            conversation_id=conversation.conversation_id,
            model_info={
                "model_path": model.config.model_path,
                "use_custom_tokenizer": model.config.use_custom_tokenizer,
                "temperature": model.config.temperature,
                "max_tokens": model.config.max_new_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    model: TamilGPTInference = Depends(get_model),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    Generate a streaming chat response.
    
    This endpoint returns a Server-Sent Events (SSE) stream of the generated response.
    """
    try:
        # Get or create conversation
        conversation = get_or_create_conversation(request.conversation_id)
        
        # Update model parameters if provided
        if request.max_tokens:
            model.config.max_new_tokens = request.max_tokens
        if request.temperature:
            model.config.temperature = request.temperature
        if request.top_p:
            model.config.top_p = request.top_p
        
        async def generate():
            """Async generator for streaming response."""
            try:
                full_response = ""
                
                # Send conversation ID first
                yield f"data: {json.dumps({'type': 'conversation_id', 'data': conversation.conversation_id})}\n\n"
                
                # Generate streaming response
                for token in model.generate_response_stream(
                    user_message=request.message,
                    include_history=request.include_history,
                    update_history=False  # We'll update manually
                ):
                    full_response += token
                    data = {
                        "type": "token",
                        "data": token,
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete', 'data': full_response})}\n\n"
                
                # Update conversation history
                conversation.messages.extend([
                    ChatMessage(role="user", content=request.message),
                    ChatMessage(role="assistant", content=full_response)
                ])
                conversation.updated_at = datetime.now()
                
                # Limit conversation history
                if len(conversation.messages) > 20:
                    conversation.messages = conversation.messages[-20:]
                
            except Exception as e:
                error_data = {
                    "type": "error",
                    "data": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat stream endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/chat/ws")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.
    
    Protocol:
    - Client sends: {"message": "text", "conversation_id": "optional"}
    - Server sends: {"type": "token", "data": "text"} or {"type": "complete", "data": "full_response"}
    """
    await websocket.accept()
    
    try:
        if tamil_gpt is None:
            await websocket.send_json({
                "type": "error",
                "data": "Model not loaded"
            })
            return
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")
            conversation_id = data.get("conversation_id")
            
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "data": "Empty message"
                })
                continue
            
            # Get or create conversation
            conversation = get_or_create_conversation(conversation_id)
            
            # Send conversation ID
            await websocket.send_json({
                "type": "conversation_id",
                "data": conversation.conversation_id
            })
            
            # Generate streaming response
            full_response = ""
            try:
                for token in tamil_gpt.generate_response_stream(
                    user_message=message,
                    include_history=True,
                    update_history=False
                ):
                    full_response += token
                    await websocket.send_json({
                        "type": "token",
                        "data": token,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Send completion
                await websocket.send_json({
                    "type": "complete",
                    "data": full_response
                })
                
                # Update conversation history
                conversation.messages.extend([
                    ChatMessage(role="user", content=message),
                    ChatMessage(role="assistant", content=full_response)
                ])
                conversation.updated_at = datetime.now()
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "data": str(e)
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": str(e)
            })
        except:
            pass


@app.get("/conversations", response_model=List[ConversationHistory])
async def list_conversations():
    """List all conversations."""
    return list(conversations.values())


@app.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(conversation_id: str):
    """Get a specific conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversations[conversation_id]


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversations[conversation_id]
    return {"message": "Conversation deleted successfully"}


@app.post("/conversations/{conversation_id}/reset")
async def reset_conversation(conversation_id: str):
    """Reset a conversation (clear history)."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversations[conversation_id].messages.clear()
    conversations[conversation_id].updated_at = datetime.now()
    
    # Also reset model conversation history if it matches
    if tamil_gpt and hasattr(tamil_gpt, 'chat_history'):
        tamil_gpt.reset_conversation()
    
    return {"message": "Conversation reset successfully"}


@app.get("/model/info")
async def model_info(model: TamilGPTInference = Depends(get_model)):
    """Get model information."""
    return {
        "model_path": model.config.model_path,
        "use_custom_tokenizer": model.config.use_custom_tokenizer,
        "max_new_tokens": model.config.max_new_tokens,
        "temperature": model.config.temperature,
        "top_k": model.config.top_k,
        "top_p": model.config.top_p,
        "device": str(model.device),
        "torch_dtype": str(model.torch_dtype)
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    # Run server
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )
