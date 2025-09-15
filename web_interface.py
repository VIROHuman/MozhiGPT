#!/usr/bin/env python3
"""
TLM 1.0 Web Interface
A simple web interface for interacting with Tamil Language Model
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from tlm_1_0 import TamilLanguageModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="TLM 1.0 Web Interface")

# Initialize TLM
tlm = None

@app.on_event("startup")
async def startup_event():
    """Initialize TLM on startup"""
    global tlm
    try:
        logger.info("🚀 Starting TLM 1.0 Web Interface...")
        tlm = TamilLanguageModel()
        logger.info("✅ TLM 1.0 Web Interface ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize TLM: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def home():
    """Main web interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🇹🇦 Tamil Language Model (TLM) 1.0</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                color: #333;
            }
            .header h1 {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .tabs {
                display: flex;
                margin-bottom: 20px;
                border-bottom: 2px solid #eee;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                border: none;
                background: none;
                font-size: 16px;
                color: #666;
                border-bottom: 3px solid transparent;
                transition: all 0.3s;
            }
            .tab.active {
                color: #3498db;
                border-bottom-color: #3498db;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #333;
            }
            input, textarea, select {
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            input:focus, textarea:focus, select:focus {
                outline: none;
                border-color: #3498db;
            }
            button {
                background: linear-gradient(45deg, #3498db, #2980b9);
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
            }
            .response {
                background: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
                min-height: 100px;
                white-space: pre-wrap;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.6;
            }
            .loading {
                text-align: center;
                color: #666;
                font-style: italic;
            }
            .error {
                background: #f8d7da;
                border-color: #f5c6cb;
                color: #721c24;
            }
            .success {
                background: #d4edda;
                border-color: #c3e6cb;
                color: #155724;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🇹🇦 Tamil Language Model (TLM) 1.0</h1>
                <p>Interactive Tamil AI - Chat, Poetry, Stories, Translation & More!</p>
            </div>

            <div class="tabs">
                <button class="tab active" onclick="showTab('chat')">💬 Tamil Chat</button>
                <button class="tab" onclick="showTab('poetry')">🎭 Poetry</button>
                <button class="tab" onclick="showTab('story')">📚 Story</button>
                <button class="tab" onclick="showTab('translate')">🔄 Translation</button>
                <button class="tab" onclick="showTab('explain')">📖 Explain</button>
            </div>

            <!-- Tamil Chat Tab -->
            <div id="chat" class="tab-content active">
                <form onsubmit="submitForm(event, 'chat')">
                    <div class="form-group">
                        <label for="chat-message">Your Tamil Message:</label>
                        <textarea id="chat-message" name="message" rows="3" placeholder="வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?" required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="chat-length">Response Length:</label>
                        <select id="chat-length" name="max_length">
                            <option value="100">Short (100 words)</option>
                            <option value="150" selected>Medium (150 words)</option>
                            <option value="200">Long (200 words)</option>
                        </select>
                    </div>
                    <button type="submit">Send Message</button>
                </form>
                <div id="chat-response" class="response" style="display: none;"></div>
            </div>

            <!-- Poetry Tab -->
            <div id="poetry" class="tab-content">
                <form onsubmit="submitForm(event, 'poetry')">
                    <div class="form-group">
                        <label for="poetry-theme">Poetry Theme (in Tamil):</label>
                        <input id="poetry-theme" name="theme" type="text" placeholder="அன்பு" value="அன்பு" required>
                    </div>
                    <div class="form-group">
                        <label for="poetry-length">Poetry Length:</label>
                        <select id="poetry-length" name="max_length">
                            <option value="100">Short (100 words)</option>
                            <option value="150" selected>Medium (150 words)</option>
                            <option value="200">Long (200 words)</option>
                        </select>
                    </div>
                    <button type="submit">Generate Poetry</button>
                </form>
                <div id="poetry-response" class="response" style="display: none;"></div>
            </div>

            <!-- Story Tab -->
            <div id="story" class="tab-content">
                <form onsubmit="submitForm(event, 'story')">
                    <div class="form-group">
                        <label for="story-topic">Story Topic (in Tamil):</label>
                        <input id="story-topic" name="topic" type="text" placeholder="பழைய காலம்" value="பழைய காலம்" required>
                    </div>
                    <div class="form-group">
                        <label for="story-length">Story Length:</label>
                        <select id="story-length" name="max_length">
                            <option value="150">Short (150 words)</option>
                            <option value="200" selected>Medium (200 words)</option>
                            <option value="300">Long (300 words)</option>
                        </select>
                    </div>
                    <button type="submit">Generate Story</button>
                </form>
                <div id="story-response" class="response" style="display: none;"></div>
            </div>

            <!-- Translation Tab -->
            <div id="translate" class="tab-content">
                <form onsubmit="submitForm(event, 'translate')">
                    <div class="form-group">
                        <label for="translate-text">English Text to Translate:</label>
                        <textarea id="translate-text" name="text" rows="3" placeholder="Hello, how are you?" required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="translate-length">Translation Length:</label>
                        <select id="translate-length" name="max_length">
                            <option value="50">Short (50 words)</option>
                            <option value="100" selected>Medium (100 words)</option>
                            <option value="150">Long (150 words)</option>
                        </select>
                    </div>
                    <button type="submit">Translate to Tamil</button>
                </form>
                <div id="translate-response" class="response" style="display: none;"></div>
            </div>

            <!-- Explain Tab -->
            <div id="explain" class="tab-content">
                <form onsubmit="submitForm(event, 'explain')">
                    <div class="form-group">
                        <label for="explain-concept">Tamil Concept to Explain:</label>
                        <input id="explain-concept" name="concept" type="text" placeholder="தமிழ் மொழி" value="தமிழ் மொழி" required>
                    </div>
                    <div class="form-group">
                        <label for="explain-length">Explanation Length:</label>
                        <select id="explain-length" name="max_length">
                            <option value="100">Short (100 words)</option>
                            <option value="150" selected>Medium (150 words)</option>
                            <option value="200">Long (200 words)</option>
                        </select>
                    </div>
                    <button type="submit">Explain Concept</button>
                </form>
                <div id="explain-response" class="response" style="display: none;"></div>
            </div>
        </div>

        <script>
            function showTab(tabName) {
                // Hide all tab contents
                const contents = document.querySelectorAll('.tab-content');
                contents.forEach(content => content.classList.remove('active'));
                
                // Remove active class from all tabs
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach(tab => tab.classList.remove('active'));
                
                // Show selected tab content
                document.getElementById(tabName).classList.add('active');
                
                // Add active class to clicked tab
                event.target.classList.add('active');
            }

            async function submitForm(event, endpoint) {
                event.preventDefault();
                
                const form = event.target;
                const formData = new FormData(form);
                const data = Object.fromEntries(formData);
                
                // Convert max_length to integer
                data.max_length = parseInt(data.max_length);
                
                // Show loading
                const responseDiv = document.getElementById(endpoint + '-response');
                responseDiv.style.display = 'block';
                responseDiv.className = 'response loading';
                responseDiv.textContent = 'Generating response... Please wait...';
                
                try {
                    const response = await fetch(`/${endpoint}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        responseDiv.className = 'response success';
                        if (endpoint === 'chat') {
                            responseDiv.textContent = `User: ${result.message}\\n\\nTLM Response: ${result.response}`;
                        } else if (endpoint === 'poetry') {
                            responseDiv.textContent = `Theme: ${result.theme}\\n\\nPoetry:\\n${result.poetry}`;
                        } else if (endpoint === 'story') {
                            responseDiv.textContent = `Topic: ${result.topic}\\n\\nStory:\\n${result.story}`;
                        } else if (endpoint === 'translate') {
                            responseDiv.textContent = `English: ${result.original}\\n\\nTamil: ${result.translation}`;
                        } else if (endpoint === 'explain') {
                            responseDiv.textContent = `Concept: ${result.concept}\\n\\nExplanation: ${result.explanation}`;
                        }
                    } else {
                        responseDiv.className = 'response error';
                        responseDiv.textContent = `Error: ${result.detail || 'Something went wrong'}`;
                    }
                } catch (error) {
                    responseDiv.className = 'response error';
                    responseDiv.textContent = `Error: ${error.message}`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# API endpoints for the web interface
@app.post("/chat")
async def chat_web(message: str = Form(...), max_length: int = Form(150)):
    """Chat endpoint for web interface"""
    try:
        response = tlm.chat(message)
        return {
            "message": message,
            "response": response,
            "model": "TLM 1.0"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/poetry")
async def poetry_web(theme: str = Form(...), max_length: int = Form(150)):
    """Poetry endpoint for web interface"""
    try:
        poetry = tlm.generate_poetry(theme)
        return {
            "theme": theme,
            "poetry": poetry,
            "model": "TLM 1.0"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/story")
async def story_web(topic: str = Form(...), max_length: int = Form(200)):
    """Story endpoint for web interface"""
    try:
        story = tlm.generate_story(topic)
        return {
            "topic": topic,
            "story": story,
            "model": "TLM 1.0"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/translate")
async def translate_web(text: str = Form(...), max_length: int = Form(100)):
    """Translation endpoint for web interface"""
    try:
        translation = tlm.translate_to_tamil(text)
        return {
            "original": text,
            "translation": translation,
            "model": "TLM 1.0"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/explain")
async def explain_web(concept: str = Form(...), max_length: int = Form(150)):
    """Explain endpoint for web interface"""
    try:
        explanation = tlm.explain_tamil_concept(concept)
        return {
            "concept": concept,
            "explanation": explanation,
            "model": "TLM 1.0"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "web_interface:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
