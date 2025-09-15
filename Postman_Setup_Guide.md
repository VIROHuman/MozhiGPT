# 📮 Postman Setup Guide for TLM 1.0 API

This guide will help you set up and test the Tamil Language Model (TLM) 1.0 API using Postman.

## 🚀 Quick Setup

### 1. Import the Collection

1. Open Postman
2. Click **Import** button
3. Select **File** tab
4. Choose `TLM_1_0_Postman_Collection.json` from your MozhiGPT folder
5. Click **Import**

### 2. Set Up Environment Variables

1. In Postman, click the **Environments** tab
2. Click **Create Environment**
3. Name it: `TLM 1.0 Local`
4. Add variable:
   - **Variable**: `base_url`
   - **Initial Value**: `http://localhost:8000`
   - **Current Value**: `http://localhost:8000`
5. Click **Save**

### 3. Select Environment

1. Click the environment dropdown (top right)
2. Select `TLM 1.0 Local`

## 📡 API Endpoints Overview

### 1. **Health Check** - `GET /health`
- **Purpose**: Check if the API server is running
- **No body required**
- **Expected Response**: `{"status": "healthy", "model_loaded": true}`

### 2. **Tamil Chat** - `POST /chat`
- **Purpose**: Chat with TLM in Tamil
- **Body Example**:
```json
{
  "message": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
  "max_length": 120,
  "temperature": 0.7
}
```

### 3. **Generate Poetry** - `POST /poetry`
- **Purpose**: Generate Tamil poetry
- **Body Example**:
```json
{
  "theme": "அன்பு",
  "max_length": 150,
  "temperature": 0.8
}
```

### 4. **Generate Story** - `POST /story`
- **Purpose**: Generate Tamil stories
- **Body Example**:
```json
{
  "topic": "பழைய காலம்",
  "max_length": 200,
  "temperature": 0.8
}
```

### 5. **Translate** - `POST /translate`
- **Purpose**: English to Tamil translation
- **Body Example**:
```json
{
  "text": "Hello, how are you?",
  "max_length": 100,
  "temperature": 0.5
}
```

### 6. **Explain Concept** - `POST /explain`
- **Purpose**: Explain Tamil concepts
- **Body Example**:
```json
{
  "concept": "தமிழ் மொழி",
  "max_length": 150,
  "temperature": 0.6
}
```

## 🔧 Testing Steps

### Step 1: Start the API Server

```bash
# In your terminal, navigate to MozhiGPT folder
cd MozhiGPT

# Start the API server
python start_tlm.py
# Choose option 1: Start API Server
```

### Step 2: Test Health Check

1. Select **Health Check** request
2. Click **Send**
3. You should see: `{"status": "healthy", "model_loaded": true}`

### Step 3: Test Tamil Chat

1. Select **Tamil Chat** request
2. Click **Send**
3. You should see a Tamil response

### Step 4: Test Poetry Generation

1. Select **Generate Tamil Poetry** request
2. Click **Send**
3. You should see Tamil poetry about "அன்பு" (love)

### Step 5: Test Other Endpoints

Repeat the process for Story, Translation, and Concept Explanation endpoints.

## 🎯 Sample Test Cases

### Tamil Chat Examples

```json
// Greeting
{
  "message": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
  "max_length": 100,
  "temperature": 0.7
}

// Question about Tamil
{
  "message": "தமிழ் மொழி பற்றி சொல்லுங்கள்",
  "max_length": 150,
  "temperature": 0.7
}

// Poetry request
{
  "message": "அன்பு பற்றி ஒரு கவிதை எழுதுங்கள்",
  "max_length": 200,
  "temperature": 0.8
}
```

### Poetry Themes

```json
// Love
{
  "theme": "அன்பு",
  "max_length": 150,
  "temperature": 0.8
}

// Nature
{
  "theme": "இயற்கை",
  "max_length": 150,
  "temperature": 0.8
}

// Friendship
{
  "theme": "நட்பு",
  "max_length": 150,
  "temperature": 0.8
}
```

### Story Topics

```json
// Old times
{
  "topic": "பழைய காலம்",
  "max_length": 200,
  "temperature": 0.8
}

// Village life
{
  "topic": "கிராம வாழ்க்கை",
  "max_length": 200,
  "temperature": 0.8
}

// Family
{
  "topic": "குடும்பம்",
  "max_length": 200,
  "temperature": 0.8
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Connection Refused**
   - Make sure the API server is running
   - Check if port 8000 is available
   - Verify the base_url in environment

2. **Model Not Loaded**
   - Wait for the model to load (first request takes time)
   - Check server logs for errors
   - Ensure you have enough GPU memory

3. **Empty Responses**
   - Try increasing `max_length`
   - Adjust `temperature` (0.5-1.0)
   - Check if the prompt is in Tamil

### Debug Steps

1. Check server logs in terminal
2. Test with Health Check first
3. Try different Tamil prompts
4. Adjust generation parameters

## 📊 Response Format

All endpoints return JSON responses in this format:

```json
{
  "message": "User input",
  "response": "TLM generated response",
  "model": "TLM 1.0"
}
```

## 🎉 Success!

Once you can successfully test all endpoints, you have a fully working Tamil Language Model API! You can now:

- Build Tamil chatbots
- Generate Tamil content
- Create Tamil poetry
- Translate English to Tamil
- Explain Tamil concepts

Happy testing! 🇹🇦
