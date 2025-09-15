#!/usr/bin/env python3
"""
Test script for Tamil Language Model (TLM) 1.0
"""

import requests
import json
import time

def test_tlm_api():
    """Test TLM 1.0 API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🇹🇦 Testing Tamil Language Model (TLM) 1.0 API")
    print("=" * 60)
    
    # Test health check
    print("\n1. Testing Health Check:")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test chat
    print("\n2. Testing Tamil Chat:")
    chat_data = {
        "message": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "max_length": 100,
        "temperature": 0.7
    }
    try:
        response = requests.post(f"{base_url}/chat", json=chat_data)
        result = response.json()
        print(f"User: {result['message']}")
        print(f"TLM: {result['response']}")
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
    
    # Test poetry generation
    print("\n3. Testing Tamil Poetry Generation:")
    poetry_data = {
        "theme": "அன்பு",
        "max_length": 150,
        "temperature": 0.8
    }
    try:
        response = requests.post(f"{base_url}/poetry", json=poetry_data)
        result = response.json()
        print(f"Theme: {result['theme']}")
        print(f"Poetry: {result['poetry']}")
    except Exception as e:
        print(f"❌ Poetry test failed: {e}")
    
    # Test story generation
    print("\n4. Testing Tamil Story Generation:")
    story_data = {
        "topic": "பழைய காலம்",
        "max_length": 200,
        "temperature": 0.8
    }
    try:
        response = requests.post(f"{base_url}/story", json=story_data)
        result = response.json()
        print(f"Topic: {result['topic']}")
        print(f"Story: {result['story']}")
    except Exception as e:
        print(f"❌ Story test failed: {e}")
    
    # Test translation
    print("\n5. Testing English to Tamil Translation:")
    translate_data = {
        "text": "Hello, how are you?",
        "max_length": 100,
        "temperature": 0.5
    }
    try:
        response = requests.post(f"{base_url}/translate", json=translate_data)
        result = response.json()
        print(f"English: {result['original']}")
        print(f"Tamil: {result['translation']}")
    except Exception as e:
        print(f"❌ Translation test failed: {e}")
    
    # Test concept explanation
    print("\n6. Testing Tamil Concept Explanation:")
    explain_data = {
        "concept": "தமிழ் மொழி",
        "max_length": 150,
        "temperature": 0.6
    }
    try:
        response = requests.post(f"{base_url}/explain", json=explain_data)
        result = response.json()
        print(f"Concept: {result['concept']}")
        print(f"Explanation: {result['explanation']}")
    except Exception as e:
        print(f"❌ Explanation test failed: {e}")
    
    print("\n✅ TLM 1.0 API testing completed!")

def test_direct_tlm():
    """Test TLM directly without API"""
    print("\n🔧 Testing TLM 1.0 Directly:")
    print("-" * 40)
    
    try:
        from tlm_1_0 import TamilLanguageModel
        
        # Initialize TLM
        tlm = TamilLanguageModel()
        
        # Test conversation
        print("\n💬 Testing Tamil Conversation:")
        response = tlm.chat("வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?")
        print(f"User: வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?")
        print(f"TLM: {response}")
        
        # Test poetry
        print("\n🎭 Testing Tamil Poetry:")
        poetry = tlm.generate_poetry("அன்பு")
        print(f"Poetry about அன்பு:")
        print(f"{poetry}")
        
        print("\n✅ Direct TLM testing completed!")
        
    except Exception as e:
        print(f"❌ Direct TLM test failed: {e}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Test API (requires server running)")
    print("2. Test Direct TLM")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_tlm_api()
    elif choice == "2":
        test_direct_tlm()
    else:
        print("Invalid choice. Running direct test...")
        test_direct_tlm()
