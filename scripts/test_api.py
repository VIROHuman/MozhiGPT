#!/usr/bin/env python3
"""
Test script for MozhiGPT API endpoints.
"""

import asyncio
import json
import logging
from typing import Dict, Any
import httpx
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MozhiGPTAPITester:
    """Test suite for MozhiGPT API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def test_health_check(self) -> bool:
        """Test health check endpoint."""
        try:
            logger.info("Testing health check...")
            response = await self.client.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Health check passed: {data}")
                return True
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    async def test_chat_endpoint(self) -> bool:
        """Test chat endpoint."""
        try:
            logger.info("Testing chat endpoint...")
            
            payload = {
                "message": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
                "include_history": True,
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = await self.client.post(
                f"{self.base_url}/chat",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Chat response: {data['response'][:100]}...")
                return True
            else:
                logger.error(f"❌ Chat endpoint failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Chat endpoint error: {e}")
            return False
    
    async def test_streaming_endpoint(self) -> bool:
        """Test streaming chat endpoint."""
        try:
            logger.info("Testing streaming endpoint...")
            
            payload = {
                "message": "தமிழ் மொழி பற்றி சொல்லுங்கள்",
                "stream": True
            }
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/stream",
                json=payload
            ) as response:
                
                if response.status_code != 200:
                    logger.error(f"❌ Streaming failed: {response.status_code}")
                    return False
                
                full_response = ""
                async for chunk in response.aiter_text():
                    if chunk.startswith("data: "):
                        try:
                            data = json.loads(chunk[6:])
                            if data.get("type") == "token":
                                full_response += data.get("data", "")
                            elif data.get("type") == "complete":
                                break
                        except json.JSONDecodeError:
                            continue
                
                logger.info(f"✅ Streaming response: {full_response[:100]}...")
                return True
                
        except Exception as e:
            logger.error(f"❌ Streaming endpoint error: {e}")
            return False
    
    async def test_websocket_endpoint(self) -> bool:
        """Test WebSocket endpoint."""
        try:
            logger.info("Testing WebSocket endpoint...")
            
            ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            ws_url += "/chat/ws"
            
            async with websockets.connect(ws_url) as websocket:
                # Send message
                await websocket.send(json.dumps({
                    "message": "WebSocket மூலம் வணக்கம்!"
                }))
                
                # Receive response
                full_response = ""
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        data = json.loads(message)
                        
                        if data.get("type") == "token":
                            full_response += data.get("data", "")
                        elif data.get("type") == "complete":
                            break
                        elif data.get("type") == "error":
                            logger.error(f"WebSocket error: {data.get('data')}")
                            return False
                            
                    except asyncio.TimeoutError:
                        break
                
                logger.info(f"✅ WebSocket response: {full_response[:100]}...")
                return True
                
        except Exception as e:
            logger.error(f"❌ WebSocket endpoint error: {e}")
            return False
    
    async def test_model_info(self) -> bool:
        """Test model info endpoint."""
        try:
            logger.info("Testing model info endpoint...")
            response = await self.client.get(f"{self.base_url}/model/info")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Model info: {json.dumps(data, indent=2)}")
                return True
            else:
                logger.error(f"❌ Model info failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model info error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests."""
        logger.info("🧪 Starting MozhiGPT API tests...")
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Chat Endpoint", self.test_chat_endpoint),
            ("Streaming Endpoint", self.test_streaming_endpoint),
            ("WebSocket Endpoint", self.test_websocket_endpoint),
            ("Model Info", self.test_model_info),
        ]
        
        results = {}
        passed = 0
        
        for test_name, test_func in tests:
            logger.info(f"\n--- {test_name} ---")
            try:
                result = await test_func()
                results[test_name] = result
                if result:
                    passed += 1
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*50)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nTotal: {passed}/{len(tests)} tests passed")
        
        if passed == len(tests):
            logger.info("🎉 All tests passed! MozhiGPT API is working correctly.")
        else:
            logger.warning(f"⚠️ {len(tests) - passed} tests failed. Check the logs above.")
        
        await self.client.aclose()
        return results


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MozhiGPT API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", choices=["health", "chat", "stream", "ws", "info", "all"], 
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    tester = MozhiGPTAPITester(args.url)
    
    if args.test == "all":
        await tester.run_all_tests()
    else:
        test_map = {
            "health": tester.test_health_check,
            "chat": tester.test_chat_endpoint,
            "stream": tester.test_streaming_endpoint,
            "ws": tester.test_websocket_endpoint,
            "info": tester.test_model_info,
        }
        
        if args.test in test_map:
            result = await test_map[args.test]()
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"Test result: {status}")
        
        await tester.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
