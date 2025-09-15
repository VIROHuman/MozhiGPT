#!/usr/bin/env python3
"""
Start Tamil Language Model (TLM) 1.0
"""

import subprocess
import sys
import os
import time

def check_requirements():
    """Check if required packages are installed"""
    try:
        import torch
        import transformers
        import fastapi
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements_tlm.txt")
        return False

def start_api_server():
    """Start the TLM 1.0 API server"""
    print("🚀 Starting Tamil Language Model (TLM) 1.0 API Server...")
    print("=" * 60)
    
    if not check_requirements():
        return
    
    try:
        # Start the API server
        subprocess.run([
            sys.executable, "api_server.py"
        ])
    except KeyboardInterrupt:
        print("\n👋 TLM 1.0 API Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def start_direct_test():
    """Start direct TLM testing"""
    print("🔧 Starting Tamil Language Model (TLM) 1.0 Direct Test...")
    print("=" * 60)
    
    if not check_requirements():
        return
    
    try:
        # Run direct test
        subprocess.run([
            sys.executable, "test_tlm.py"
        ])
    except KeyboardInterrupt:
        print("\n👋 TLM 1.0 Direct Test stopped")
    except Exception as e:
        print(f"❌ Error running test: {e}")

def start_web_interface():
    """Start the web interface"""
    print("🌐 Starting Tamil Language Model (TLM) 1.0 Web Interface...")
    print("=" * 60)
    
    if not check_requirements():
        return
    
    try:
        # Start the web interface
        subprocess.run([
            sys.executable, "web_interface.py"
        ])
    except KeyboardInterrupt:
        print("\n👋 TLM 1.0 Web Interface stopped")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")

def main():
    """Main function"""
    print("🇹🇦 Tamil Language Model (TLM) 1.0")
    print("=" * 40)
    print("Choose an option:")
    print("1. Start API Server")
    print("2. Start Web Interface")
    print("3. Run Direct Test")
    print("4. Install Requirements")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            start_api_server()
            break
        elif choice == "2":
            start_web_interface()
            break
        elif choice == "3":
            start_direct_test()
            break
        elif choice == "4":
            print("Installing requirements...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_tlm.txt"])
            print("✅ Requirements installed!")
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()
