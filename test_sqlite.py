#!/usr/bin/env python3
"""
Test SQLite database setup for MozhiGPT (No PostgreSQL required)
"""

import os
import sys
import uuid

# Add current directory to path
sys.path.insert(0, '.')

try:
    from database.sqlite_models import init_sqlite_database
    print("✅ SQLite database imports successful!")
    
    # Initialize SQLite database
    print("🔗 Initializing SQLite database...")
    db = init_sqlite_database("mozhigpt.db")
    print("✅ SQLite database connection successful!")
    
    # Test insert Tamil text
    print("📝 Testing Tamil text insert...")
    db.insert_tamil_text("வணக்கம்! இது ஒரு சோதனை உரை.", "test")
    db.insert_tamil_text("நல்ல நாள்! தமிழ் மொழி அழகானது.", "test")
    print("✅ Tamil texts inserted successfully!")
    
    # Test retrieve texts
    print("📊 Testing text retrieval...")
    texts = db.get_tamil_texts(limit=5)
    print(f"✅ Retrieved {len(texts)} texts from database")
    
    # Test conversation save
    print("💬 Testing conversation storage...")
    conversation_id = str(uuid.uuid4())
    session_id = "test_session_123"
    db.save_conversation(
        conversation_id=conversation_id,
        session_id=session_id,
        user_message="வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
        ai_response="வணக்கம்! நான் நன்றாக இருக்கிறேன். நன்றி கேட்டதற்கு!",
        response_time_ms=250
    )
    print("✅ Conversation saved successfully!")
    
    # Test conversation retrieval
    print("📖 Testing conversation history...")
    history = db.get_conversation_history(session_id)
    print(f"✅ Retrieved {len(history)} conversation entries")
    
    # Get database stats
    print("📊 Getting database statistics...")
    stats = db.get_stats()
    print(f"📈 Database Stats:")
    print(f"   • Total Tamil texts: {stats['total_texts']}")
    print(f"   • Processed texts: {stats['processed_texts']}")
    print(f"   • Total conversations: {stats['total_conversations']}")
    print(f"   • Model versions: {stats['total_models']}")
    
    print("\n🎉 SQLite database setup verification completed successfully!")
    print("✅ Ready for MozhiGPT database-enhanced training with SQLite!")
    print(f"📁 Database file created: {os.path.abspath('mozhigpt.db')}")
    
except Exception as e:
    print(f"❌ Database error: {e}")
    import traceback
    traceback.print_exc()

