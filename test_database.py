#!/usr/bin/env python3
"""
Test database connection for MozhiGPT
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config.env')

# Add current directory to path
sys.path.insert(0, '.')

try:
    from database import init_database, TamilText, Conversation, ModelVersion
    print("✅ Database imports successful!")
    
    # Test database connection
    print("🔗 Testing database connection...")
    db = init_database()
    print("✅ Database connection successful!")
    
    # Test session
    session = db.get_session()
    
    # Test query
    print("📊 Testing database queries...")
    model_count = session.query(ModelVersion).count()
    print(f"✅ Found {model_count} model versions in database")
    
    # Test insert
    print("📝 Testing database insert...")
    test_text = TamilText(
        text="வணக்கம்! இது ஒரு சோதனை உரை.",
        source="test"
    )
    session.add(test_text)
    session.commit()
    
    # Verify insert
    test_count = session.query(TamilText).filter(TamilText.source == "test").count()
    print(f"✅ Successfully inserted test data. Test records: {test_count}")
    
    # Cleanup test data
    session.query(TamilText).filter(TamilText.source == "test").delete()
    session.commit()
    
    session.close()
    
    print("\n🎉 Database setup verification completed successfully!")
    print("✅ Ready for MozhiGPT database-enhanced training!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you've installed the database dependencies:")
    print("   pip install sqlalchemy psycopg2-binary")
    
except Exception as e:
    print(f"❌ Database connection error: {e}")
    print("💡 Check your database configuration:")
    print("   1. PostgreSQL is running")
    print("   2. Database 'mozhigpt' exists")
    print("   3. Credentials in config.env are correct")
    print("   4. Tables were created successfully")

