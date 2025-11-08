"""
AI Planner Server with SSE Support + Competition Analysis
Save as: run_server.py
Run: python run_server.py
"""

import sys
import os

# Add path if needed
sys.path.append(r'C:\Users\LENOVO\OneDrive')

def run_server():
    """Run Flask development server with SSE support"""
    
    # Import after path is set
    from app import app, init_db
    
    # Initialize database
    print("\n🔧 Initializing database...")
    init_db()
    print("✅ Database initialized successfully!")
    
    print("\n" + "="*70)
    print("🚀 AI PLANNER SERVER - ENHANCED EDITION")
    print("="*70)
    print("📊 Database: planner.db")
    print("🌐 Server: http://0.0.0.0:5000")
    print("🌐 Admin Panel: http://localhost:5000/admin.html")
    print("="*70)
    print("\n🎯 FEATURES ENABLED:")
    print("   ✅ Real-time SSE Streaming")
    print("   ✅ Feature Detection System")
    print("   ✅ Competition Analysis (AIM/Steps/Summary)")
    print("   ✅ Natural Language Processing")
    print("   ✅ Multi-AI Agent Planning")
    print("="*70)
    print("\n📡 API ENDPOINTS:")
    print("   🔴 POST   /api/plan/stream              → Real-time planning (SSE)")
    print("   📋 POST   /api/plan                     → Standard planning (JSON)")
    print("   🧠 POST   /api/parse-natural-language   → NLP to AIM+Steps")
    print("   🔐 POST   /api/login                    → User authentication")
    print("   📝 POST   /api/signup                   → User registration")
    print("   📊 GET    /api/plans/<username>         → Get user plans")
    print("   🏢 GET    /api/competition/<plan_id>    → Get competition data")
    print("   🔒 GET    /api/admin/stats              → Admin statistics")
    print("   🔒 GET    /api/admin/users              → Admin user list")
    print("   🔒 GET    /api/admin/plans              → Admin plans list")
    print("   🔒 GET    /api/admin/competition-stats  → Competition analytics")
    print("   ❤️  GET    /health                       → Health check")
    print("="*70)
    print("\n⚙️  SERVER CONFIGURATION:")
    print("   • Host: 0.0.0.0 (accessible from network)")
    print("   • Port: 5000")
    print("   • Mode: Production (debug=False)")
    print("   • Threading: Enabled (for SSE)")
    print("   • Timeout: None (long-running requests supported)")
    print("="*70)
    print("\n💡 TIPS:")
    print("   • Use admin.html for system monitoring")
    print("   • Admin key: admin123")
    print("   • Competition analysis runs automatically")
    print("   • All data saved to planner.db")
    print("="*70)
    print("\n✅ Server is ready! Press Ctrl+C to stop.\n")
    
    try:
        # Flask development server - supports SSE
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Server stopped by user")
        print("👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    run_server()
