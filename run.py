"""
RTA Scene Analyzer - Startup Script
Run this file to start the application
"""

import uvicorn
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚦 Starting RTA Scene Analyzer...")
    print("📍 Application will be available at: http://localhost:8000")
    print("📱 Use Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )
