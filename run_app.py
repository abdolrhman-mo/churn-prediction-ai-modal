#!/usr/bin/env python3
"""
SVM Recall Optimization Dashboard Runner
Run this file to launch the specialized SVM dashboard for churn prediction
"""

import subprocess
import sys
import os

def main():
    """Launch the SVM Recall Optimization Dashboard"""
    # Check if we're in the right directory
    if not os.path.exists("ui/main_app.py"):
        print("Error: ui/main_app.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Launch the app
    print("🎯 Launching SVM Recall Optimization Dashboard...")
    print("📁 Using specialized SVM dashboard from ui/ folder")
    print("🚀 This dashboard demonstrates threshold optimization for maximum recall!")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/main_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
