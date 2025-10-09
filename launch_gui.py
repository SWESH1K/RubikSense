#!/usr/bin/env python3
"""
RubikSense GUI Launcher

Quick launcher for the modern GUI application.
Run this script to start the RubikSenseApp.
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the RubikSenseApp GUI"""
    
    print("🎯 RubikSense - Modern GUI Application")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    app_dir = current_dir / "RubikSenseApp"
    
    if not app_dir.exists():
        print("❌ RubikSenseApp directory not found!")
        print("Please run this script from the RubikSense project root directory.")
        return 1
        
    # Add current directory to Python path
    sys.path.insert(0, str(current_dir))
    
    try:
        # Import and run the application
        from RubikSenseApp import main as run_app
        
        print("🚀 Starting RubikSense GUI...")
        print("\n📋 Quick Start Guide:")
        print("1. Setup Camera (choose one):")
        print("   📱 IP Webcam: run 'python configure_ip_webcam.py' first")
        print("   📹 Local Camera: connect USB/built-in camera")
        print("2. Go to Calibration tab")
        print("3. Click 'Start Camera'")
        print("4. Click 'Start Calibration' and follow instructions")
        print("5. Show each cube face and press SPACE to capture")
        print("6. Save calibration when complete")
        print("\n💡 Tips:")
        print("• For IP Webcam: install 'IP Webcam' app on Android phone")
        print("• If fonts are too small/large: run 'python RubikSenseApp/configure_fonts.py'")
        print("• Camera settings: Settings → Camera Settings in the app")
        print("\nEnjoy using RubikSense! 🧩")
        print("-" * 50)
        
        # Launch the GUI
        run_app()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Please install required dependencies:")
        print("   pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)