#!/usr/bin/env python3
"""
Quick Large Fonts Applier for RubikSenseApp

This script immediately applies large fonts to make the GUI more readable.
Run this before starting the application if fonts are too small.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path.cwd()))

try:
    from RubikSenseApp.core.settings_manager import SettingsManager
    
    print("🔧 Applying Large Fonts to RubikSenseApp...")
    
    # Initialize settings
    settings = SettingsManager()
    
    # Apply large font preset
    settings.set("ui.font_size_small", 11)
    settings.set("ui.font_size_medium", 13)  
    settings.set("ui.font_size_large", 14)
    settings.set("ui.font_size_title", 18)
    settings.set("ui.font_size_display", 64)
    
    print("✅ Large fonts applied successfully!")
    print("\n📋 New Font Sizes:")
    print(f"   Small text: {settings.get('ui.font_size_small')}")
    print(f"   Medium text: {settings.get('ui.font_size_medium')}")
    print(f"   Large text: {settings.get('ui.font_size_large')}")
    print(f"   Title text: {settings.get('ui.font_size_title')}")
    print(f"   Display text: {settings.get('ui.font_size_display')}")
    
    print("\n🚀 Ready to launch! Run: python launch_gui.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure you're running this from the RubikSense directory.")
except Exception as e:
    print(f"❌ Error: {e}")