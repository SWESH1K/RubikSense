#!/usr/bin/env python3
"""
Font Size Configuration Utility for RubikSenseApp

This utility allows you to easily adjust font sizes in the application.
"""

import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from RubikSenseApp.core.settings_manager import SettingsManager

def main():
    """Interactive font size configuration"""
    
    print("🎯 RubikSenseApp - Font Size Configuration")
    print("=" * 50)
    
    # Initialize settings
    settings = SettingsManager()
    
    print("\n📋 Current Font Sizes:")
    print(f"Small text (status, logs): {settings.get('ui.font_size_small', 10)}")
    print(f"Medium text (labels, buttons): {settings.get('ui.font_size_medium', 11)}")
    print(f"Large text (tabs, headers): {settings.get('ui.font_size_large', 12)}")
    print(f"Title text (section titles): {settings.get('ui.font_size_title', 16)}")
    print(f"Display text (timer): {settings.get('ui.font_size_display', 56)}")
    
    print("\n🔧 Font Size Presets:")
    print("1. Small fonts (good for small screens)")
    print("2. Medium fonts (default, balanced)")
    print("3. Large fonts (good for accessibility)")
    print("4. Extra large fonts (maximum readability)")
    print("5. Custom sizes")
    print("6. Exit without changes")
    
    try:
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == "1":
            # Small fonts
            settings.set("ui.font_size_small", 9)
            settings.set("ui.font_size_medium", 10)
            settings.set("ui.font_size_large", 11)
            settings.set("ui.font_size_title", 14)
            settings.set("ui.font_size_display", 48)
            print("✅ Small fonts applied!")
            
        elif choice == "2":
            # Medium fonts (default)
            settings.set("ui.font_size_small", 10)
            settings.set("ui.font_size_medium", 11)
            settings.set("ui.font_size_large", 12)
            settings.set("ui.font_size_title", 16)
            settings.set("ui.font_size_display", 56)
            print("✅ Medium fonts applied!")
            
        elif choice == "3":
            # Large fonts
            settings.set("ui.font_size_small", 11)
            settings.set("ui.font_size_medium", 13)
            settings.set("ui.font_size_large", 14)
            settings.set("ui.font_size_title", 18)
            settings.set("ui.font_size_display", 64)
            print("✅ Large fonts applied!")
            
        elif choice == "4":
            # Extra large fonts
            settings.set("ui.font_size_small", 12)
            settings.set("ui.font_size_medium", 14)
            settings.set("ui.font_size_large", 16)
            settings.set("ui.font_size_title", 20)
            settings.set("ui.font_size_display", 72)
            print("✅ Extra large fonts applied!")
            
        elif choice == "5":
            # Custom sizes
            print("\n🔧 Custom Font Configuration:")
            print("(Press Enter to keep current value)")
            
            # Get current values
            current_small = settings.get('ui.font_size_small', 10)
            current_medium = settings.get('ui.font_size_medium', 11)
            current_large = settings.get('ui.font_size_large', 12)
            current_title = settings.get('ui.font_size_title', 16)
            current_display = settings.get('ui.font_size_display', 56)
            
            # Small font
            new_small = input(f"Small text font size (current: {current_small}): ").strip()
            if new_small and new_small.isdigit():
                settings.set("ui.font_size_small", int(new_small))
                
            # Medium font
            new_medium = input(f"Medium text font size (current: {current_medium}): ").strip()
            if new_medium and new_medium.isdigit():
                settings.set("ui.font_size_medium", int(new_medium))
                
            # Large font
            new_large = input(f"Large text font size (current: {current_large}): ").strip()
            if new_large and new_large.isdigit():
                settings.set("ui.font_size_large", int(new_large))
                
            # Title font
            new_title = input(f"Title font size (current: {current_title}): ").strip()
            if new_title and new_title.isdigit():
                settings.set("ui.font_size_title", int(new_title))
                
            # Display font
            new_display = input(f"Display font size (current: {current_display}): ").strip()
            if new_display and new_display.isdigit():
                settings.set("ui.font_size_display", int(new_display))
                
            print("✅ Custom fonts applied!")
            
        elif choice == "6":
            print("👋 No changes made. Exiting...")
            return
            
        else:
            print("❌ Invalid choice. No changes made.")
            return
            
        print("\n🚀 Font sizes updated! Restart the application to see changes.")
        print("   Run: python launch_gui.py")
        
    except KeyboardInterrupt:
        print("\n\n👋 Configuration cancelled.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()