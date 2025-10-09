#!/usr/bin/env python3
"""
IP Webcam URL Configuration Utility for RubikSenseApp

This utility helps you quickly configure the IP webcam URL.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path.cwd()))

try:
    from RubikSenseApp.core.settings_manager import SettingsManager
    
    def main():
        """Configure IP webcam URL"""
        
        print("📱 RubikSenseApp - IP Webcam Configuration")
        print("=" * 50)
        
        # Initialize settings
        settings = SettingsManager()
        
        current_url = settings.get("camera.device_id", "http://192.168.29.220:8080/video")
        current_fallback = settings.get("camera.fallback_device_id", 0)
        
        print(f"\n📋 Current Settings:")
        print(f"   IP Webcam URL: {current_url}")
        print(f"   Fallback Camera ID: {current_fallback}")
        
        print(f"\n📱 Setup Instructions:")
        print("1. Install 'IP Webcam' app on your Android phone")
        print("2. Connect your phone to the same WiFi as your computer")
        print("3. Open IP Webcam app and tap 'Start Server'")
        print("4. Note the IP address shown (e.g., 192.168.1.100)")
        print("5. The URL format is: http://[IP_ADDRESS]:8080/video")
        
        print(f"\n🔧 Configuration:")
        
        # Get new URL
        new_url = input(f"Enter new IP Webcam URL (or press Enter to keep current): ").strip()
        if not new_url:
            new_url = current_url
            
        # Validate URL format
        if not new_url.startswith("http"):
            new_url = "http://" + new_url
        if not new_url.endswith("/video"):
            if not new_url.endswith("/"):
                new_url += "/"
            new_url += "video"
        
        # Get fallback camera ID
        fallback_input = input(f"Enter fallback camera ID (0 for first camera, 1 for second, etc.) [{current_fallback}]: ").strip()
        if fallback_input.isdigit():
            new_fallback = int(fallback_input)
        else:
            new_fallback = current_fallback
            
        # Save settings
        settings.set("camera.device_id", new_url)
        settings.set("camera.fallback_device_id", new_fallback)
        
        print(f"\n✅ Configuration saved!")
        print(f"   IP Webcam URL: {new_url}")
        print(f"   Fallback Camera ID: {new_fallback}")
        
        print(f"\n🚀 Ready to use! Run: python launch_gui.py")
        print(f"\n💡 Tips:")
        print("• Make sure IP Webcam app is running on your phone")
        print("• Both devices should be on the same WiFi network")
        print("• If IP camera fails, app will automatically use local camera")
        
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run this script from the RubikSense directory.")
except KeyboardInterrupt:
    print("\n\n👋 Configuration cancelled.")
except Exception as e:
    print(f"❌ Error: {e}")