#!/usr/bin/env python3
"""
RubikSenseApp Main Launcher

Modern GUI application for Rubik's Cube recognition and timing.
"""

import sys
import os
import logging
from pathlib import Path

# Add the parent directory to Python path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from RubikSenseApp import main
    
    if __name__ == "__main__":
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        print("Starting RubikSense GUI Application...")
        print("=" * 50)
        
        try:
            main()
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"Application error: {e}")
            logging.exception("Application crashed")
            sys.exit(1)
            
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)