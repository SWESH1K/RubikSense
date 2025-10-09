"""
RubikSenseApp - Modern GUI Application for Rubik's Cube Recognition and Timing

A modern, user-friendly GUI application that provides:
- Real-time cube detection and color recognition
- Interactive solving timer with gesture control
- Calibration wizard for different lighting conditions
- Solve statistics and progress tracking
- Modern, responsive interface

Author: SWESH1K
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "SWESH1K"

# Import main components
from .gui.main_window import RubikSenseGUI
from .core.settings_manager import SettingsManager

def main():
    """Main entry point for the application"""
    app = RubikSenseGUI()
    app.run()

if __name__ == "__main__":
    main()