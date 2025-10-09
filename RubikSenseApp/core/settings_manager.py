"""
Settings Manager for RubikSenseApp

Handles application configuration, calibration data, and user preferences.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class SettingsManager:
    """Manages application settings and configuration files"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the settings manager
        
        Args:
            config_dir: Directory to store config files. If None, uses default.
        """
        if config_dir is None:
            self.config_dir = Path.home() / ".rubiksense"
        else:
            self.config_dir = Path(config_dir)
        
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configuration
        self.default_config = {
            "camera": {
                "device_id": "http://192.168.29.220:8080/video",  # IP webcam URL from prototype
                "fallback_device_id": 0,  # Local camera as fallback
                "frame_width": 1280,
                "frame_height": 720,
                "fps": 30
            },
            "detection": {
                "min_contour_area": 10000,
                "gaussian_blur_kernel": 5,
                "canny_low_threshold": 50,
                "canny_high_threshold": 150,
                "morph_kernel_size": 7
            },
            "ui": {
                "theme": "modern_dark",
                "auto_save": True,
                "show_fps": True,
                "window_width": 1200,
                "window_height": 800,
                "font_size_small": 10,
                "font_size_medium": 11,
                "font_size_large": 12,
                "font_size_title": 16,
                "font_size_display": 56
            },
            "timer": {
                "countdown_seconds": 3,
                "auto_reset": False,
                "sound_enabled": True
            }
        }
        
        self.config_file = self.config_dir / "config.json"
        self.calibration_file = self.config_dir / "calibration.json"
        self.history_file = self.config_dir / "solve_history.json"
        self.best_time_file = self.config_dir / "best_time.json"
        
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with default config to ensure all keys exist
                return self._merge_configs(self.default_config, config)
            except (json.JSONDecodeError, Exception) as e:
                logging.warning(f"Failed to load config: {e}. Using defaults.")
                
        return self.default_config.copy()
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def get(self, key_path: str, default=None):
        """
        Get a configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the config value (e.g., "camera.device_id")
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set a configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        config[keys[-1]] = value
        self.save_config()
    
    def load_calibration(self) -> Optional[Dict[str, Any]]:
        """Load color calibration data"""
        if self.calibration_file.exists():
            try:
                with open(self.calibration_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load calibration: {e}")
        return None
    
    def save_calibration(self, calibration_data: Dict[str, Any]):
        """Save color calibration data"""
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save calibration: {e}")
    
    def load_solve_history(self) -> list:
        """Load solve time history"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load solve history: {e}")
        return []
    
    def save_solve_time(self, solve_time: float, scramble: str = "", solved: bool = True):
        """Add a new solve time to history"""
        history = self.load_solve_history()
        
        solve_entry = {
            "time": solve_time,
            "scramble": scramble,
            "solved": solved,
            "timestamp": int(time.time()),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        history.append(solve_entry)
        
        # Keep only the last 1000 solves
        if len(history) > 1000:
            history = history[-1000:]
        
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            # Update best time if this is better
            self._update_best_time(solve_time)
            
        except Exception as e:
            logging.error(f"Failed to save solve time: {e}")
    
    def get_best_time(self) -> Optional[float]:
        """Get the best solve time"""
        if self.best_time_file.exists():
            try:
                with open(self.best_time_file, 'r') as f:
                    data = json.load(f)
                    return data.get('best_time')
            except Exception as e:
                logging.error(f"Failed to load best time: {e}")
        return None
    
    def _update_best_time(self, solve_time: float):
        """Update best time if the new time is better"""
        current_best = self.get_best_time()
        if current_best is None or solve_time < current_best:
            try:
                with open(self.best_time_file, 'w') as f:
                    json.dump({'best_time': solve_time}, f)
            except Exception as e:
                logging.error(f"Failed to update best time: {e}")
    
    def _merge_configs(self, default: dict, user: dict) -> dict:
        """Recursively merge user config with default config"""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result

# Add missing import
import time