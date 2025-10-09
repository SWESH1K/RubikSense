"""
Main GUI Window for RubikSenseApp

Modern, responsive interface with tabbed navigation and dark/light theme support.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from typing import Optional
import logging
from pathlib import Path

# Import application modules
from ..core.settings_manager import SettingsManager
from ..vision.cube_detector import CubeDetector, CameraManager
from .camera_frame import CameraFrame
from .timer_frame import TimerFrame
from .analysis_frame import AnalysisFrame
from .settings_dialog import SettingsDialog

class RubikSenseGUI:
    """Main application GUI class"""

    def __init__(self):
        """Initialize the main GUI"""
        self.root = tk.Tk()
        self.setup_window()

        # Initialize settings manager
        self.settings = SettingsManager()

        # Initialize components
        self.camera_manager = None
        self.cube_detector = None
        self.current_theme = self.settings.get("ui.theme", "modern_dark")

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create GUI elements
        self.create_menu()
        self.create_main_interface()
        self.create_status_bar()

        # Apply theme
        self.apply_theme(self.current_theme)

        # Initialize camera and cube detector
        self.initialize_vision_components()

    def setup_window(self):
        """Setup main window properties"""
        self.root.title("RubikSense - Advanced Cube Recognition & Timer")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (800 // 2)
        self.root.geometry(f"1200x800+{x}+{y}")

        # Configure window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Configure grid weights for responsiveness
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def create_menu(self):
        """Create application menu bar"""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        # File menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Calibration...", command=self.import_calibration)
        file_menu.add_command(label="Export Calibration...", command=self.export_calibration)
        file_menu.add_separator()
        file_menu.add_command(label="Import Solve History...", command=self.import_history)
        file_menu.add_command(label="Export Solve History...", command=self.export_history)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        # View menu
        view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=view_menu)

        # Theme submenu
        theme_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Theme", menu=theme_menu)
        theme_menu.add_command(label="Modern Dark", command=lambda: self.apply_theme("modern_dark"))
        theme_menu.add_command(label="Modern Light", command=lambda: self.apply_theme("modern_light"))
        theme_menu.add_command(label="Classic", command=lambda: self.apply_theme("classic"))

        view_menu.add_separator()
        view_menu.add_checkbutton(label="Show FPS", command=self.toggle_fps_display)

        # Settings menu
        settings_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Camera Settings...", command=self.open_camera_settings)
        settings_menu.add_command(label="Detection Settings...", command=self.open_detection_settings)
        settings_menu.add_command(label="Preferences...", command=self.open_preferences)

        # Help menu
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)

    def create_main_interface(self):
        """Create the main tabbed interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Create tabs
        self.create_calibration_tab()
        self.create_timer_tab()
        self.create_analysis_tab()

        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def create_calibration_tab(self):
        """Create the calibration tab"""
        # Create frame for calibration tab
        self.calibration_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.calibration_frame, text="📹 Calibration")

        # Initialize camera frame component
        self.camera_frame = CameraFrame(self.calibration_frame, self.settings)
        self.camera_frame.pack(fill="both", expand=True)

    def create_timer_tab(self):
        """Create the timer tab"""
        # Create frame for timer tab
        self.timer_frame_container = ttk.Frame(self.notebook)
        self.notebook.add(self.timer_frame_container, text="⏱️ Timer")

        # Initialize timer frame component
        self.timer_frame = TimerFrame(self.timer_frame_container, self.settings)
        self.timer_frame.pack(fill="both", expand=True)

    def create_analysis_tab(self):
        """Create the analysis tab"""
        # Create frame for analysis tab
        self.analysis_frame_container = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame_container, text="📊 Analysis")

        # Initialize analysis frame component
        self.analysis_frame = AnalysisFrame(self.analysis_frame_container, self.settings)
        self.analysis_frame.pack(fill="both", expand=True)

    def create_status_bar(self):
        """Create application status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.grid(row=2, column=0, sticky="ew", padx=5, pady=2)

        # Status label
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side="left")

        # Camera status
        self.camera_status = ttk.Label(self.status_bar, text="Camera: Disconnected")
        self.camera_status.pack(side="left", padx=(20, 0))

        # FPS display
        self.fps_label = ttk.Label(self.status_bar, text="FPS: --")
        self.fps_label.pack(side="right")

        # Version info
        version_label = ttk.Label(self.status_bar, text="v1.0.0")
        version_label.pack(side="right", padx=(20, 0))

    def initialize_vision_components(self):
        """Initialize camera and cube detection components"""
        try:
            # Initialize camera manager with IP webcam and fallback
            camera_config = self.settings.config.get("camera", {})
            self.camera_manager = CameraManager(
                device_id=camera_config.get("device_id", "http://192.168.29.220:8080/video"),
                fallback_device_id=camera_config.get("fallback_device_id", 0),
                width=camera_config.get("frame_width", 1280),
                height=camera_config.get("frame_height", 720)
            )

            # Initialize cube detector
            detection_config = self.settings.config.get("detection", {})
            self.cube_detector = CubeDetector(detection_config)

            # Load existing calibration if available
            calibration_data = self.settings.load_calibration()
            if calibration_data:
                self.cube_detector.load_calibration(calibration_data)
                self.update_status("Calibration loaded")
            else:
                self.update_status("No calibration found - please calibrate first")

        except Exception as e:
            self.logger.error(f"Failed to initialize vision components: {e}")
            messagebox.showerror("Initialization Error",
                               f"Failed to initialize camera/detection components: {e}")

    def apply_theme(self, theme_name: str):
        """Apply a visual theme to the application"""
        self.current_theme = theme_name
        self.settings.set("ui.theme", theme_name)

        try:
            if theme_name == "modern_dark":
                self.apply_dark_theme()
            elif theme_name == "modern_light":
                self.apply_light_theme()
            else:
                self.apply_classic_theme()

            self.update_status(f"Theme changed to {theme_name}")

        except Exception as e:
            self.logger.error(f"Failed to apply theme: {e}")

    def apply_dark_theme(self):
        """Apply modern dark theme"""
        style = ttk.Style()

        # Configure dark theme colors
        bg_color = "#2b2b2b"
        fg_color = "#ffffff"
        select_color = "#404040"

        # Get font sizes from settings
        font_large = self.settings.get("ui.font_size_large", 12)
        font_medium = self.settings.get("ui.font_size_medium", 11)
        font_small = self.settings.get("ui.font_size_small", 10)

        # Configure styles
        style.theme_use("clam")

        # Configure notebook
        style.configure("TNotebook", background=bg_color, borderwidth=0)
        style.configure("TNotebook.Tab", background=select_color, foreground=fg_color,
                       padding=[12, 8], font=("Segoe UI", font_large))
        style.map("TNotebook.Tab", background=[("selected", "#0078d4")],
                 foreground=[("selected", "white")])

        # Configure frames and labels
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color,
                       font=("Segoe UI", font_medium))

        # Configure buttons
        style.configure("TButton", font=("Segoe UI", font_medium))

        # Configure text widgets
        style.configure("TText", font=("Segoe UI", font_small))

        # Configure entry widgets
        style.configure("TEntry", font=("Segoe UI", font_medium))

        # Update root background
        self.root.configure(bg=bg_color)

    def apply_light_theme(self):
        """Apply modern light theme"""
        style = ttk.Style()
        style.theme_use("default")

        # Get font sizes from settings
        font_large = self.settings.get("ui.font_size_large", 12)
        font_medium = self.settings.get("ui.font_size_medium", 11)
        font_small = self.settings.get("ui.font_size_small", 10)

        # Configure light theme with configurable fonts
        style.configure("TNotebook.Tab", font=("Segoe UI", font_large), padding=[12, 8])
        style.configure("TLabel", font=("Segoe UI", font_medium))
        style.configure("TButton", font=("Segoe UI", font_medium))
        style.configure("TText", font=("Segoe UI", font_small))
        style.configure("TEntry", font=("Segoe UI", font_medium))

    def apply_classic_theme(self):
        """Apply classic Windows theme"""
        style = ttk.Style()
        style.theme_use("winnative" if self.root.tk.call("tk", "windowingsystem") == "win32" else "default")

        # Get font sizes from settings
        font_large = self.settings.get("ui.font_size_large", 12)
        font_medium = self.settings.get("ui.font_size_medium", 11)
        font_small = self.settings.get("ui.font_size_small", 10)

        # Configure classic theme with configurable fonts
        style.configure("TNotebook.Tab", font=("Segoe UI", font_large), padding=[12, 8])
        style.configure("TLabel", font=("Segoe UI", font_medium))
        style.configure("TButton", font=("Segoe UI", font_medium))
        style.configure("TText", font=("Segoe UI", font_small))
        style.configure("TEntry", font=("Segoe UI", font_medium))

    def on_tab_changed(self, event):
        """Handle tab change events"""
        selected_tab = self.notebook.select()
        tab_text = self.notebook.tab(selected_tab, "text")

        if "Calibration" in tab_text:
            self.camera_frame.on_tab_selected()
        elif "Timer" in tab_text:
            self.timer_frame.on_tab_selected()
        elif "Analysis" in tab_text:
            self.analysis_frame.on_tab_selected()

    def update_status(self, message: str):
        """Update status bar message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def update_camera_status(self, connected: bool):
        """Update camera connection status"""
        status = "Connected" if connected else "Disconnected"
        color = "green" if connected else "red"
        self.camera_status.config(text=f"Camera: {status}", foreground=color)

    def update_fps_display(self, fps: float):
        """Update FPS display"""
        if self.settings.get("ui.show_fps", True):
            self.fps_label.config(text=f"FPS: {fps:.1f}")
        else:
            self.fps_label.config(text="")

    # Menu handlers
    def import_calibration(self):
        """Import calibration from file"""
        filename = filedialog.askopenfilename(
            title="Import Calibration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            # Implementation will be added in camera_frame.py
            self.camera_frame.import_calibration(filename)

    def export_calibration(self):
        """Export calibration to file"""
        filename = filedialog.asksaveasfilename(
            title="Export Calibration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            # Implementation will be added in camera_frame.py
            self.camera_frame.export_calibration(filename)

    def import_history(self):
        """Import solve history from file"""
        filename = filedialog.askopenfilename(
            title="Import Solve History",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.analysis_frame.import_history(filename)

    def export_history(self):
        """Export solve history to file"""
        filename = filedialog.asksaveasfilename(
            title="Export Solve History",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.analysis_frame.export_history(filename)

    def toggle_fps_display(self):
        """Toggle FPS display on/off"""
        current = self.settings.get("ui.show_fps", True)
        self.settings.set("ui.show_fps", not current)

    def open_camera_settings(self):
        """Open camera settings dialog"""
        dialog = SettingsDialog(self.root, self.settings, "camera")
        if dialog.result:
            # Stop current camera if running
            if hasattr(self, 'camera_frame') and self.camera_frame.is_running:
                self.camera_frame.stop_camera()
            
            # Reinitialize with new settings
            self.initialize_vision_components()
            
            # Update status
            new_device = self.settings.get("camera.device_id")
            self.update_status(f"Camera settings updated: {new_device}")

    def open_detection_settings(self):
        """Open detection settings dialog"""
        dialog = SettingsDialog(self.root, self.settings, "detection")
        if dialog.result:
            detection_config = self.settings.config.get("detection", {})
            if self.cube_detector:
                self.cube_detector.config = detection_config

    def open_preferences(self):
        """Open general preferences dialog"""
        dialog = SettingsDialog(self.root, self.settings, "preferences")

    def show_help(self):
        """Show help documentation"""
        help_text = """
RubikSense User Guide

1. Calibration:
   - Select the Calibration tab
   - Follow the on-screen instructions to show each face
   - Press Space to capture each face
   - Save calibration when complete

2. Timer:
   - Select the Timer tab
   - Place hands over start zones
   - Remove hands to start timer
   - Place hands back to stop timer

3. Analysis:
   - View solve statistics and history
   - Export data for further analysis

For more detailed instructions, visit the project documentation.
        """

        messagebox.showinfo("User Guide", help_text)

    def show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts_text = """
Keyboard Shortcuts:

General:
  Ctrl+Q - Quit application
  F1 - Show help
  F11 - Toggle fullscreen

Calibration:
  Space - Capture current face
  R - Reset calibration
  S - Save calibration

Timer:
  Space - Start/Stop timer
  R - Reset timer

Analysis:
  Ctrl+E - Export data
  Ctrl+I - Import data
        """

        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)

    def show_about(self):
        """Show about dialog"""
        about_text = """
RubikSense v1.0.0

Advanced Rubik's Cube Recognition & Timer

Features:
• Real-time cube detection
• Color calibration system
• Interactive solving timer
• Solve statistics and analysis
• Modern, responsive interface

Developed by: SWESH1K
Built with: Python, OpenCV, Tkinter
        """

        messagebox.showinfo("About RubikSense", about_text)

    def on_closing(self):
        """Handle application closing"""
        try:
            # Stop camera if running
            if hasattr(self, 'camera_frame'):
                self.camera_frame.stop_camera()

            # Release camera resources
            if self.camera_manager:
                self.camera_manager.release()

            # Save settings
            self.settings.save_config()

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            self.root.destroy()

    def run(self):
        """Start the application main loop"""
        try:
            self.update_status("Application started")
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            messagebox.showerror("Application Error", f"An error occurred: {e}")
            self.on_closing()
