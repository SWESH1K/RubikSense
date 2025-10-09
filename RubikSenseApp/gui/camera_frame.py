"""
Camera Frame Component for Calibration Tab

Handles camera display, calibration wizard, and cube detection visualization.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json
from typing import Optional, Dict, Any

from ..vision.cube_detector import CubeDetector, CameraManager

class CameraFrame(ttk.Frame):
    """Camera display and calibration interface"""
    
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.settings = settings
        self.camera_manager = None
        self.cube_detector = None
        self.is_running = False
        self.current_frame = None
        self.calibration_mode = False
        self.calibration_step = 0
        self.color_order = ["WHITE", "RED", "BLUE", "YELLOW", "ORANGE", "GREEN"]
        
        self.setup_ui()
        self.initialize_camera()
        
    def setup_ui(self):
        """Setup the camera frame UI"""
        # Configure grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        
        # Left panel - Camera display
        self.camera_panel = ttk.LabelFrame(self, text="Camera Feed")
        self.camera_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Camera display label with larger font
        self.camera_label = ttk.Label(self.camera_panel, text="Camera feed will appear here", font=("Segoe UI", 12))
        self.camera_label.pack(expand=True)
        
        # Right panel - Controls
        self.control_panel = ttk.LabelFrame(self, text="Controls")
        self.control_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Camera controls
        camera_frame = ttk.Frame(self.control_panel)
        camera_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(camera_frame, text="Camera Controls", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        
        btn_frame = ttk.Frame(camera_frame)
        btn_frame.pack(fill="x", pady=5)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(side="left", padx=2)
        
        self.stop_btn = ttk.Button(btn_frame, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.pack(side="left", padx=2)
        
        # Calibration controls
        cal_frame = ttk.Frame(self.control_panel)
        cal_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(cal_frame, text="Calibration Wizard", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        
        self.start_cal_btn = ttk.Button(cal_frame, text="Start Calibration", command=self.start_calibration)
        self.start_cal_btn.pack(fill="x", pady=2)
        
        self.save_cal_btn = ttk.Button(cal_frame, text="Save Calibration", command=self.save_calibration, state="disabled")
        self.save_cal_btn.pack(fill="x", pady=2)
        
        self.reset_cal_btn = ttk.Button(cal_frame, text="Reset Calibration", command=self.reset_calibration)
        self.reset_cal_btn.pack(fill="x", pady=2)
        
        # Status display
        status_frame = ttk.Frame(self.control_panel)
        status_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(status_frame, text="Status", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        
        self.status_text = tk.Text(status_frame, height=8, width=30, state="disabled", font=("Segoe UI", 10))
        self.status_text.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
    def initialize_camera(self):
        """Initialize camera and cube detector"""
        try:
            camera_config = self.settings.config.get("camera", {})
            self.camera_manager = CameraManager(
                device_id=camera_config.get("device_id", "http://192.168.29.220:8080/video"),
                fallback_device_id=camera_config.get("fallback_device_id", 0),
                width=camera_config.get("frame_width", 640),  # Reduced for GUI
                height=camera_config.get("frame_height", 480)
            )
            
            detection_config = self.settings.config.get("detection", {})
            self.cube_detector = CubeDetector(detection_config)
            
            # Load existing calibration
            calibration_data = self.settings.load_calibration()
            if calibration_data:
                self.cube_detector.load_calibration(calibration_data)
                self.log_status("Existing calibration loaded")
            else:
                self.log_status("No calibration found - please calibrate")
                
        except Exception as e:
            self.log_status(f"Failed to initialize camera: {e}")
            
    def log_status(self, message: str):
        """Add a message to the status display"""
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.status_text.config(state="normal")
        self.status_text.insert("end", full_message)
        self.status_text.see("end")
        self.status_text.config(state="disabled")
        
    def start_camera(self):
        """Start camera feed"""
        if not self.camera_manager.open():
            messagebox.showerror("Camera Error", 
                               "Failed to connect to both IP webcam and local camera.\n"
                               "Please check:\n"
                               "1. IP Webcam app is running on phone\n"
                               "2. Phone and computer are on same network\n"
                               "3. Local camera is not being used by another app")
            return
            
        self.is_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        
        # Show which camera is being used
        if isinstance(self.camera_manager.active_device, str) and "http" in self.camera_manager.active_device:
            camera_type = f"IP Webcam: {self.camera_manager.active_device}"
        else:
            camera_type = f"Local Camera: {self.camera_manager.active_device}"
            
        self.log_status(f"Camera started - {camera_type}")
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
    def stop_camera(self):
        """Stop camera feed"""
        self.is_running = False
        if self.camera_manager:
            self.camera_manager.release()
            
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        
        # Clear camera display
        self.camera_label.configure(image="", text="Camera stopped")
        
        self.log_status("Camera stopped")
        
    def camera_loop(self):
        """Main camera processing loop"""
        while self.is_running:
            try:
                ret, frame = self.camera_manager.read_frame()
                if not ret:
                    break
                    
                self.current_frame = frame.copy()
                
                # Process frame for cube detection
                if self.cube_detector and self.cube_detector.average_hsv:
                    self.process_frame(frame)
                    
                # Convert to display format
                self.display_frame(frame)
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                self.log_status(f"Camera error: {e}")
                break
                
        self.is_running = False
        
    def process_frame(self, frame):
        """Process frame for cube detection"""
        try:
            # Preprocess frame
            edged = self.cube_detector.preprocess_frame(frame)
            
            # Find cube contour
            contour = self.cube_detector.find_largest_square_contour(edged)
            
            if contour is not None:
                # Draw contour
                cv2.polylines(frame, [contour], True, (0, 255, 0), 2)
                
                # Draw grid overlay
                self.cube_detector.draw_cube_overlay(frame, contour)
                
                # Get colors if calibrated
                if self.cube_detector.average_hsv:
                    warped = self.cube_detector.warp_perspective(frame, contour)
                    colors = self.cube_detector.get_cubelet_colors(warped)
                    
                    # Draw color preview
                    self.cube_detector.draw_color_preview(frame, colors, "top_right")
                    
        except Exception as e:
            self.log_status(f"Processing error: {e}")
            
    def display_frame(self, frame):
        """Display frame in GUI"""
        try:
            # Resize frame to fit display
            height, width = frame.shape[:2]
            max_width, max_height = 640, 480
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.log_status(f"Display error: {e}")
            
    def start_calibration(self):
        """Start calibration wizard"""
        if not self.is_running:
            messagebox.showwarning("Camera Required", "Please start the camera first")
            return
            
        self.calibration_mode = True
        self.calibration_step = 0
        self.calibration_data = {}
        
        self.start_cal_btn.config(state="disabled")
        self.save_cal_btn.config(state="disabled")
        
        self.log_status("Calibration started")
        self.log_status(f"Show {self.color_order[0]} face and press SPACE to capture")
        
        # Bind keyboard events
        self.focus_set()
        self.bind("<KeyPress-space>", self.capture_calibration_frame)
        
    def capture_calibration_frame(self, event=None):
        """Capture current frame for calibration"""
        if not self.calibration_mode or self.current_frame is None:
            return
            
        try:
            # Find cube contour
            edged = self.cube_detector.preprocess_frame(self.current_frame)
            contour = self.cube_detector.find_largest_square_contour(edged)
            
            if contour is None:
                self.log_status("No cube face detected - please try again")
                return
                
            # Extract and get HSV values
            warped = self.cube_detector.warp_perspective(self.current_frame, contour)
            hsv_values = self.cube_detector.get_center_hsv_values(warped)
            
            # Store calibration data
            color_name = self.color_order[self.calibration_step]
            self.calibration_data[color_name] = hsv_values
            
            self.log_status(f"Captured {color_name} face")
            
            # Move to next step
            self.calibration_step += 1
            
            if self.calibration_step < len(self.color_order):
                next_color = self.color_order[self.calibration_step]
                self.log_status(f"Show {next_color} face and press SPACE to capture")
            else:
                # Calibration complete
                self.calibration_mode = False
                self.save_cal_btn.config(state="normal")
                self.start_cal_btn.config(state="normal")
                self.log_status("Calibration complete! You can now save it.")
                
        except Exception as e:
            self.log_status(f"Calibration error: {e}")
            
    def save_calibration(self):
        """Save calibration data"""
        if not hasattr(self, 'calibration_data') or not self.calibration_data:
            messagebox.showwarning("No Data", "No calibration data to save")
            return
            
        try:
            self.settings.save_calibration(self.calibration_data)
            self.cube_detector.load_calibration(self.calibration_data)
            
            self.save_cal_btn.config(state="disabled")
            self.log_status("Calibration saved successfully!")
            
            messagebox.showinfo("Success", "Calibration saved successfully!")
            
        except Exception as e:
            self.log_status(f"Save error: {e}")
            messagebox.showerror("Save Error", f"Failed to save calibration: {e}")
            
    def reset_calibration(self):
        """Reset calibration data"""
        self.calibration_mode = False
        self.calibration_step = 0
        self.calibration_data = {}
        
        self.start_cal_btn.config(state="normal")
        self.save_cal_btn.config(state="disabled")
        
        self.log_status("Calibration reset")
        
    def import_calibration(self, filename: str):
        """Import calibration from file"""
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
                
            self.settings.save_calibration(calibration_data)
            if self.cube_detector:
                self.cube_detector.load_calibration(calibration_data)
                
            self.log_status(f"Calibration imported from {filename}")
            messagebox.showinfo("Success", "Calibration imported successfully!")
            
        except Exception as e:
            self.log_status(f"Import error: {e}")
            messagebox.showerror("Import Error", f"Failed to import calibration: {e}")
            
    def export_calibration(self, filename: str):
        """Export calibration to file"""
        try:
            calibration_data = self.settings.load_calibration()
            if not calibration_data:
                messagebox.showwarning("No Data", "No calibration data to export")
                return
                
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)
                
            self.log_status(f"Calibration exported to {filename}")
            messagebox.showinfo("Success", "Calibration exported successfully!")
            
        except Exception as e:
            self.log_status(f"Export error: {e}")
            messagebox.showerror("Export Error", f"Failed to export calibration: {e}")
            
    def on_tab_selected(self):
        """Called when this tab is selected"""
        self.log_status("Calibration tab selected")