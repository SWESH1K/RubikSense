"""
Settings Dialog - Stub Implementation

Basic settings dialog that can be expanded with full functionality.
"""

import tkinter as tk
from tkinter import ttk, messagebox

class SettingsDialog:
    """Settings configuration dialog stub"""
    
    def __init__(self, parent, settings, category="general"):
        self.parent = parent
        self.settings = settings
        self.category = category
        self.result = False
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Settings - {category.title()}")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog
        self.dialog.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - 200
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - 150
        self.dialog.geometry(f"400x300+{x}+{y}")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup dialog UI"""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title with larger font
        title_label = ttk.Label(main_frame, text=f"{self.category.title()} Settings",
                               font=("Segoe UI", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Content based on category
        if self.category == "camera":
            self.create_camera_settings(main_frame)
        elif self.category == "detection":
            self.create_detection_settings(main_frame)
        else:
            self.create_general_settings(main_frame)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side="bottom", fill="x", pady=(20, 0))
        
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=self.cancel)
        cancel_btn.pack(side="right", padx=(5, 0))
        
        ok_btn = ttk.Button(button_frame, text="OK", command=self.ok)
        ok_btn.pack(side="right")
        
    def create_camera_settings(self, parent):
        """Create camera settings controls"""
        # Current settings display
        current_device = self.settings.get("camera.device_id", "http://192.168.29.220:8080/video")
        current_fallback = self.settings.get("camera.fallback_device_id", 0)
        
        # IP Webcam settings
        ip_frame = ttk.LabelFrame(parent, text="IP Webcam Settings")
        ip_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(ip_frame, text="IP Webcam URL:", font=("Segoe UI", 11)).pack(anchor="w", padx=10, pady=(5,0))
        self.ip_url_var = tk.StringVar(value=current_device if isinstance(current_device, str) else "http://192.168.29.220:8080/video")
        ip_entry = ttk.Entry(ip_frame, textvariable=self.ip_url_var, width=40, font=("Segoe UI", 10))
        ip_entry.pack(fill="x", padx=10, pady=(2,10))
        
        # Local camera fallback
        fallback_frame = ttk.LabelFrame(parent, text="Fallback Camera")
        fallback_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(fallback_frame, text="Local Camera ID (fallback):", font=("Segoe UI", 11)).pack(anchor="w", padx=10, pady=(5,0))
        self.fallback_var = tk.StringVar(value=str(current_fallback))
        fallback_entry = ttk.Entry(fallback_frame, textvariable=self.fallback_var, width=10, font=("Segoe UI", 10))
        fallback_entry.pack(anchor="w", padx=10, pady=(2,10))
        
        # Instructions
        instructions = ttk.Label(parent, 
            text="Instructions:\n" +
                 "1. Install 'IP Webcam' app on your phone\n" +
                 "2. Connect phone and computer to same WiFi\n" +
                 "3. Start IP Webcam, note the URL shown\n" +
                 "4. Enter the URL above (e.g., http://192.168.1.100:8080/video)\n" +
                 "5. Set fallback to 0 for first local camera, 1 for second, etc.",
            font=("Segoe UI", 9), justify="left")
        instructions.pack(padx=10, pady=10)
        
        # Store references for later use
        self.ip_entry = ip_entry
        self.fallback_entry = fallback_entry
        
    def create_detection_settings(self, parent):
        """Create detection settings controls"""
        info_label = ttk.Label(parent, text="Detection settings will be implemented here.", 
                              font=("Segoe UI", 11))
        info_label.pack(pady=20)
        
    def create_general_settings(self, parent):
        """Create general settings controls"""
        info_label = ttk.Label(parent, text="General preferences will be implemented here.", 
                              font=("Segoe UI", 11))
        info_label.pack(pady=20)
        
    def ok(self):
        """Handle OK button"""
        try:
            # Save camera settings if this is camera settings dialog
            if self.category == "camera" and hasattr(self, 'ip_url_var'):
                new_url = self.ip_url_var.get().strip()
                new_fallback = self.fallback_var.get().strip()
                
                if new_url:
                    self.settings.set("camera.device_id", new_url)
                    
                if new_fallback.isdigit():
                    self.settings.set("camera.fallback_device_id", int(new_fallback))
                    
            self.result = True
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
        
    def cancel(self):
        """Handle Cancel button"""
        self.result = False
        self.dialog.destroy()