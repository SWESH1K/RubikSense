# RubikSenseApp - Modern GUI Application

A modern, user-friendly GUI application for Rubik's Cube recognition, color calibration, and solve timing.

## Features

### ✅ **Currently Implemented**
- **Modern Tabbed Interface** - Clean, responsive GUI with dark/light themes
- **Real-time Camera Feed** - Live camera display with cube detection
- **Interactive Calibration Wizard** - Step-by-step color calibration process
- **Cube Detection & Visualization** - Real-time cube face detection with grid overlay
- **Color Classification** - HSV-based color recognition system
- **Settings Management** - Persistent configuration storage
- **Import/Export** - Calibration data management

### 🚧 **Coming Soon**
- **Interactive Timer** - Hand gesture-controlled solve timer
- **Statistics & Analysis** - Solve time graphs and statistical breakdowns
- **Advanced Settings** - Detailed camera and detection parameter tuning
- **3D Cube Visualization** - Real-time 3D representation of detected cube state

## Installation

1. **Clone or navigate to the RubikSenseApp directory:**
   ```bash
   cd RubikSenseApp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

## Usage

### 1. **Camera Setup**

#### **Option A: IP Webcam (Recommended)**
- Install **"IP Webcam"** app on your Android phone
- Connect phone and computer to the **same WiFi network**
- Open IP Webcam app and tap **"Start Server"**
- Note the URL displayed (e.g., `http://192.168.1.100:8080/video`)
- Run configuration utility: `python configure_ip_webcam.py`
- Enter your phone's IP address when prompted

#### **Option B: Local Camera**
- Connect a USB webcam or use your built-in camera
- The app will automatically fallback to local camera if IP webcam fails

#### **Start Camera Feed**
- Go to the **Calibration** tab
- Click **"Start Camera"** to begin video feed
- The app will try IP webcam first, then fallback to local camera

### 2. **Cube Calibration**
- With the camera running, click **"Start Calibration"**
- Show each face of your cube as prompted:
  1. WHITE face (center square should be white)
  2. RED face
  3. BLUE face  
  4. YELLOW face
  5. ORANGE face
  6. GREEN face
- Press **SPACE** to capture each face
- Click **"Save Calibration"** when complete

### 3. **Cube Detection**
- Once calibrated, the app will automatically detect and classify cube colors
- Green grid overlay shows detected cube face
- Color preview shows recognized colors in real-time

### 4. **Timer (Coming Soon)**
- Interactive solve timer with hand gesture control
- Statistics tracking and personal best records

## Interface Overview

### 📹 **Calibration Tab**
- **Camera Feed**: Live video with cube detection overlay
- **Controls Panel**: Camera start/stop, calibration wizard
- **Status Log**: Real-time feedback and instructions

### ⏱️ **Timer Tab**
- Modern solve timer interface
- Start/stop controls
- Best time tracking

### 📊 **Analysis Tab**
- Solve statistics and graphs
- Data export functionality
- Session management

## Keyboard Shortcuts

- **Space**: Capture calibration frame / Start-stop timer
- **R**: Reset calibration / Reset timer  
- **S**: Save calibration
- **Ctrl+Q**: Quit application
- **F1**: Show help

## Settings & Configuration

The app automatically creates a configuration directory at `~/.rubiksense/` containing:
- `config.json` - Application settings
- `calibration.json` - Color calibration data
- `solve_history.json` - Solve time records
- `best_time.json` - Personal best time

## Themes

Choose from multiple visual themes:
- **Modern Dark** - Sleek dark interface (default)
- **Modern Light** - Clean light interface  
- **Classic** - Traditional system theme

Access themes via **View → Theme** in the menu bar.

## Font Size Configuration

If the font sizes are too small or large for your display, you can easily adjust them:

### **Quick Font Adjustment**
```bash
# From the RubikSenseApp directory
python configure_fonts.py
```

### **Font Size Presets**
- **Small fonts** - Good for small screens or laptops
- **Medium fonts** - Default, balanced for most displays
- **Large fonts** - Good for accessibility and readability
- **Extra large fonts** - Maximum readability
- **Custom sizes** - Set individual font sizes for different elements

### **What Each Font Size Controls**
- **Small text** - Status messages, log entries
- **Medium text** - Labels, buttons, normal text
- **Large text** - Tab headers, section titles
- **Title text** - Main section headings
- **Display text** - Timer display, large numbers

**Note:** Restart the application after changing font sizes to see the changes.

## Troubleshooting

### Camera Issues

#### **IP Webcam Problems**
- Ensure **IP Webcam app is running** on your phone
- Check that **both devices are on the same WiFi network**
- Verify the **IP address is correct** (run `python configure_ip_webcam.py`)
- Try accessing the URL in a web browser first
- Check your **router's firewall settings**

#### **Local Camera Problems**
- Ensure camera is **not being used by another application**
- Check **camera permissions** in system settings
- Try different camera device IDs (0, 1, 2, etc.) in Settings → Camera

### Calibration Problems
- Ensure good lighting conditions
- Hold cube steady during capture
- Make sure the cube face fills most of the detection area
- Clean cube faces for better color detection

### Performance
- Close unnecessary applications to free up camera resources
- Reduce camera resolution in settings if experiencing lag
- Ensure adequate system RAM for real-time processing

## System Requirements

- **Python 3.7+**
- **OpenCV 4.5+**
- **Webcam or USB camera**
- **Minimum 4GB RAM recommended**
- **Windows, macOS, or Linux**

## Development

Built with:
- **Python** - Core application logic
- **OpenCV** - Computer vision and camera handling
- **Tkinter** - Native GUI framework
- **PIL/Pillow** - Image processing for GUI display
- **NumPy** - Numerical computations

## License

This project is part of the RubikSense cube recognition system.

## Author

**SWESH1K** - Advanced Rubik's Cube Recognition & Timer

---

*For issues or feature requests, please refer to the main RubikSense project documentation.*