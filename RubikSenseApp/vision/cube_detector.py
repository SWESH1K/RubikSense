"""
Cube Detection and Computer Vision Module

Refactored from the original prototype_demo.py to provide clean, modular
computer vision functionality for Rubik's cube detection and color classification.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

class CubeDetector:
    """Handles Rubik's cube detection and color analysis"""
    
    # Standard Rubik's cube colors in BGR format for OpenCV
    RUBIKS_COLORS = {
        "WHITE": (255, 255, 255),
        "YELLOW": (0, 255, 255),  # BGR format
        "RED": (0, 0, 255),
        "GREEN": (0, 255, 0),
        "BLUE": (255, 0, 0),
        "ORANGE": (0, 165, 255)
    }
    
    # Standard color order for calibration
    COLOR_ORDER = ["WHITE", "RED", "BLUE", "YELLOW", "ORANGE", "GREEN"]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cube detector
        
        Args:
            config: Detection configuration parameters
        """
        self.config = config
        self.average_hsv = {}  # Calibrated color HSV values
        self.logger = logging.getLogger(__name__)
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for edge detection
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed binary image with edges
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        kernel_size = self.config.get("gaussian_blur_kernel", 5)
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # Edge detection
        low_thresh = self.config.get("canny_low_threshold", 50)
        high_thresh = self.config.get("canny_high_threshold", 150)
        edged = cv2.Canny(blur, low_thresh, high_thresh)
        
        # Morphological operations to close gaps
        morph_size = self.config.get("morph_kernel_size", 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
        return cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    def find_largest_square_contour(self, edged: np.ndarray, 
                                   min_area: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Find the largest square-like contour in the image
        
        Args:
            edged: Binary edge image
            min_area: Minimum area threshold for contours
            
        Returns:
            Contour points of the largest square, or None if not found
        """
        if min_area is None:
            min_area = self.config.get("min_contour_area", 10000)
            
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        best_approx = None
        
        for cnt in contours:
            # Approximate contour to reduce number of points
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Check if it's a quadrilateral and convex
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > max_area and area >= min_area:
                    max_area = area
                    best_approx = approx
                    
        return best_approx
    
    def warp_perspective(self, frame: np.ndarray, corners: np.ndarray, 
                        size: int = 300) -> np.ndarray:
        """
        Extract and warp the detected cube face to a square
        
        Args:
            frame: Input frame
            corners: Four corner points of the detected square
            size: Size of the output square
            
        Returns:
            Warped square image of the cube face
        """
        # Reshape corners and identify corner positions
        corners = corners.reshape((4, 2))
        
        # Calculate corner positions based on sums and differences
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)
        
        top_left = corners[np.argmin(s)]
        bottom_right = corners[np.argmax(s)]
        top_right = corners[np.argmin(diff)]
        bottom_left = corners[np.argmax(diff)]
        
        # Define source and destination points
        src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
        dst_pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype="float32")
        
        # Calculate perspective transform matrix and apply
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(frame, matrix, (size, size))
    
    def get_cubelet_colors(self, warped_face: np.ndarray) -> List[List[str]]:
        """
        Extract colors from a 3x3 grid of cubelets
        
        Args:
            warped_face: Warped square image of cube face
            
        Returns:
            3x3 grid of color names
        """
        height, width, _ = warped_face.shape
        step_y, step_x = height // 3, width // 3
        colors = []
        
        for row in range(3):
            row_colors = []
            for col in range(3):
                # Define ROI for this cubelet (center region)
                center_y = row * step_y + step_y // 2
                center_x = col * step_x + step_x // 2
                
                # Small region around center
                roi_size = 5
                y1 = max(0, center_y - roi_size // 2)
                y2 = min(height, center_y + roi_size // 2 + 1)
                x1 = max(0, center_x - roi_size // 2)
                x2 = min(width, center_x + roi_size // 2 + 1)
                
                roi = warped_face[y1:y2, x1:x2]
                
                if roi.size == 0:
                    avg_hsv = np.array([0, 0, 0])
                else:
                    # Convert to HSV and get average
                    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    avg_hsv = np.mean(roi_hsv, axis=(0, 1))
                
                # Classify the color
                color_name = self.classify_color(avg_hsv)
                row_colors.append(color_name)
                
            colors.append(row_colors)
            
        return colors
    
    def classify_color(self, hsv: np.ndarray) -> str:
        """
        Classify HSV color to nearest calibrated color
        
        Args:
            hsv: HSV color values
            
        Returns:
            Name of the closest calibrated color
        """
        if not self.average_hsv:
            return "UNKNOWN"
            
        min_dist = float('inf')
        closest_color = "UNKNOWN"
        
        for color_name, ref_hsv in self.average_hsv.items():
            # Calculate Euclidean distance in HSV space
            dist = np.linalg.norm(np.array(hsv) - np.array(ref_hsv))
            if dist < min_dist:
                min_dist = dist
                closest_color = color_name
                
        return closest_color
    
    def get_center_hsv_values(self, warped_face: np.ndarray) -> List[List[float]]:
        """
        Get HSV values from center of each cubelet for calibration
        
        Args:
            warped_face: Warped square image of cube face
            
        Returns:
            List of HSV values for each cubelet
        """
        hsv_face = cv2.cvtColor(warped_face, cv2.COLOR_BGR2HSV)
        height, width = hsv_face.shape[:2]
        cell_size = height // 3
        hsv_values = []
        
        for row in range(3):
            for col in range(3):
                # Center coordinates
                cx = col * cell_size + cell_size // 2
                cy = row * cell_size + cell_size // 2
                
                # Small ROI around center
                roi = hsv_face[max(cy-2, 0):cy+3, max(cx-2, 0):cx+3]
                avg_hsv = np.mean(roi.reshape(-1, 3), axis=0)
                hsv_values.append(avg_hsv.tolist())
                
        return hsv_values
    
    def load_calibration(self, calibration_data: Dict[str, Any]) -> bool:
        """
        Load color calibration data
        
        Args:
            calibration_data: Dictionary of color calibration values
            
        Returns:
            True if calibration loaded successfully
        """
        try:
            self.average_hsv = {}
            for color_name, values in calibration_data.items():
                hsv_array = np.array(values, dtype=np.float32)
                mean_hsv = np.mean(hsv_array, axis=0)
                self.average_hsv[color_name] = mean_hsv
            return True
        except Exception as e:
            self.logger.error(f"Failed to load calibration: {e}")
            return False
    
    def draw_cube_overlay(self, frame: np.ndarray, corners: np.ndarray):
        """
        Draw cube grid overlay on detected cube face
        
        Args:
            frame: Input frame to draw on
            corners: Corner points of detected cube face
        """
        if corners is None:
            return
            
        # Ensure correct shape
        if corners.shape != (4, 1, 2) and corners.shape != (4, 2):
            return
            
        corners = corners.reshape((4, 2))
        top_left, top_right, bottom_right, bottom_left = corners
        
        def interpolate(p1, p2, t):
            """Linear interpolation between two points"""
            return (1 - t) * np.array(p1) + t * np.array(p2)
        
        # Draw grid lines
        for i in range(1, 3):
            # Horizontal lines
            pt1 = tuple(interpolate(top_left, bottom_left, i / 3).astype(int))
            pt2 = tuple(interpolate(top_right, bottom_right, i / 3).astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # Vertical lines
            pt1 = tuple(interpolate(top_left, top_right, i / 3).astype(int))
            pt2 = tuple(interpolate(bottom_left, bottom_right, i / 3).astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    def draw_color_preview(self, frame: np.ndarray, colors: List[List[str]], 
                          position: str = "top_right", size: int = 150):
        """
        Draw color grid preview on frame
        
        Args:
            frame: Frame to draw on
            colors: 3x3 grid of color names
            position: Where to place the preview ("top_right", "bottom_left", etc.)
            size: Size of the preview grid
        """
        cell_size = size // 3
        
        # Determine position
        if position == "top_right":
            x_offset = frame.shape[1] - size - 20
            y_offset = 20
        elif position == "bottom_left":
            x_offset = 20
            y_offset = frame.shape[0] - size - 20
        else:
            x_offset, y_offset = 20, 20
        
        # Draw each cell
        for row, row_colors in enumerate(colors):
            for col, color_name in enumerate(row_colors):
                # Cell boundaries
                top_left = (x_offset + col * cell_size, y_offset + row * cell_size)
                bottom_right = (top_left[0] + cell_size, top_left[1] + cell_size)
                
                # Get color (convert BGR to RGB for display)
                color_bgr = self.RUBIKS_COLORS.get(color_name, (128, 128, 128))
                
                # Draw filled rectangle
                cv2.rectangle(frame, top_left, bottom_right, color_bgr, -1)
                # Draw border
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), 1)
                
                # Add text label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.3
                font_thickness = 1
                
                text_size = cv2.getTextSize(color_name[:3], font, font_scale, font_thickness)[0]
                text_x = top_left[0] + (cell_size - text_size[0]) // 2
                text_y = top_left[1] + (cell_size + text_size[1]) // 2
                
                cv2.putText(frame, color_name[:3], (text_x, text_y), 
                           font, font_scale, (0, 0, 0), font_thickness)

class CameraManager:
    """Manages camera connection and frame capture"""
    
    def __init__(self, device_id = 0, fallback_device_id: int = None, width: int = 1280, height: int = 720):
        """
        Initialize camera manager
        
        Args:
            device_id: Camera device ID (int) or URL (str) for IP camera
            fallback_device_id: Fallback local camera ID if primary fails
            width: Frame width
            height: Frame height
        """
        self.device_id = device_id
        self.fallback_device_id = fallback_device_id
        self.width = width
        self.height = height
        self.cap = None
        self.is_opened = False
        self.active_device = None  # Track which device is actually being used
        
    def open(self) -> bool:
        """
        Open camera connection, trying IP webcam first then fallback
        
        Returns:
            True if camera opened successfully
        """
        # Try primary device (IP webcam or local camera)
        try:
            logging.info(f"Attempting to connect to primary camera: {self.device_id}")
            self.cap = cv2.VideoCapture(self.device_id)
            
            # Test if the camera actually works
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    # Primary camera works
                    self._configure_camera_properties()
                    self.is_opened = True
                    self.active_device = self.device_id
                    logging.info(f"Successfully connected to primary camera: {self.device_id}")
                    return True
                else:
                    logging.warning("Primary camera opened but failed to read frame")
                    self.cap.release()
            else:
                logging.warning("Primary camera failed to open")
                
        except Exception as e:
            logging.warning(f"Primary camera connection failed: {e}")
            if self.cap:
                self.cap.release()
        
        # Try fallback device if available
        if self.fallback_device_id is not None:
            try:
                logging.info(f"Attempting to connect to fallback camera: {self.fallback_device_id}")
                self.cap = cv2.VideoCapture(self.fallback_device_id)
                
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        # Fallback camera works
                        self._configure_camera_properties()
                        self.is_opened = True
                        self.active_device = self.fallback_device_id
                        logging.info(f"Successfully connected to fallback camera: {self.fallback_device_id}")
                        return True
                    else:
                        logging.error("Fallback camera opened but failed to read frame")
                        self.cap.release()
                else:
                    logging.error("Fallback camera failed to open")
                    
            except Exception as e:
                logging.error(f"Fallback camera connection failed: {e}")
                if self.cap:
                    self.cap.release()
        
        # Both cameras failed
        logging.error("All camera connection attempts failed")
        self.cap = None
        self.is_opened = False
        self.active_device = None
        return False
    
    def _configure_camera_properties(self):
        """Configure camera properties like resolution and FPS"""
        try:
            # Set camera properties (may not work for all camera types)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # For IP cameras, we might need to set buffer size
            if isinstance(self.active_device, str) and "http" in str(self.active_device).lower():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for IP cameras
                
        except Exception as e:
            logging.warning(f"Failed to set camera properties: {e}")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened or self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        
        if ret:
            # Resize frame to desired dimensions
            frame = cv2.resize(frame, (self.width, self.height))
            
        return ret, frame
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            self.cap = None

def is_solved_state(cube_state: Dict[str, List[List[str]]]) -> bool:
    """
    Check if the cube state represents a solved cube
    
    Args:
        cube_state: Dictionary of face colors
        
    Returns:
        True if cube appears solved
    """
    if cube_state is None:
        return False
        
    for face_name, grid in cube_state.items():
        # Get center color (this defines what color the face should be)
        center_color = grid[1][1]  # Center piece
        
        if center_color is None:
            return False
            
        # Check if all squares on this face match the center
        for row in range(3):
            for col in range(3):
                if grid[row][col] != center_color:
                    return False
                    
    return True