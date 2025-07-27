import cv2
import numpy as np
import json


# Compute average HSV for each face
AVERAGE_HSV = {}

RUBIKS_COLORS = {
    "WHITE": (255, 255, 255),
    "YELLOW": (255, 255, 0),
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "BLUE": (0, 0, 255),
    "ORANGE": (255, 165, 0)
}

# Use color names instead of face positions
COLOR_ORDER = ["WHITE", "RED", "BLUE", "YELLOW", "ORANGE", "GREEN"]
SAVE_PATH = "cube_calibration.json"

URL = "http://192.168.29.220:8080/video"
FRAME_SIZE = (1280, 720)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

def find_largest_square_contour(edged, min_area=10000):
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area, best_approx = 0, None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > max_area and area >= min_area:
                max_area = area
                best_approx = approx
    return best_approx

def warp_perspective(frame, corners):
    corners = corners.reshape((4, 2))
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    top_left = corners[np.argmin(s)]
    bottom_right = corners[np.argmax(s)]
    top_right = corners[np.argmin(diff)]
    bottom_left = corners[np.argmax(diff)]

    src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    dst_pts = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(frame, matrix, (300, 300))

def get_center_hsv(warped_face):
    hsv_face = cv2.cvtColor(warped_face, cv2.COLOR_BGR2HSV)
    h, w = hsv_face.shape[:2]
    cell_size = h // 3
    hsv_values = []
    for row in range(3):
        for col in range(3):
            cx = col * cell_size + cell_size // 2
            cy = row * cell_size + cell_size // 2
            roi = hsv_face[cy-2:cy+3, cx-2:cx+3]
            avg_hsv = np.mean(roi.reshape(-1, 3), axis=0)
            hsv_values.append(avg_hsv.tolist())
    return hsv_values

def draw_cubelets(frame, corners):
    if corners.shape != (4, 1, 2):
        return

    corners = corners.reshape((4, 2))
    top_left, top_right, bottom_right, bottom_left = corners

    def interp(p1, p2, t):
        return (1 - t) * np.array(p1) + t * np.array(p2)

    for i in range(1, 3):
        pt1 = tuple(interp(top_left, bottom_left, i / 3).astype(int))
        pt2 = tuple(interp(top_right, bottom_right, i / 3).astype(int))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 1)

        pt1 = tuple(interp(top_left, top_right, i / 3).astype(int))
        pt2 = tuple(interp(bottom_left, bottom_right, i / 3).astype(int))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 1)

def classify_color(hsv):
    min_dist = float('inf')
    closest_face = None
    for face, ref_hsv in AVERAGE_HSV.items():
        dist = np.linalg.norm(np.array(hsv) - np.array(ref_hsv))
        if dist < min_dist:
            min_dist = dist
            closest_face = face
    return closest_face

def get_cubelet_colors(warped_face):
    height, width, _ = warped_face.shape
    step_y, step_x = height // 3, width // 3
    colors = []

    for row in range(3):
        row_colors = []
        for col in range(3):
            y1 = row * step_y + step_y // 2 - 2
            y2 = row * step_y + step_y // 2 + 3
            x1 = col * step_x + step_x // 2 - 2
            x2 = col * step_x + step_x // 2 + 3

            roi = warped_face[y1:y2, x1:x2]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_hsv = np.mean(roi_hsv, axis=(0, 1))
            label = classify_color(avg_hsv)
            row_colors.append(label)
        colors.append(row_colors)
    return colors

def draw_color_gui(frame, colors):
    gui_size = 150
    cell_size = gui_size // 3
    x_offset = frame.shape[1] - gui_size - 20
    y_offset = 20

    for row, row_colors in enumerate(colors):
        for col, color_name in enumerate(row_colors):
            top_left = (x_offset + col * cell_size, y_offset + row * cell_size)
            bottom_right = (top_left[0] + cell_size, top_left[1] + cell_size)

            # Draw filled rectangle with color
            color = RUBIKS_COLORS.get(color_name, (0, 0, 0))
            color = tuple(reversed(color))  # Convert RGB to BGR
            cv2.rectangle(frame, top_left, bottom_right, color, -1)
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), 1)  # black border

            # Draw text at center of each cell
            text_size = cv2.getTextSize(color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = top_left[0] + (cell_size - text_size[0]) // 2
            text_y = top_left[1] + (cell_size + text_size[1]) // 2
            cv2.putText(frame, color_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return frame

def calibrate_cube():
    cap = cv2.VideoCapture(URL)
    calibration = {}

    for color_name in COLOR_ORDER:
        print(f"\n➡️  Show the '{color_name}' face (center should be {color_name}). Press SPACE to capture.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, FRAME_SIZE)
            edged = preprocess_frame(frame)
            contour = find_largest_square_contour(edged)

            if contour is not None:
                cv2.polylines(frame, [contour], True, (0, 255, 0), 2)

            cv2.putText(frame, f"Show {color_name} face, press SPACE to capture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == 32 and contour is not None:  # SPACE to capture
                warped = warp_perspective(frame, contour)
                calibration[color_name] = get_center_hsv(warped)
                print(f"✅ Captured {color_name} face.")
                break

    cap.release()
    cv2.destroyAllWindows()

    with open(SAVE_PATH, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\n✅ Calibration complete. Saved to {SAVE_PATH}")

def classify_cube():
    # Load calibration HSV data from file
    with open("cube_calibration.json", "r") as f:
        calibration_data = json.load(f)
        
    for face, values in calibration_data.items():
        hsv_array = np.array(values, dtype=np.float32)
        mean_hsv = np.mean(hsv_array, axis=0)
        AVERAGE_HSV[face] = mean_hsv

    cap = cv2.VideoCapture(URL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, FRAME_SIZE)

        edged = preprocess_frame(frame)
        best_contour = find_largest_square_contour(edged)

        if best_contour is not None:
            warped_face = warp_perspective(frame, best_contour)
            cv2.polylines(frame, [best_contour], True, (0, 255, 0), 2)  # Green boundary for cube face
            draw_cubelets(frame, best_contour)

            colors = get_cubelet_colors(warped_face)
            frame = draw_color_gui(frame, colors)

        cv2.imshow("Rubik's Cube Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.imwrite(f"captured/frame_{i}.jpg", frame)
            cv2.imwrite(f"captured/warped_face_{i}.jpg", warped_face)
            print(f"[INFO] Saved frame_{i}.jpg and warped_face_{i}.jpg")
            i += 1

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    calibrate_cube()
    classify_cube()