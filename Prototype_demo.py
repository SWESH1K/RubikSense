# integrated_rubiks_gamified.py
# Gamified wrapper around the user's original integrated_rubiks.py
# NOTE: Core logic and flow are preserved exactly as in the original file.
# This file only adds visual GUI features and animations on top of the frames.

import time
import cv2
import numpy as np
import json
import os
import random

# --- Configuration ---
URL = "http://192.168.29.220:8080/video"
# URL = 1
# FRAME_SIZE = (1280, 720)
FRAME_SIZE = (640, 480)
SAVE_PATH = "cube_calibration.json"
WINDOW_NAME = "Rubik's Cube"
BEST_TIME_PATH = "best_time.json"

RUBIKS_COLORS = {
    "WHITE": (255, 255, 255),
    "YELLOW": (255, 255, 0),
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "BLUE": (0, 0, 255),
    "ORANGE": (255, 165, 0)
}
# Order used during calibration/capture (use color names)
COLOR_ORDER = ["WHITE", "RED", "BLUE", "YELLOW", "ORANGE", "GREEN"]

# --- Globals ---
AVERAGE_HSV = {}

# ----------------- ORIGINAL HELPERS (kept intact) -----------------

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
            roi = hsv_face[max(cy-2,0):cy+3, max(cx-2,0):cx+3]
            avg_hsv = np.mean(roi.reshape(-1, 3), axis=0)
            hsv_values.append(avg_hsv.tolist())
    return hsv_values


def draw_cubelets(frame, corners):
    if corners is None:
        return
    if corners.shape != (4, 1, 2) and corners.shape != (4, 2):
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
            y1, y2 = max(0,y1), min(height,y2)
            x1, x2 = max(0,x1), min(width,x2)
            roi = warped_face[y1:y2, x1:x2]
            if roi.size == 0:
                avg_hsv = np.array([0,0,0])
            else:
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                avg_hsv = np.mean(roi_hsv, axis=(0, 1))
            label = classify_color(avg_hsv)
            row_colors.append(label)
        colors.append(row_colors)
    return colors

def draw_color_gui(frame, colors, pos="top_right"):
    gui_size = 150
    cell_size = gui_size // 3
    if pos == "top_right":
        x_offset = frame.shape[1] - gui_size - 20
        y_offset = 20
    else: # bottom_left
        x_offset = 20
        y_offset = frame.shape[0] - gui_size - 20
    for row, row_colors in enumerate(colors):
        for col, color_name in enumerate(row_colors):
            top_left = (x_offset + col * cell_size, y_offset + row * cell_size)
            bottom_right = (top_left[0] + cell_size, top_left[1] + cell_size)
            color = RUBIKS_COLORS.get(color_name, (0, 0, 0))
            color = tuple(reversed(color))
            cv2.rectangle(frame, top_left, bottom_right, color, -1)
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), 1)
            text_size = cv2.getTextSize(str(color_name), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = top_left[0] + (cell_size - text_size[0]) // 2
            text_y = top_left[1] + (cell_size + text_size[1]) // 2
            cv2.putText(frame, str(color_name), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return frame

# ----------------- NEW VISUAL / ANIMATION HELPERS -----------------

def load_best_time(path=BEST_TIME_PATH):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            return float(data.get('best_time'))
    except Exception:
        return None


def save_best_time(best, path=BEST_TIME_PATH):
    try:
        with open(path, 'w') as f:
            json.dump({'best_time': best}, f)
    except Exception:
        pass


class ConfettiParticle:
    def __init__(self, frame_w, frame_h):
        self.x = random.randint(0, frame_w)
        self.y = random.randint(-frame_h//2, 0)
        self.vy = random.uniform(2, 6)
        self.size = random.randint(4, 10)
        self.col = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
        self.life = random.randint(40, 120)

    def update(self):
        self.y += self.vy
        self.vy += 0.05
        self.life -= 1

    def draw(self, frame):
        if self.life>0:
            cv2.circle(frame, (int(self.x), int(self.y)), self.size, self.col, -1)


def trigger_confetti(frame, particles, amount=80):
    h, w = frame.shape[:2]
    for _ in range(amount):
        particles.append(ConfettiParticle(w, h))


def draw_confetti(frame, particles):
    for p in particles[:]:
        p.update()
        p.draw(frame)
        if p.life <= 0 or p.y > frame.shape[0] + 50:
            particles.remove(p)


# progress bar - smooth fill tied to elapsed time and a soft max

def draw_progress_bar(frame, elapsed, max_time=60):
    h, w = frame.shape[:2]
    bar_w = int((w - 120) * min(elapsed / max_time, 1.0))
    y = h - 60
    cv2.rectangle(frame, (60, y), (w - 60, y + 20), (50, 50, 50), -1)
    cv2.rectangle(frame, (60, y), (60 + bar_w, y + 20), (0, 200, 0), -1)
    cv2.rectangle(frame, (60, y), (w - 60, y + 20), (0,0,0), 1)
    cv2.putText(frame, f"Time: {elapsed:.2f}s", (70, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


# Countdown big overlay

def draw_countdown(frame, val):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    alpha = 0.7
    cv2.rectangle(overlay, (0, 0), (w, h), (0,0,0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    text = "GO!" if val == 0 else str(val)
    font_scale = 4.5 if val == 0 else 3.0
    color = (0,255,0) if val == 0 else (0,255,255)
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 8)[0]
    cv2.putText(frame, text, ((w - size[0]) // 2, (h + size[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 8)


# Stage banner with fade effect

def show_stage_banner(frame, text, alpha=1.0):
    h, w = frame.shape[:2]
    banner_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, banner_h), (20,20,20), -1)
    cv2.addWeighted(overlay, alpha*0.6, frame, 1 - alpha*0.6, 0, frame)
    txt = f"{text}"
    size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    cv2.putText(frame, txt, ((w - size[0]) // 2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)


# Pulsing FPS

def draw_fps(frame, fps, t):
    h, w = frame.shape[:2]
    pulse = int((1 + 0.5 * np.sin(t * 6)) * 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (50, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), pulse)


# small checkmark animation used after capture

def draw_checkmark(frame, center, progress):
    # progress: 0.0 -> 1.0
    x, y = center
    length = int(40 * progress)
    # draw a simple checkmark using two lines
    if progress > 0:
        cv2.line(frame, (x - 20, y), (x - 20 + min(length, 20), y + min(length, 20)), (0,200,0), 4)
    if progress > 0.5:
        length2 = int(40 * (progress - 0.5) * 2)
        cv2.line(frame, (x, y + 20), (x + min(length2, 40), y - min(length2, 20)), (0,200,0), 4)


# ----------------- calibration / load (original with GUI overlays) -----------------

def calibrate_cube(save_path=SAVE_PATH):
    cap = cv2.VideoCapture(URL)
    calibration = {}
    for color_name in COLOR_ORDER:
        print(f"\n➡️  Show the '{color_name}' face (center should be {color_name}). Press SPACE to capture.")
        contour = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera read failed")
                break
            frame = cv2.resize(frame, FRAME_SIZE)
            edged = preprocess_frame(frame)
            contour = find_largest_square_contour(edged)
            if contour is not None:
                cv2.polylines(frame, [contour], True, (0, 255, 0), 2)
            show_stage_banner(frame, f"Calibration: Show {color_name} face")
            cv2.putText(frame, f"Show {color_name} face, press SPACE to capture", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cap.release()
                return False
            if key == 32 and contour is not None:
                warped = warp_perspective(frame, contour)
                calibration[color_name] = get_center_hsv(warped)
                # show checkmark animation for 0.8s
                start = time.time()
                while time.time() - start < 0.8:
                    ret2, f2 = cap.read()
                    if not ret2:
                        break
                    f2 = cv2.resize(f2, FRAME_SIZE)
                    draw_checkmark(f2, (FRAME_SIZE[0] - 200, 120), (time.time() - start) / 0.8)
                    show_stage_banner(f2, f"Captured {color_name}")
                    cv2.imshow(WINDOW_NAME, f2)
                    cv2.waitKey(1)
                print(f"✅ Captured {color_name} face.")
                break
    cap.release()
    with open(save_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\n✅ Calibration complete. Saved to {save_path}")
    return True


def load_calibration(save_path=SAVE_PATH):
    if not os.path.exists(save_path):
        return False
    with open(save_path, "r") as f:
        calibration_data = json.load(f)
    for face, values in calibration_data.items():
        hsv_array = np.array(values, dtype=np.float32)
        mean_hsv = np.mean(hsv_array, axis=0)
        AVERAGE_HSV[face] = mean_hsv
    return True


# ----------------- single-face scan (original with GUI overlay/checkmark) -----------------

def scan_single_face(prompt="Show face and press SPACE to capture"):
    cap = cv2.VideoCapture(URL)
    captured = None
    colors_preview = [[None]*3 for _ in range(3)]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        edged = preprocess_frame(frame)
        contour = find_largest_square_contour(edged)
        if contour is not None:
            cv2.polylines(frame, [contour], True, (0, 255, 0), 2)
            draw_cubelets(frame, contour)
            warped_face = warp_perspective(frame, contour)
            colors_preview = get_cubelet_colors(warped_face)
        show_stage_banner(frame, prompt)
        # draw colors in bottom left
        draw_color_gui(frame, colors_preview, pos="bottom_left")
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            captured = None
            break
        if key == 32 and contour is not None:
            warped_face = warp_perspective(frame, contour)
            colors = get_cubelet_colors(warped_face)
            # show check animation for 0.6s
            start = time.time()
            while time.time() - start < 0.6:
                ret2, f2 = cap.read()
                if not ret2:
                    break
                f2 = cv2.resize(f2, FRAME_SIZE)
                draw_checkmark(f2, (FRAME_SIZE[0] - 200, 120), (time.time() - start) / 0.6)
                show_stage_banner(f2, f"Captured face")
                draw_color_gui(f2, colors, pos="bottom_left")
                cv2.imshow(WINDOW_NAME, f2)
                cv2.waitKey(1)
            captured = colors
            break
    cap.release()
    return captured


# ----------------- capture full cube state (original with small GUI feedback) -----------------

def capture_cube_state(order=COLOR_ORDER):
    print("\n➡️  Capture the 6 faces in this order:")
    print(order)
    faces = {}
    for face_name in order:
        prompt = f"Show {face_name} face (center should be {face_name}), press SPACE"
        colors = scan_single_face(prompt)
        if colors is None:
            print("[WARN] Capture aborted or no contour. Aborting capture.")
            return None
        faces[face_name] = colors
        print(f"[INFO] Captured {face_name}")
        time.sleep(0.5)
    return faces


# ----------------- solved check (unchanged) -----------------

def is_solved_state(state_dict):
    if state_dict is None:
        return False
    for face_name, grid in state_dict.items():
        center = grid[1][1]
        if center is None:
            return False
        for r in range(3):
            for c in range(3):
                if grid[r][c] != center:
                    return False
    return True


# ----------------- textual state (unchanged) -----------------

def print_cube_state(state):
    for face, grid in state.items():
        print(f"\n{face}:")
        for row in grid:
            print(" ".join([str(x)[:3] if x else "None" for x in row]))


# ----------------- cube timer (original logic preserved, with overlays + countdown + progress) -----------------

def cube_timer_interactive():
    cap = cv2.VideoCapture(URL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    frame_width, frame_height = FRAME_SIZE
    zone_size = 100
    offset_x = 100
    left_zone_x = int(frame_width * 0.25 - zone_size // 2 + offset_x)
    right_zone_x = int(frame_width * 0.75 - zone_size // 2 + offset_x)
    y1 = frame_height - zone_size - 20
    y2 = frame_height - 20
    left_zone = (left_zone_x, y1, left_zone_x + zone_size, y2)
    right_zone = (right_zone_x, y1, right_zone_x + zone_size, y2)
    state = "IDLE"
    start_time = None
    elapsed_time = 0.0
    prev_time = time.time()
    frame_count = 0

    # for GUI features
    confetti_particles = []
    countdown_triggered = False
    countdown_start = None
    countdown_value = 3
    best_time = load_best_time()

    def is_blocked(frame, zone, threshold=0.3):
        x1, y1_, x2, y2_ = zone
        roi = frame[y1_:y2_, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.mean(gray < 100)
        return dark_ratio < threshold

    stopped_time = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame_count += 1
        left_blocked = is_blocked(frame, left_zone)
        right_blocked = is_blocked(frame, right_zone)

        # highlight zones when blocked
        if left_blocked:
            cv2.rectangle(frame, (left_zone[0], left_zone[1]), (left_zone[2], left_zone[3]), (0, 0, 255), -1)
        if right_blocked:
            cv2.rectangle(frame, (right_zone[0], right_zone[1]), (right_zone[2], right_zone[3]), (0, 0, 255), -1)

        # original state transitions (kept intact)
        if state == "IDLE":
            if left_blocked and right_blocked:
                state = "READY"
        elif state == "READY":
            if not left_blocked and not right_blocked:
                state = "RUNNING"
                start_time = time.time()
        elif state == "RUNNING":
            elapsed_time = time.time() - start_time
            if left_blocked and right_blocked:
                state = "STOPPED"
                elapsed_time = time.time() - start_time
                stopped_time = elapsed_time

        # --- GUI: Countdown when in READY (visual only) ---
        if state == "READY":
            if not countdown_triggered:
                countdown_triggered = True
                countdown_start = time.time()
            # compute countdown value (visual only)
            cd_elapsed = time.time() - countdown_start
            cd_val = max(0, 3 - int(cd_elapsed))
            # show countdown if still ongoing
            if cd_elapsed < 3:
                draw_countdown(frame, cd_val)
        else:
            countdown_triggered = False

        # draw overlays
        cv2.rectangle(frame, (left_zone[0], left_zone[1]), (left_zone[2], left_zone[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_zone[0], right_zone[1]), (right_zone[2], right_zone[3]), (0, 255, 0), 2)

        if state == "IDLE":
            cv2.putText(frame, "Place both hands (IDLE)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif state == "READY":
            cv2.putText(frame, "READY - Remove hands", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        elif state == "RUNNING":
            cv2.putText(frame, f"RUNNING: {elapsed_time:.2f}s", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            # progress bar
            draw_progress_bar(frame, elapsed_time, max_time=180)
        elif state == "STOPPED":
            cv2.putText(frame, f"STOPPED: {elapsed_time:.2f}s", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, "Press 'r' to reset or ESC to continue", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # trigger confetti once when stopped_time is first set
            if stopped_time is not None and len(confetti_particles) == 0:
                trigger_confetti(frame, confetti_particles, amount=120)
                # check and update best time
                if best_time is None or stopped_time < best_time:
                    best_time = stopped_time
                    save_best_time(best_time)

        # HUD / stage banner
        stage_text = f"State: {state}"
        show_stage_banner(frame, stage_text)

        # draw confetti if any
        if len(confetti_particles) > 0:
            draw_confetti(frame, confetti_particles)

        # draw best time
        if best_time is not None:
            cv2.putText(frame, f"Best: {best_time:.2f}s", (FRAME_SIZE[0] - 260, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2)

        # fps and pulsing
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0.0
        prev_time = curr_time
        t = time.time()
        draw_fps(frame, fps, t)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            state = "IDLE"
            start_time = None
            elapsed_time = 0.0
            stopped_time = None
            confetti_particles.clear()

    cap.release()
    return stopped_time


# ----------------- main integrated flow (kept logic intact; added visual confetti after solved) -----------------

def main():
    print("=== Integrated Rubik's Cube flow ===")
    if not load_calibration():
        print("[INFO] No calibration found. Starting calibration.")
        ok = calibrate_cube()
        if not ok:
            print("[ERROR] Calibration aborted. Exiting.")
            return
        load_calibration()
    else:
        print("[INFO] Calibration loaded.")

    # 1) Capture scrambled cube
    print("\nSTEP 1: Capture the scrambled cube (6 faces).")
    scrambled = capture_cube_state(order=COLOR_ORDER)
    if scrambled is None:
        print("[ERROR] Failed to capture scrambled cube. Exiting.")
        return
    print("\nScrambled state captured (text view):")
    print_cube_state(scrambled)
    solved_before = is_solved_state(scrambled)
    if solved_before:
        print("\n[INFO] Cube is already solved.")
    else:
        print("\n[INFO] Cube appears scrambled.")

    # 2) Timer
    print("\nSTEP 2: Solve the cube while timer runs.")
    stopped_time = cube_timer_interactive()
    if stopped_time is None:
        print("[INFO] Timer session aborted.")
        return
    print(f"\nTimer stopped at {stopped_time:.2f}s")

    # 3) Capture solved cube
    print("\nSTEP 3: Capture cube state again to check solved.")
    solved_state = capture_cube_state(order=COLOR_ORDER)
    if solved_state is None:
        print("[ERROR] Failed to capture solved cube.")
        return
    print("\nCaptured solved state (text view):")
    print_cube_state(solved_state)
    solved_after = is_solved_state(solved_state)
    if solved_after:
        print(f"\n✅ Cube solved in {stopped_time:.2f}s!")
        # show celebratory confetti in a short loop
        cap = cv2.VideoCapture(URL)
        particles = []
        start = time.time()
        while time.time() - start < 2.5:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, FRAME_SIZE)
            trigger_confetti(frame, particles, amount=10)
            draw_confetti(frame, particles)
            show_stage_banner(frame, "Solved! 🎉")
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)
        cap.release()
    else:
        print("\n❌ Cube not solved.")

    print("\n=== Session Complete ===")


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
