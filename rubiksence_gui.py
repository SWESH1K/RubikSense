# rubiks_timer_gui.py
import threading
import time
import queue
import json
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# Import helpers and constants from Prototype_demo.py (must be in same directory)
import Prototype_demo as core

# Configuration
FRAME_SIZE = core.FRAME_SIZE
URL = core.URL
COLOR_ORDER = core.COLOR_ORDER
RUBIKS_COLORS = core.RUBIKS_COLORS
BEST_TIME_PATH = core.BEST_TIME_PATH
SAVE_PATH = core.SAVE_PATH

# Helper to safely push logs to UI from threads
class Logger:
    def __init__(self, text_widget):
        self.q = queue.Queue()
        self.text_widget = text_widget
        self._update_ui()

    def log(self, msg):
        self.q.put(msg)

    def _update_ui(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, msg + "\n")
                self.text_widget.see(tk.END)
                self.text_widget.configure(state='disabled')
        except queue.Empty:
            pass
        self.text_widget.after(200, self._update_ui)


class VideoCaptureThread:
    def __init__(self, src=URL):
        self.src = src
        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        self.cap = cv2.VideoCapture(self.src)
        # attempt to set frame size if using numeric device
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        except Exception:
            pass
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.resize(frame, FRAME_SIZE)
            with self.lock:
                self.frame = frame.copy()
        try:
            self.cap.release()
        except Exception:
            pass

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False


class RubiksGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rubik's Cube Vision Timer")
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

        # Video capture
        self.vcap = VideoCaptureThread(URL)
        self.vcap.start()

        # Load calibration if exists
        core.AVERAGE_HSV.clear()
        core.load_calibration(SAVE_PATH)

       # Load best time BEFORE building UI (since label uses it)
        self.best_time = core.load_best_time(BEST_TIME_PATH)

        self._build_ui()

        # Logger
        self.logger = Logger(self.log_text)
        self.logger.log("GUI started. Camera URL: " + str(URL))
        if core.AVERAGE_HSV:
            self.logger.log("Loaded calibration from " + SAVE_PATH)
        else:
            self.logger.log("No calibration loaded. Please calibrate.")

        # Preview update loop
        self.running_preview = True
        self._update_preview()

        # State used for capture flows
        self.capture_mode = None
        self.calibration_index = 0
        self.capture_faces = {}
        self.capture_order = COLOR_ORDER.copy()

        # Timer state
        self.timer_thread = None
        self.timer_running = False
        self.current_time = 0.0


    def _build_ui(self):
        # Top frame holds preview and control panels
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left: video preview
        self.preview_label = tk.Label(top)
        self.preview_label.pack(side=tk.LEFT, padx=8, pady=8)
        # Drawn overlay size will match FRAME_SIZE

        # Right: controls
        ctrl = ttk.Frame(top)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        # Buttons
        btn_cal = ttk.Button(ctrl, text="Start Calibration", command=self.start_calibration)
        btn_cal.pack(fill=tk.X, pady=4)
        self.btn_capture_face = ttk.Button(ctrl, text="Capture Face (when contour visible)", command=self.capture_face)
        self.btn_capture_face.pack(fill=tk.X, pady=4)

        btn_scramble = ttk.Button(ctrl, text="Capture Scrambled Cube", command=self.start_capture_scrambled)
        btn_scramble.pack(fill=tk.X, pady=4)
        btn_solved = ttk.Button(ctrl, text="Capture Solved Cube", command=self.start_capture_solved)
        btn_solved.pack(fill=tk.X, pady=4)

        sep = ttk.Separator(ctrl, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=6)

        self.btn_start_timer = ttk.Button(ctrl, text="Start Timer", command=self.start_timer)
        self.btn_start_timer.pack(fill=tk.X, pady=4)
        self.btn_stop_timer = ttk.Button(ctrl, text="Stop Timer", command=self.stop_timer, state=tk.DISABLED)
        self.btn_stop_timer.pack(fill=tk.X, pady=4)

        btn_check = ttk.Button(ctrl, text="Check Solved?", command=self.check_solved)
        btn_check.pack(fill=tk.X, pady=4)
        btn_reset_best = ttk.Button(ctrl, text="Reset Best Time", command=self.reset_best_time)
        btn_reset_best.pack(fill=tk.X, pady=4)

        sep2 = ttk.Separator(ctrl, orient=tk.HORIZONTAL)
        sep2.pack(fill=tk.X, pady=6)

        self.best_label = ttk.Label(ctrl, text=f"Best: {self.best_time:.2f}s" if self.best_time else "Best: N/A")
        self.best_label.pack(fill=tk.X, pady=4)

        btn_exit = ttk.Button(ctrl, text="Exit", command=self.on_exit)
        btn_exit.pack(side=tk.BOTTOM, fill=tk.X, pady=8)

        # Bottom: log console
        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, padx=8, pady=4)
        self.log_text = scrolledtext.ScrolledText(bottom, height=8, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _update_preview(self):
        if not self.running_preview:
            return
        frame = self.vcap.read()
        if frame is not None:
            display = frame.copy()

            # Draw detection contour preview
            try:
                edged = core.preprocess_frame(display)
                contour = core.find_largest_square_contour(edged)
                if contour is not None:
                    # draw polygon
                    cv2.polylines(display, [contour], True, (0, 255, 0), 2)
                    # draw cubelet grid
                    core.draw_cubelets(display, contour)
                    # small preview of colors
                    warped = core.warp_perspective(display, contour)
                    preview_colors = core.get_cubelet_colors(warped)
                    core.draw_color_gui(display, preview_colors, pos="top_right")
            except Exception:
                pass

            # draw zones for timer preview (same as Prototype_demo)
            try:
                fw, fh = FRAME_SIZE
                zone_size = 100
                offset_x = 100
                left_zone_x = int(fw * 0.25 - zone_size // 2 + offset_x)
                right_zone_x = int(fw * 0.75 - zone_size // 2 + offset_x)
                y1 = fh - zone_size - 20
                y2 = fh - 20
                cv2.rectangle(display, (left_zone_x, y1), (left_zone_x+zone_size, y2), (0,255,0), 1)
                cv2.rectangle(display, (right_zone_x, y1), (right_zone_x+zone_size, y2), (0,255,0), 1)
            except Exception:
                pass

            # convert BGR->RGB then to PhotoImage
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_label.imgtk = imgtk
            self.preview_label.config(image=imgtk)

        self.after(30, self._update_preview)

    # ---------------- Calibration / Capture controls ----------------
    def start_calibration(self):
        if self.capture_mode is not None:
            self.logger.log("Another capture mode is active. Please finish it first.")
            return
        self.capture_mode = 'calibration'
        self.calibration_index = 0
        self.calibration_data = {}
        self.logger.log("Calibration started. Follow prompts and press 'Capture Face' when the cube face is centered.")
        self._prompt_current_calibration_color()

    def _prompt_current_calibration_color(self):
        color = COLOR_ORDER[self.calibration_index]
        self.logger.log(f"Calibration: Show the '{color}' face (center should be {color}). Then press 'Capture Face'.")

    def capture_face(self):
        frame = self.vcap.read()
        if frame is None:
            self.logger.log("No frame available to capture.")
            return
        edged = core.preprocess_frame(frame)
        contour = core.find_largest_square_contour(edged)
        if contour is None:
            self.logger.log("No cube contour detected. Adjust cube and try again.")
            return
        # warp and compute center hsv or colors
        warped = core.warp_perspective(frame, contour)
        if self.capture_mode == 'calibration':
            color_name = COLOR_ORDER[self.calibration_index]
            hsvvals = core.get_center_hsv(warped)
            self.calibration_data[color_name] = hsvvals
            self.logger.log(f"Captured calibration for {color_name}.")
            # animate checkmark via small message
            self.calibration_index += 1
            if self.calibration_index >= len(COLOR_ORDER):
                # save calibration
                try:
                    with open(SAVE_PATH, "w") as f:
                        json.dump(self.calibration_data, f, indent=2)
                    # reload into core.AVERAGE_HSV
                    core.AVERAGE_HSV.clear()
                    core.load_calibration(SAVE_PATH)
                    self.logger.log("Calibration complete and saved to " + SAVE_PATH)
                except Exception as e:
                    self.logger.log("Failed to save calibration: " + str(e))
                self.capture_mode = None
            else:
                self._prompt_current_calibration_color()

        elif self.capture_mode in ('capture_scrambled', 'capture_solved'):
            # capture one face for the next face in capture_order
            face_name = self.capture_order[len(self.capture_faces)]
            colors = core.get_cubelet_colors(warped)
            self.capture_faces[face_name] = colors
            self.logger.log(f"Captured face {face_name}. ({len(self.capture_faces)}/{len(self.capture_order)})")
            if len(self.capture_faces) >= len(self.capture_order):
                # finished capturing all faces
                mode = self.capture_mode
                faces_copy = dict(self.capture_faces)
                self.capture_mode = None
                self.capture_faces = {}
                # show result
                self.logger.log(f"Finished capturing all faces for {mode}.")
                # print textual state to console
                core.print_cube_state(faces_copy)
                # store last_captured
                self.last_captured = faces_copy
            else:
                next_face = self.capture_order[len(self.capture_faces)]
                self.logger.log(f"Next: Show {next_face} face and press 'Capture Face'.")

        else:
            self.logger.log("Not in any capture mode. Use 'Start Calibration' or 'Capture Scrambled/Solved' first.")

    def start_capture_scrambled(self):
        if self.capture_mode is not None:
            self.logger.log("Another capture mode is active. Please finish it first.")
            return
        if not core.AVERAGE_HSV:
            self.logger.log("Calibration missing. Calibrate first.")
            return
        self.capture_mode = 'capture_scrambled'
        self.capture_faces = {}
        self.capture_order = COLOR_ORDER.copy()
        self.logger.log("Capture Scrambled: Show faces in this order: " + ", ".join(self.capture_order))
        self.logger.log("Press 'Capture Face' when each face is centered (green contour visible).")

    def start_capture_solved(self):
        if self.capture_mode is not None:
            self.logger.log("Another capture mode is active. Please finish it first.")
            return
        if not core.AVERAGE_HSV:
            self.logger.log("Calibration missing. Calibrate first.")
            return
        self.capture_mode = 'capture_solved'
        self.capture_faces = {}
        self.capture_order = COLOR_ORDER.copy()
        self.logger.log("Capture Solved: Show faces in this order: " + ", ".join(self.capture_order))
        self.logger.log("Press 'Capture Face' when each face is centered (green contour visible).")

    # ---------------- Timer controls ----------------
    def start_timer(self):
        if self.timer_running:
            self.logger.log("Timer already running.")
            return
        # ensure calibration present
        if not core.AVERAGE_HSV:
            self.logger.log("Calibration missing. Calibrate first.")
            return
        self.timer_running = True
        self.btn_start_timer.config(state=tk.DISABLED)
        self.btn_stop_timer.config(state=tk.NORMAL)
        self.logger.log("Timer started. Place both hands in the green zones to begin.")
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()

    def stop_timer(self):
        if not self.timer_running:
            return
        self.timer_running = False
        self.btn_start_timer.config(state=tk.NORMAL)
        self.btn_stop_timer.config(state=tk.DISABLED)
        self.logger.log("Timer stopped by user.")

    def _timer_loop(self):
        # use same algorithm as Prototype_demo.cube_timer_interactive but adapted to embedded feed
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
        stopped_time = None
        confetti_particles = []

        def is_blocked(frame, zone, threshold=0.3):
            x1, y1_, x2, y2_ = zone
            roi = frame[y1_:y2_, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            dark_ratio = np.mean(gray < 100)
            return dark_ratio < threshold

        self.logger.log("Timer state: IDLE")
        prev_frame_time = time.time()
        while self.timer_running:
            frame = self.vcap.read()
            if frame is None:
                time.sleep(0.02)
                continue
            left_blocked = is_blocked(frame, left_zone)
            right_blocked = is_blocked(frame, right_zone)

            # state transitions
            if state == "IDLE":
                if left_blocked and right_blocked:
                    state = "READY"
                    self.logger.log("Timer READY - remove hands to start.")
                    countdown_start = time.time()
            elif state == "READY":
                # visual countdown of 3s before RUNNING only if hands removed
                if not (left_blocked or right_blocked):
                    # start running
                    state = "RUNNING"
                    start_time = time.time()
                    self.logger.log("Timer RUNNING...")
            elif state == "RUNNING":
                elapsed = time.time() - start_time
                if left_blocked and right_blocked:
                    stopped_time = elapsed
                    state = "STOPPED"
                    self.logger.log(f"Timer STOPPED at {stopped_time:.2f}s")
                    # update best
                    bt = core.load_best_time(BEST_TIME_PATH)
                    if bt is None or stopped_time < bt:
                        core.save_best_time(stopped_time, BEST_TIME_PATH)
                        self.best_time = stopped_time
                        self.logger.log(f"New best time: {stopped_time:.2f}s")
                        self.best_label.config(text=f"Best: {self.best_time:.2f}s")
                    else:
                        self.logger.log(f"Best remains: {bt:.2f}s")
            elif state == "STOPPED":
                # wait until user stops the timer or reset
                self.timer_running = False
                break

            # sleep small time
            time.sleep(0.02)

        # Ensure UI buttons reset
        self.timer_running = False
        self.btn_start_timer.config(state=tk.NORMAL)
        self.btn_stop_timer.config(state=tk.DISABLED)
        self.logger.log("Timer loop ended.")

    # ---------------- Misc ----------------
    def check_solved(self):
        if not hasattr(self, 'last_captured') or self.last_captured is None:
            messagebox.showinfo("Check Solved", "No captured cube state available. Capture the cube first.")
            return
        solved = core.is_solved_state(self.last_captured)
        core.print_cube_state(self.last_captured)
        if solved:
            messagebox.showinfo("Check Solved", "Cube appears SOLVED!")
        else:
            messagebox.showinfo("Check Solved", "Cube is NOT solved.")

    def reset_best_time(self):
        if os.path.exists(BEST_TIME_PATH):
            try:
                os.remove(BEST_TIME_PATH)
                self.best_time = None
                self.best_label.config(text="Best: N/A")
                self.logger.log("Best time reset.")
            except Exception as e:
                self.logger.log("Failed to reset best time: " + str(e))
        else:
            self.logger.log("No best time file present.")

    def on_exit(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            try:
                self.running_preview = False
                self.vcap.stop()
            except Exception:
                pass
            self.destroy()


if __name__ == "__main__":
    app = RubiksGUI()
    app.mainloop()
