"""
Drowsy Driver Alert System
Uses OpenCV Haar Cascades for face/eye detection.
Alerts when eyes remain closed, logs events to CSV, captures screenshots.
Compatible with Python 3.8+
"""

import cv2
import csv
import os
import time
import datetime
import sys


# ─────────────────────────── Configuration ───────────────────────────────────

EYE_CLOSED_FRAMES_THRESHOLD = 20   # consecutive frames before alert fires
ALERT_COOLDOWN_SECONDS      = 5    # minimum gap between repeated alerts
SCREENSHOT_DIR              = "drowsy_screenshots"
CSV_LOG_FILE                = "drowsy_log.csv"

FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
EYE_SCALE_FACTOR  = 1.1
EYE_MIN_NEIGHBORS = 8

MOUTH_SCALE_FACTOR  = 1.7
MOUTH_MIN_NEIGHBORS = 11
YAWN_OPEN_FRAMES_THRESHOLD = 15   # consecutive open-mouth frames = yawn
YAWN_COOLDOWN_SECONDS      = 8    # min gap between logged yawn events
COLOR_YAWN = (255, 140, 0)        # orange (BGR)

# Visual colours (BGR)
COLOR_AWAKE   = (0, 220, 80)    # green
COLOR_DROWSY  = (0, 60, 255)    # red
COLOR_WARNING = (0, 180, 255)   # amber
COLOR_TEXT    = (240, 240, 240)
COLOR_SHADOW  = (20,  20,  20)
OVERLAY_ALPHA = 0.35


# ──────────────────────────── Helpers ────────────────────────────────────────

def load_cascade(name: str) -> cv2.CascadeClassifier:
    """Load a Haar cascade by OpenCV data name, abort if missing."""
    path = cv2.data.haarcascades + name
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        sys.exit(f"[ERROR] Could not load cascade: {path}")
    return cascade


def ensure_dirs() -> None:
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def init_csv() -> None:
    if not os.path.exists(CSV_LOG_FILE):
        with open(CSV_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event", "closed_frames", "screenshot"])


def log_event(event: str, closed_frames: int, screenshot_path: str = "") -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, event, closed_frames, screenshot_path])
    print(f"[LOG] {ts}  {event}  frames={closed_frames}  {screenshot_path}")


def save_screenshot(frame) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SCREENSHOT_DIR, f"alert_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def draw_text_with_shadow(img, text, pos, font, scale, color, thickness=2):
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), font, scale, COLOR_SHADOW,
                thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame, status: str, closed_frames: int, fps: float,
             alert_active: bool, yawn_alert: bool, yawn_count: int) -> None:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX

    # ── top status bar ──────────────────────────────────────────────────────
    bar_color = COLOR_DROWSY if alert_active else (30, 30, 30)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), bar_color, -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA if not alert_active else 0.6,
                    frame, 1 - (OVERLAY_ALPHA if not alert_active else 0.6),
                    0, frame)

    label_color = COLOR_DROWSY if alert_active else COLOR_AWAKE
    draw_text_with_shadow(frame, f"Status: {status}", (10, 33), font, 0.75,
                          label_color)

    fps_str = f"FPS: {fps:4.1f}"
    (tw, _), _ = cv2.getTextSize(fps_str, font, 0.55, 1)
    draw_text_with_shadow(frame, fps_str, (w - tw - 10, 33), font, 0.55,
                          COLOR_TEXT)

    # ── yawn counter (top-right, below FPS) ─────────────────────────────────
    yawn_str = f"Yawns: {yawn_count}"
    (yw, _), _ = cv2.getTextSize(yawn_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    draw_text_with_shadow(frame, yawn_str, (w - yw - 10, 75),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_YAWN, 1)

    # ── drowsiness progress bar ─────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 10, h - 30, w - 20, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)
    ratio = min(closed_frames / EYE_CLOSED_FRAMES_THRESHOLD, 1.0)
    fill_color = COLOR_DROWSY if ratio >= 1.0 else (
        COLOR_WARNING if ratio > 0.5 else COLOR_AWAKE)
    filled_w = int(bar_w * ratio)
    if filled_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + filled_w, bar_y + bar_h), fill_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (120, 120, 120), 1)
    draw_text_with_shadow(frame, "Drowsiness", (bar_x + 2, bar_y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT, 1)

    # ── big drowsy alert banner ──────────────────────────────────────────────
    if alert_active:
        msg = "DROWSY DRIVER ALERT!"
        (mw, mh), _ = cv2.getTextSize(msg, font, 1.2, 2)
        mx = (w - mw) // 2
        my = h // 2 + mh // 2
        cv2.rectangle(frame, (mx - 18, my - mh - 14),
                      (mx + mw + 18, my + 10), COLOR_DROWSY, -1)
        draw_text_with_shadow(frame, msg, (mx, my), font, 1.2,
                              (255, 255, 255), 2)

    # ── yawn alert banner (shown below drowsy banner or centred) ────────────
    if yawn_alert:
        ymsg = "YAWN DETECTED"
        (ymw, ymh), _ = cv2.getTextSize(ymsg, font, 1.0, 2)
        ymx = (w - ymw) // 2
        # sit below drowsy banner when both active, else centre-ish
        ymy = (h // 2 + ymh // 2 + 60) if alert_active else (h // 2 + ymh // 2)
        cv2.rectangle(frame, (ymx - 14, ymy - ymh - 10),
                      (ymx + ymw + 14, ymy + 8), COLOR_YAWN, -1)
        draw_text_with_shadow(frame, ymsg, (ymx, ymy), font, 1.0,
                              (255, 255, 255), 2)


# ─────────────────────────── Detection core ──────────────────────────────────

def detect_eyes_open(gray_roi, eye_cascade) -> bool:
    """Return True if at least one eye is detected in the (gray) face ROI."""
    eyes = eye_cascade.detectMultiScale(
        gray_roi,
        scaleFactor=EYE_SCALE_FACTOR,
        minNeighbors=EYE_MIN_NEIGHBORS,
        minSize=(20, 20),
    )
    return len(eyes) > 0


def detect_mouth_open(gray_roi, mouth_cascade):
    """
    Run mouth cascade on the lower-face ROI.
    Returns list of detected mouth rectangles (may be empty).
    Uses a high minNeighbors to suppress false positives.
    """
    mouths = mouth_cascade.detectMultiScale(
        gray_roi,
        scaleFactor=MOUTH_SCALE_FACTOR,
        minNeighbors=MOUTH_MIN_NEIGHBORS,
        minSize=(30, 15),
    )
    return mouths


# ─────────────────────────── Main loop ───────────────────────────────────────

def main() -> None:
    ensure_dirs()
    init_csv()

    face_cascade  = load_cascade("haarcascade_frontalface_default.xml")
    eye_cascade   = load_cascade("haarcascade_eye.xml")
    mouth_cascade = load_cascade("haarcascade_smile.xml")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam. Check device index or permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    closed_frames    = 0
    alert_active     = False
    last_alert_time  = 0.0
    open_mouth_frames = 0
    yawn_alert        = False
    last_yawn_time    = 0.0
    yawn_count        = 0
    prev_tick        = cv2.getTickCount()
    fps              = 0.0

    log_event("SESSION_START", 0)
    print("[INFO] Drowsy Driver Alert System running. Press 'q' to quit, "
          "'s' to manually save screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed — retrying…")
            time.sleep(0.05)
            continue

        # ── FPS ──────────────────────────────────────────────────────────────
        tick  = cv2.getTickCount()
        fps   = cv2.getTickFrequency() / (tick - prev_tick)
        prev_tick = tick

        # ── Detection ────────────────────────────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=(80, 80),
        )

        eyes_open      = False
        face_found     = len(faces) > 0
        mouth_detected = False

        for (fx, fy, fw, fh) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh),
                          COLOR_AWAKE, 2)

            # Examine upper-half of face for eyes (reduces false positives)
            roi_y1, roi_y2 = fy, fy + int(fh * 0.65)
            roi_gray = gray[roi_y1:roi_y2, fx:fx + fw]
            roi_color = frame[roi_y1:roi_y2, fx:fx + fw]

            if detect_eyes_open(roi_gray, eye_cascade):
                eyes_open = True
                # Draw eyes
                eyes_det = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=EYE_SCALE_FACTOR,
                    minNeighbors=EYE_MIN_NEIGHBORS,
                    minSize=(20, 20),
                )
                for (ex, ey, ew, eh) in eyes_det:
                    cx, cy = ex + ew // 2, ey + eh // 2
                    cv2.circle(roi_color, (cx, cy), max(ew, eh) // 2,
                               COLOR_AWAKE, 2)

            # ── Mouth detection (lower 40 % of face) ─────────────────────────
            mouth_y1 = fy + int(fh * 0.60)
            mouth_y2 = fy + fh
            mouth_gray  = gray[mouth_y1:mouth_y2, fx:fx + fw]
            mouth_color = frame[mouth_y1:mouth_y2, fx:fx + fw]
            mouths = detect_mouth_open(mouth_gray, mouth_cascade)
            mouth_detected = len(mouths) > 0
            for (mx2, my2, mw2, mh2) in mouths:
                cv2.rectangle(mouth_color, (mx2, my2),
                              (mx2 + mw2, my2 + mh2), COLOR_YAWN, 2)

        # ── Drowsiness state machine ──────────────────────────────────────────
        if face_found and not eyes_open:
            closed_frames += 1
        else:
            if closed_frames > 0 and alert_active:
                log_event("EYES_REOPENED", closed_frames)
            closed_frames = 0
            alert_active  = False

        now = time.time()
        if closed_frames >= EYE_CLOSED_FRAMES_THRESHOLD:
            if not alert_active or (now - last_alert_time >= ALERT_COOLDOWN_SECONDS):
                alert_active    = True
                last_alert_time = now
                shot_path       = save_screenshot(frame)
                log_event("DROWSY_ALERT", closed_frames, shot_path)

        # ── Yawn state machine ────────────────────────────────────────────────
        if face_found and mouth_detected:
            open_mouth_frames += 1
        else:
            if yawn_alert:
                log_event("YAWN_ENDED", open_mouth_frames)
            open_mouth_frames = 0
            yawn_alert        = False

        if open_mouth_frames >= YAWN_OPEN_FRAMES_THRESHOLD:
            if not yawn_alert or (now - last_yawn_time >= YAWN_COOLDOWN_SECONDS):
                yawn_alert     = True
                last_yawn_time = now
                yawn_count    += 1
                shot_path      = save_screenshot(frame)
                log_event("YAWN_DETECTED", open_mouth_frames, shot_path)

        # ── HUD ───────────────────────────────────────────────────────────────
        if not face_found:
            status = "No face detected"
        elif not eyes_open:
            status = f"Eyes CLOSED  [{closed_frames}/{EYE_CLOSED_FRAMES_THRESHOLD}]"
        elif yawn_alert:
            status = f"Yawning  [{open_mouth_frames}/{YAWN_OPEN_FRAMES_THRESHOLD}]"
        else:
            status = "Eyes open — Awake"

        draw_hud(frame, status, closed_frames, fps, alert_active,
                 yawn_alert, yawn_count)

        cv2.imshow("Drowsy Driver Alert System  |  q=quit  s=screenshot", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            shot_path = save_screenshot(frame)
            log_event("MANUAL_SCREENSHOT", closed_frames, shot_path)
            print(f"[INFO] Screenshot saved: {shot_path}")

    cap.release()
    cv2.destroyAllWindows()
    log_event("SESSION_END", closed_frames)
    print(f"[INFO] Session ended. Yawns recorded: {yawn_count}")
    print(f"[INFO] Log → {CSV_LOG_FILE}  Screenshots → {SCREENSHOT_DIR}/")


if __name__ == "__main__":
    main()
