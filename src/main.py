import cv2
import os
import json
import pickle
import time
from datetime import datetime

from ultralytics import YOLO
import telebot
import numpy as np
from plyer import notification
import winsound  # Windows-only beep

# -------------------------
# PATH SETUP
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CONFIG_PATH = os.path.join(ROOT_DIR, "telegram_config.json")
LOG_DIR = os.path.join(ROOT_DIR, "incident_logs")
LOG_CSV_PATH = os.path.join(LOG_DIR, "events_log.csv")
VIDEO_DIR = os.path.join(LOG_DIR, "videos")

FACES_DIR = os.path.join(ROOT_DIR, "faces")
FACE_MODEL_PATH = os.path.join(ROOT_DIR, "face_model.yml")
FACE_LABELS_PATH = os.path.join(ROOT_DIR, "face_labels.pkl")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# -------------------------
# LOAD TELEGRAM CONFIG
# -------------------------

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

BOT_TOKEN = cfg["bot_token"]
CHAT_ID = cfg["chat_id"]
bot = telebot.TeleBot(BOT_TOKEN)

# -------------------------
# LOAD MODEL
# -------------------------

model = YOLO("yolov8n.pt")
print("✅ YOLOv8n model loaded.")

PERSON_LABEL = "person"
DANGEROUS_LABELS = {"knife", "scissors", "bottle", "backpack"}

# -------------------------
# SEVERITY LEVELS
# -------------------------

EVENT_SEVERITY = {
    "weapon": "HIGH",
    "fire": "HIGH",
    "fight": "HIGH",
    "fall": "MEDIUM",
    "child": "MEDIUM",
    "crowd": "MEDIUM",
    "zone": "LOW",
    "face_unknown": "MEDIUM",
    "behavior": "MEDIUM",  # NEW risk alert
}

ALERT_COOLDOWN = {
    "weapon": 8,
    "crowd": 15,
    "fall": 10,
    "child": 15,
    "fire": 12,
    "fight": 10,
    "zone": 15,
    "face_unknown": 20,
    "behavior": 12,  # NEW
}

last_alert_time = {key: 0 for key in ALERT_COOLDOWN}

# -------------------------
# TRACKING + HEATMAP + RISK MODULES
# -------------------------

prev_tracks = {}
next_track_id = 1

# Heatmap accumulator
heatmap = None

# Risk system memory
speed_prev_pos = {}   # pid -> (cx, cy)
loiter_map = {}       # pid -> {"pos": (cx,cy), "frames": count}

# Video recording
recording = False
record_end_time = 0
video_writer = None

# -------------------------
# FACE RECOGNITION (LBPH)
# -------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_labels = {}

if os.path.exists(FACE_MODEL_PATH) and os.path.exists(FACE_LABELS_PATH):
    recognizer.read(FACE_MODEL_PATH)
    with open(FACE_LABELS_PATH, "rb") as f:
        lbl = pickle.load(f)
        face_labels = {v: k for k, v in lbl.items()}
    print("😊 Face recognition model loaded.")
else:
    print("⚠️ Face model not found. Run: python src\\face_train.py")

# -------------------------
# SIMPLE HELPERS
# -------------------------

def append_event_log(event_type, description, image_path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    severity = EVENT_SEVERITY.get(event_type, "UNKNOWN")
    file_exists = os.path.exists(LOG_CSV_PATH)

    with open(LOG_CSV_PATH, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("timestamp,event_type,severity,description,image_path\n")
        safe_desc = description.replace(",", ";")
        f.write(f"{ts},{event_type},{severity},{safe_desc},{image_path}\n")

    print(f"[CSV] Logged event → {event_type} ({severity})")


def save_screenshot(frame, event_type):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{event_type}_{ts}.jpg"
    path = os.path.join(LOG_DIR, filename)
    cv2.imwrite(path, frame)
    return path


def play_alarm():
    try:
        winsound.Beep(1200, 400)
        winsound.Beep(1200, 400)
    except:
        pass


def show_popup(title, message):
    try:
        notification.notify(title=title, message=message, timeout=4)
    except:
        pass


def send_telegram_alert(event_type, description, image_path=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    severity = EVENT_SEVERITY.get(event_type, "UNKNOWN")

    text = (
        f"⚠️ AI Safety Alert [{severity}] ⚠️\n\n"
        f"Event: {event_type}\n"
        f"Time: {ts}\n\n"
        f"Details: {description}"
    )

    try:
        if image_path:
            with open(image_path, "rb") as img:
                bot.send_photo(CHAT_ID, img, caption=text)
        else:
            bot.send_message(CHAT_ID, text)
    except:
        print("[ERROR] Telegram failed.")


def start_video_recording(event_key, frame):
    global recording, record_end_time, video_writer

    severity = EVENT_SEVERITY.get(event_key, "LOW")
    if severity not in ["HIGH", "MEDIUM"]:
        return

    now = time.time()
    record_end_time = now + 8  # keep recording for 8 sec

    if not recording:
        h, w, _ = frame.shape
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{event_key}_{ts}.avi"
        save_path = os.path.join(VIDEO_DIR, fname)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(save_path, fourcc, 20.0, (w, h))
        recording = True


def write_video_frame(frame):
    global recording, video_writer

    if not recording:
        return

    now = time.time()
    if now > record_end_time:
        video_writer.release()
        recording = False
        return

    video_writer.write(frame)
# -------------------------
# EVENT TRIGGER
# -------------------------

def trigger_event(event_key, description, frame):
    now = time.time()
    if now - last_alert_time[event_key] < ALERT_COOLDOWN[event_key]:
        return  # cooldown active

    last_alert_time[event_key] = now
    severity = EVENT_SEVERITY.get(event_key, "UNKNOWN")

    img_path = save_screenshot(frame, event_key)
    append_event_log(event_key, description, img_path)
    send_telegram_alert(event_key, description, img_path)
    show_popup(f"AI Alert ({severity})", description)
    play_alarm()
    start_video_recording(event_key, frame)


# -------------------------
# FIRE DETECTION
# -------------------------

def detect_fire_like_region(frame, min_ratio=0.02):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0,120,150]), np.array([25,255,255]))
    mask2 = cv2.inRange(hsv, np.array([160,120,150]), np.array([179,255,255]))

    mask = cv2.bitwise_or(mask1, mask2)

    fire_pixels = cv2.countNonZero(mask)
    total_pixels = frame.shape[0] * frame.shape[1]

    return fire_pixels / total_pixels > min_ratio if total_pixels else False


# -------------------------
# PERSON ID TRACKING
# -------------------------

def assign_ids(persons):
    global prev_tracks, next_track_id

    new_tracks = {}
    max_dist = 80

    for p in persons:
        cx, cy = p["cx"], p["cy"]
        assigned = None
        best_d = None

        for tid, (tx, ty) in prev_tracks.items():
            d = ((cx - tx)**2 + (cy - ty)**2)**0.5
            if d < max_dist and (best_d is None or d < best_d):
                best_d = d
                assigned = tid

        if assigned is None:
            assigned = next_track_id
            next_track_id += 1

        new_tracks[assigned] = (cx, cy)
        p["id"] = assigned

    prev_tracks = new_tracks


# -------------------------
# HEATMAP
# -------------------------

def update_heatmap(frame, persons):
    global heatmap

    h, w, _ = frame.shape
    if heatmap is None:
        heatmap = np.zeros((h//8, w//8), dtype=np.float32)

    for p in persons:
        x = min(max(p["cx"] // 8, 0), heatmap.shape[1]-1)
        y = min(max(p["cy"] // 8, 0), heatmap.shape[0]-1)
        heatmap[y, x] += 1


def overlay_heatmap(frame, alpha=0.30):
    if heatmap is None:
        return frame

    hm = heatmap.copy()
    hm = hm / (hm.max() + 1e-5)
    hm = cv2.resize(hm, (frame.shape[1], frame.shape[0]))
    hm_col = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    return cv2.addWeighted(frame, 1 - alpha, hm_col, alpha, 0)


# -------------------------
# FACE RECOGNITION
# -------------------------

def detect_and_recognize_faces(frame):
    if not face_labels:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detected = []

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi)

        name = face_labels.get(id_, "Unknown") if conf < 70 else "Unknown"

        detected.append({
            "name": name,
            "bbox": (x, y, x+w, y+h)
        })

    return detected


# -------------------------
# BEHAVIOR ANALYSIS (Option A)
# -------------------------

def compute_behavior_risk(persons):
    global speed_prev_pos, loiter_map

    risk_level = "LOW"
    risky_id = None

    for p in persons:
        pid = p["id"]
        cx, cy = p["cx"], p["cy"]

        # -------------------------
        # SPEED CHECK
        # -------------------------
        if pid in speed_prev_pos:
            px, py = speed_prev_pos[pid]
            speed = ((cx - px)**2 + (cy - py)**2)**0.5

            if speed > 35:     # Running fast
                risk_level = "HIGH"
                risky_id = pid

            elif speed > 20:   # Fast walking
                if risk_level != "HIGH":
                    risk_level = "MEDIUM"
                    risky_id = pid

        speed_prev_pos[pid] = (cx, cy)

        # -------------------------
        # LOITERING CHECK
        # -------------------------
        if pid not in loiter_map:
            loiter_map[pid] = {"frames": 0, "pos": (cx, cy)}

        old_cx, old_cy = loiter_map[pid]["pos"]
        dist = ((cx - old_cx)**2 + (cy - old_cy)**2)**0.5

        if dist < 12:  # very small movement
            loiter_map[pid]["frames"] += 1
        else:
            loiter_map[pid]["frames"] = 0
            loiter_map[pid]["pos"] = (cx, cy)

        if loiter_map[pid]["frames"] > 90:  # 3 seconds
            if risk_level != "HIGH":
                risk_level = "MEDIUM"
                risky_id = pid

    return risk_level, risky_id


# -------------------------
# MAIN LOOP
# -------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    print("🚀 AI Monitor Running — FULL VERSION ACTIVE")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False)

        persons = []
        dangerous = []

        # -------------------------
        # YOLO PARSE
        # -------------------------
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = r.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2)//2
                cy = (y1 + y2)//2

                if label == PERSON_LABEL:
                    persons.append({
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "cx": cx, "cy": cy
                    })

                if label.lower() in DANGEROUS_LABELS:
                    dangerous.append(label.lower())

        # -------------------------
        # PERSON ID + HEATMAP
        # -------------------------
        if persons:
            assign_ids(persons)
            update_heatmap(frame, persons)

        # -------------------------
        # DRAW PERSONS (with color by risk)
        # -------------------------
        risk_level, risky_id = compute_behavior_risk(persons)

        for p in persons:
            color = (0, 255, 0)  # green = low
            if risk_level == "MEDIUM" and p["id"] == risky_id:
                color = (0, 255, 255)  # yellow
            if risk_level == "HIGH" and p["id"] == risky_id:
                color = (0, 0, 255)  # red

            cv2.rectangle(frame, (p["x1"], p["y1"]), (p["x2"], p["y2"]), color, 2)
            cv2.putText(frame, f"ID {p['id']}", (p["x1"], p["y1"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # -------------------------
        # BEHAVIOR ALERT
        # -------------------------
        if risk_level == "MEDIUM":
            trigger_event("behavior", "Suspicious movement detected", frame)

        if risk_level == "HIGH":
            trigger_event("behavior", "HIGH-RISK: Aggressive/Rapid behavior", frame)

        # -------------------------
        # OTHER EVENTS: weapon, fire, fall, crowd
        # -------------------------

        if dangerous:
            trigger_event("weapon", f"Dangerous objects: {dangerous}", frame)

        if len(persons) >= 3:
            trigger_event("crowd", f"Crowd detected ({len(persons)})", frame)

        for p in persons:
            h = p["y2"] - p["y1"]
            w = p["x2"] - p["x1"]
            if h/w < 0.8 and p["cy"] > frame.shape[0] * 0.55:
                trigger_event("fall", "Possible fall", frame)
                break

        if detect_fire_like_region(frame):
            trigger_event("fire", "Fire-like region detected", frame)

        # -------------------------
        # FACE RECOGNITION ALERT
        # -------------------------
        faces = detect_and_recognize_faces(frame)

        unknown_found = any(f["name"] == "Unknown" for f in faces)

        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            name = f["name"]
            col = (0, 0, 255) if name == "Unknown" else (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        if unknown_found:
            trigger_event("face_unknown", "Unknown face detected!", frame)

        # -------------------------
        # WRITE VIDEO IF RECORDING
        # -------------------------
        write_video_frame(frame)

        # -------------------------
        # SHOW FINAL OUTPUT WITH HEATMAP
        # -------------------------
        disp = overlay_heatmap(frame)
        cv2.imshow("AI Public Safety Monitor — Advanced", disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
