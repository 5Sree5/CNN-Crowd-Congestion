# realtime_inference.py
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import tensorflow as tf
import joblib
import os

# ============================================================================
# PATHS - CHANGE THESE 4 LINES ONLY
# ============================================================================
HEAD_MODEL_PATH = r"C:\Users\srees\LSTM\best.pt"           
BODY_MODEL_PATH = r"C:\Users\srees\LSTM\yolov8n.pt"       
LSTM_MODEL_DIR = r"C:\Users\srees\LSTM\congestion_prediction\models"
VIDEO_SOURCE = r"C:\Users\srees\LSTM\c1.mp4"

# ============================================================================
# SETTINGS
# ============================================================================
ALERT_THRESHOLD = 0.76
ROI_AREA = 100.0

# ============================================================================
# LOAD LSTM MODEL
# ============================================================================
print("Loading LSTM model...")
lstm_model = tf.keras.models.load_model(os.path.join(LSTM_MODEL_DIR, "lstm_congestion.h5"))
scaler = joblib.load(os.path.join(LSTM_MODEL_DIR, "scaler.pkl"))
params = joblib.load(os.path.join(LSTM_MODEL_DIR, "inference_params.pkl"))

LOOKBACK_FRAMES = params['lookback_frames']
FEATURE_COLS = params['feature_cols']

# ============================================================================
# LOAD YOLO MODELS
# ============================================================================
print("Loading YOLO models...")
head_model = YOLO(HEAD_MODEL_PATH)
body_model = YOLO(BODY_MODEL_PATH)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
track_history = defaultdict(list)
feature_buffer = deque(maxlen=LOOKBACK_FRAMES)
prev_density = 0.0
prev_speed = 0.0
alert_active = False
FPS = 30

# ============================================================================
# IoU FUNCTION (from your YOLO code)
# ============================================================================
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union = box1_area + box2_area - inter_area

    return inter_area / union if union > 0 else 0

# ============================================================================
# MAIN LOOP
# ============================================================================
print("Starting video... Press 'q' to quit")
print("="*60)

cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # -------- HEAD DETECTION (from your code) --------
    head_results = head_model(frame, conf=0.3, verbose=False)
    head_boxes = head_results[0].boxes.xyxy.cpu().numpy() if head_results[0].boxes is not None else []

    # -------- BODY TRACKING (from your code) --------
    body_results = body_model.track(frame, persist=True, conf=0.3, tracker="bytetrack.yaml", verbose=False)

    body_boxes = []
    speeds = []

    if body_results[0].boxes is not None and body_results[0].boxes.id is not None:
        body_boxes = body_results[0].boxes.xyxy.cpu().numpy()
        ids = body_results[0].boxes.id.cpu().numpy()
        xywh_boxes = body_results[0].boxes.xywh.cpu().numpy()

        for (box, track_id) in zip(xywh_boxes, ids):
            x, y, w_box, h_box = box
            track_history[track_id].append((x, y))
            if len(track_history[track_id]) > 10:
                track_history[track_id].pop(0)
            if len(track_history[track_id]) >= 2:
                (x1, y1) = track_history[track_id][-2]
                (x2, y2) = track_history[track_id][-1]
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                speed = dist * FPS
                speeds.append(speed)

    # -------- COUNT PEOPLE (from your code) --------
    people_count = len(body_boxes)
    for hbox in head_boxes:
        overlap = False
        for bbox in body_boxes:
            if compute_iou(hbox, bbox) > 0.3:
                overlap = True
                break
        if not overlap:
            people_count += 1

    # -------- CALCULATE FEATURES --------
    avg_speed = np.mean(speeds) if speeds else 0.0
    density = people_count / ROI_AREA
    
    speed_delta = avg_speed - prev_speed
    density_delta = density - prev_density
    prev_speed = avg_speed
    prev_density = density
    
    # -------- ADD TO LSTM BUFFER --------
    features = np.array([avg_speed, density, speed_delta, density_delta], dtype=np.float32)
    feature_buffer.append(features)
    
    # -------- LSTM PREDICTION --------
    lstm_prob = 0.0
    if len(feature_buffer) == LOOKBACK_FRAMES:
        seq = np.array(feature_buffer).reshape(1, LOOKBACK_FRAMES, len(FEATURE_COLS))
        seq_norm = scaler.transform(seq.reshape(-1, len(FEATURE_COLS))).reshape(1, LOOKBACK_FRAMES, -1)
        lstm_prob = lstm_model.predict(seq_norm, verbose=0)[0, 0]
        
        # Alert
        if lstm_prob >= ALERT_THRESHOLD and not alert_active:
            print(f"🚨 ALERT! Congestion in 60s (prob={lstm_prob:.2f})")
            alert_active = True
        elif lstm_prob < ALERT_THRESHOLD and alert_active:
            print(f"✅ Alert cleared")
            alert_active = False
    
    # -------- DRAW ON FRAME --------
    for box in body_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for box in head_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.putText(frame, f"People: {people_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Density: {density:.2f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, f"Speed: {avg_speed:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    
    if len(feature_buffer) == LOOKBACK_FRAMES:
        color = (0, 0, 255) if lstm_prob >= ALERT_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"60s Risk: {lstm_prob:.0%}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(frame, f"Buffer: {len(feature_buffer)}/{LOOKBACK_FRAMES}", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
    
    cv2.imshow('Crowd Congestion Predictor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")