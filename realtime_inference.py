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
VIDEO_SOURCE = r"C:\Users\srees\LSTM\c1.mp4"  # 0 for webcam, or r"C:\path\to\video.mp4"

# ============================================================================
# SETTINGS
# ============================================================================
ALERT_THRESHOLD = 0.73   # for console alert
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
critical_risk_start_time = None
critical_popup_shown = False

# New for high‑risk popup
high_risk_start_time = None
popup_shown = False

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
        
        # Console alert (80% threshold)
        if lstm_prob >= ALERT_THRESHOLD and not alert_active:
            print(f"🚨 ALERT! Congestion in 60s (prob={lstm_prob:.2f})")
            alert_active = True
        elif lstm_prob < ALERT_THRESHOLD and alert_active:
            print(f"✅ Alert cleared")
            alert_active = False
        
                # ================================================================
        # POPUP LOGIC - TWO THRESHOLDS
        # ================================================================
        HIGH_RISK_THRESHOLD = 0.73
        HIGH_RISK_DURATION = 2.5
        
        CRITICAL_THRESHOLD = 0.90
        CRITICAL_DURATION = 1.5   # Shorter time for immediate action
        
        current_time = time.time()
        
        # ---- CRITICAL (>95%) Check - Takes Priority ----
        if lstm_prob >= CRITICAL_THRESHOLD:
            if critical_risk_start_time is None:
                critical_risk_start_time = current_time
                print(f"[DEBUG] CRITICAL risk started at {critical_risk_start_time:.1f}")
            else:
                elapsed = current_time - critical_risk_start_time
                print(f"[DEBUG] CRITICAL risk sustained: {elapsed:.2f}s / {CRITICAL_DURATION}s")
                if elapsed >= CRITICAL_DURATION and not critical_popup_shown:
                    print("[DEBUG] Triggering CRITICAL popup!")
                    critical_popup_shown = True
                    
                    popup_frame = frame.copy()
                    cv2.putText(popup_frame, "!!! HIGHLY CONGESTED !!!", 
                                (50, popup_frame.shape[0]//2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
                    cv2.putText(popup_frame, "IMMEDIATE ACTION NEEDED", 
                                (50, popup_frame.shape[0]//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                    cv2.putText(popup_frame, "Congestion still likely in 60 seconds!", 
                                (50, popup_frame.shape[0]//2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    cv2.putText(popup_frame, "Press any key to acknowledge", 
                                (50, popup_frame.shape[0]//2 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                    
                    cv2.imshow('Crowd Congestion Predictor', popup_frame)
                    print("[DEBUG] Critical popup displayed. Waiting for key press...")
                    cv2.waitKey(0)
                    print("[DEBUG] Key pressed. Resuming...")
                    critical_risk_start_time = None
                    critical_popup_shown = False
                    high_risk_start_time = None   # Reset the lower threshold timer too
                    popup_shown = False
        else:
            if critical_risk_start_time is not None:
                print("[DEBUG] Risk dropped below critical threshold. Resetting critical timer.")
            critical_risk_start_time = None
            critical_popup_shown = False
        
        # ---- HIGH RISK (>73%) Check - Only if not already in critical state ----
        if lstm_prob >= HIGH_RISK_THRESHOLD and lstm_prob < CRITICAL_THRESHOLD:
            if high_risk_start_time is None:
                high_risk_start_time = current_time
                print(f"[DEBUG] High risk started at {high_risk_start_time:.1f}")
            else:
                elapsed = current_time - high_risk_start_time
                print(f"[DEBUG] High risk sustained: {elapsed:.2f}s / {HIGH_RISK_DURATION}s")
                if elapsed >= HIGH_RISK_DURATION and not popup_shown:
                    print("[DEBUG] Triggering HIGH RISK popup!")
                    popup_shown = True
                    
                    popup_frame = frame.copy()
                    cv2.putText(popup_frame, "WARNING: Congestion likely in 60s!", 
                                (50, popup_frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,165,255), 3)
                    cv2.putText(popup_frame, "If this pattern continues...", 
                                (50, popup_frame.shape[0]//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
                    cv2.putText(popup_frame, "Press any key to resume monitoring", 
                                (50, popup_frame.shape[0]//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    
                    cv2.imshow('Crowd Congestion Predictor', popup_frame)
                    print("[DEBUG] High risk popup displayed. Waiting for key press...")
                    cv2.waitKey(0)
                    print("[DEBUG] Key pressed. Resuming...")
                    high_risk_start_time = None
                    popup_shown = False
        else:
            # Reset high risk timer if below 73% (unless in critical, which is handled separately)
            if lstm_prob < HIGH_RISK_THRESHOLD:
                if high_risk_start_time is not None:
                    print("[DEBUG] Risk dropped below high threshold. Resetting high timer.")
                high_risk_start_time = None
                popup_shown = False

        # -------- DRAW ON FRAME --------
    # Draw bounding boxes
    for box in body_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for box in head_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Current statistics
    cv2.putText(frame, f"People: {people_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Density: {density:.2f} p/m2", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Speed: {avg_speed:.2f} px/frame", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # LSTM Risk Prediction
    if len(feature_buffer) == LOOKBACK_FRAMES:
        # Choose color based on risk level
        if lstm_prob >= 0.95:
            color = (0, 0, 255)      # Red for critical
        elif lstm_prob >= ALERT_THRESHOLD:
            color = (0, 165, 255)    # Orange for high
        else:
            color = (0, 255, 0)      # Green for normal
        cv2.putText(frame, f"60s Risk: {lstm_prob:.0%}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(frame, f"Buffering: {len(feature_buffer)}/{LOOKBACK_FRAMES}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # Timer displays (only when active and below threshold duration)
    CRITICAL_DURATION = 1.5
    HIGH_RISK_DURATION = 2.5
    
    if critical_risk_start_time is not None:
        elapsed = time.time() - critical_risk_start_time
        if elapsed < CRITICAL_DURATION:
            timer_text = f"CRITICAL: {elapsed:.1f}s / {CRITICAL_DURATION}s"
            cv2.putText(frame, timer_text, (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    if high_risk_start_time is not None:
        elapsed = time.time() - high_risk_start_time
        if elapsed < HIGH_RISK_DURATION:
            timer_text = f"High risk: {elapsed:.1f}s / {HIGH_RISK_DURATION}s"
            cv2.putText(frame, timer_text, (20, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Show the frame
    cv2.imshow('Crowd Congestion Predictor', frame)
    
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # Show high risk timer if active
    if high_risk_start_time is not None:
        elapsed = time.time() - high_risk_start_time
        if elapsed < HIGH_RISK_DURATION:
            timer_text = f"High risk: {elapsed:.1f}s / {HIGH_RISK_DURATION}s"
            cv2.putText(frame, timer_text, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
    # Show high‑risk timer if active
    if high_risk_start_time is not None:
        elapsed = time.time() - high_risk_start_time
        if elapsed < 2.0:
            timer_text = f"High risk sustained: {elapsed:.1f}s / 2.0s"
            cv2.putText(frame, timer_text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
    
    cv2.imshow('Crowd Congestion Predictor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")