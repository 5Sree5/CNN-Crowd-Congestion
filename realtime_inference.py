# realtime_inference.py
import cv2
import numpy as np
import tensorflow as tf
import joblib
from collections import deque
import time
import random

# ============================================================================
# MODIFY HERE: Set your paths
# ============================================================================
MODEL_DIR = r"C:\Users\srees\LSTM\congestion_prediction\models"
VIDEO_SOURCE = 0  # 0 for webcam, or r"C:\path\to\test_video.mp4" for file
# ============================================================================

print("="*60)
print("REALTIME CROWD MONITORING")
print("="*60)

# -------------------------------------------------------------------
# Load model and parameters
# -------------------------------------------------------------------
print("\n[1/3] Loading model...")
model = tf.keras.models.load_model(f"{MODEL_DIR}/lstm_congestion.h5")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
params = joblib.load(f"{MODEL_DIR}/inference_params.pkl")

LOOKBACK_FRAMES = params['lookback_frames']
FEATURE_COLS = params['feature_cols']

print(f"      Model loaded successfully")
print(f"      Lookback frames: {LOOKBACK_FRAMES}")
print(f"      Features: {FEATURE_COLS}")

# Congestion alert settings
ALERT_THRESHOLD = 0.6  # Adjusted based on validation results
PREDICT_AHEAD_SEC = 60

# -------------------------------------------------------------------
# Simulated YOLO feature extraction (REPLACE THIS WITH YOUR YOLO CODE)
# -------------------------------------------------------------------
print("\n[2/3] Setting up feature extraction (SIMULATION MODE)")

feature_buffer = deque(maxlen=LOOKBACK_FRAMES)
prev_density = 0.5
prev_speed = 1.0

def get_yolo_features_simulated():
    """Simulates YOLO output: returns [speed_mean, density, speed_delta, density_delta]"""
    global prev_density, prev_speed
    
    # Simulate crowd increasing over 60 seconds then resetting
    time_factor = (time.time() % 60) / 60.0
    
    # Base values with increasing trend
    density = 0.3 + 2.0 * time_factor + random.uniform(-0.1, 0.1)
    speed = max(0.2, 1.2 - 0.8 * time_factor + random.uniform(-0.05, 0.05))
    
    speed_delta = speed - prev_speed
    density_delta = density - prev_density
    
    prev_speed = speed
    prev_density = density
    
    return np.array([speed, density, speed_delta, density_delta], dtype=np.float32)

# -------------------------------------------------------------------
# THIS IS WHERE YOU'LL PUT YOUR REAL YOLO CODE LATER:
# -------------------------------------------------------------------
def get_yolo_features_real(frame):
    """
    REPLACE THIS FUNCTION with your actual YOLO + tracking code.
    
    Must return a numpy array with EXACTLY these 4 features in this order:
    [speed_mean, density, speed_delta, density_delta]
    
    Example implementation:
    """
    # 1. Run YOLO on frame
    # results = yolo_model(frame)
    
    # 2. Count people in ROI
    # count = len([d for d in results if d in roi])
    
    # 3. Calculate density (use any area - even frame pixels)
    # frame_area = frame.shape[0] * frame.shape[1]
    # density = count / frame_area
    
    # 4. Track centroids and compute speed
    # speed_mean = compute_average_speed(tracked_objects)
    
    # 5. Calculate deltas
    # speed_delta = speed_mean - prev_speed
    # density_delta = density - prev_density
    
    # return np.array([speed_mean, density, speed_delta, density_delta])
    
    # For now, just use simulation
    return get_yolo_features_simulated()

# -------------------------------------------------------------------
# Main monitoring loop
# -------------------------------------------------------------------
print("\n[3/3] Starting monitoring...")
print("      Press 'q' to quit")
print("")
print("      ⚠️  SIMULATION MODE - Replace get_yolo_features_real() with actual YOLO code")
print("="*60)

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"❌ ERROR: Could not open video source {VIDEO_SOURCE}")
    exit(1)

alert_active = False
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended or camera disconnected.")
        break
    
    # ==============================================================
    # GET FEATURES - Change this line when using real YOLO:
    features = get_yolo_features_real(frame)
    # ==============================================================
    
    feature_buffer.append(features)
    frame_count += 1
    
    if len(feature_buffer) == LOOKBACK_FRAMES:
        # Prepare sequence for LSTM
        seq = np.array(feature_buffer).reshape(1, LOOKBACK_FRAMES, len(FEATURE_COLS))
        seq_flat = seq.reshape(-1, len(FEATURE_COLS))
        seq_norm = scaler.transform(seq_flat).reshape(1, LOOKBACK_FRAMES, len(FEATURE_COLS))
        
        # Predict
        prob = model.predict(seq_norm, verbose=0)[0, 0]
        
        # Alert logic
        if prob >= ALERT_THRESHOLD and not alert_active:
            print(f"🚨 ALERT! Congestion predicted in {PREDICT_AHEAD_SEC}s (prob={prob:.3f})")
            alert_active = True
        elif prob < ALERT_THRESHOLD and alert_active:
            print(f"✅ Alert cleared (prob={prob:.3f})")
            alert_active = False
        
        # Display prediction
        color = (0, 0, 255) if prob >= ALERT_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"Congestion Risk: {prob:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Show threshold line
        bar_x = int(frame.shape[1] * 0.7)
        cv2.rectangle(frame, (bar_x, 10), (bar_x + 100, 25), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, 10), (bar_x + int(prob * 100), 25), color, -1)
        cv2.putText(frame, f"{ALERT_THRESHOLD:.0%}", (bar_x + 105, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Display current features
    cv2.putText(frame, f"Density: {features[1]:.3f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Speed: {features[0]:.3f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Buffer: {len(feature_buffer)}/{LOOKBACK_FRAMES}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show frame
    cv2.imshow('Crowd Congestion Monitor', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Monitoring stopped.")
print("="*60)