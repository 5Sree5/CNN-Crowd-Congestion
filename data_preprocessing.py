# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

# -------------------------------------------------------------------
# CONFIGURATION – ADJUST THESE
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "data", "ucsd_crowd.csv")
FALLBACK_CSV_PATH = os.path.join(BASE_DIR, "..", "ucsd_features.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
LOOKBACK_SEC = 10          # seconds of history
PREDICT_AHEAD_SEC = 5      # seconds ahead to forecast

# Features from your CSV
FEATURE_COLS = ['speed_mean', 'density_map_avg', 'speed_delta', 'density_delta']
TARGET_COL = 'congestion_now'

# -------------------------------------------------------------------
# Load and prepare data
# -------------------------------------------------------------------
print("Loading CSV...")
if os.path.exists(CSV_PATH):
    csv_file = CSV_PATH
elif os.path.exists(FALLBACK_CSV_PATH):
    csv_file = os.path.abspath(FALLBACK_CSV_PATH)
    print(f"Using fallback dataset at {csv_file}")
else:
    raise FileNotFoundError(
        f"Dataset not found. Expected {CSV_PATH} or {FALLBACK_CSV_PATH}"
    )

df = pd.read_csv(csv_file)


# Calculate FPS per video (median frame interval)
print("Calculating FPS per video...")
fps_dict = {}
for vid, group in df.groupby('video'):
    time_diffs = group['time_sec'].diff().dropna()
    if len(time_diffs) > 0:
        fps = 1.0 / time_diffs.median()
    else:
        fps = 2.0  # fallback
    fps_dict[vid] = fps
df['fps'] = df['video'].map(fps_dict)

# -------------------------------------------------------------------
# Create sequences per video
# -------------------------------------------------------------------
def create_sequences(group, lookback_frames, future_steps):
    data = group[FEATURE_COLS].values
    targets = group[TARGET_COL].values
    X, y = [], []
    for i in range(len(data) - lookback_frames - future_steps):
        X.append(data[i : i+lookback_frames])
        y.append(targets[i + lookback_frames + future_steps - 1])
    return np.array(X), np.array(y)

print("Creating sequences...")
X_list, y_list = [], []
video_indices = []  # keep track which video each sequence belongs to (for proper split)

for vid, group in df.groupby('video'):
    fps = fps_dict[vid]
    lookback_frames = int(LOOKBACK_SEC * fps)
    future_steps = int(PREDICT_AHEAD_SEC * fps)
    
    if len(group) < lookback_frames + future_steps:
        print(f"Video {vid} too short, skipping.")
        continue
    
    X_vid, y_vid = create_sequences(group, lookback_frames, future_steps)
    if len(X_vid) == 0:
        continue
    X_list.append(X_vid)
    y_list.append(y_vid)
    video_indices.extend([vid] * len(X_vid))

if len(X_list) == 0:
    raise ValueError(
        "No sequences were created. Try reducing PREDICT_AHEAD_SEC or using longer videos."
    )

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
video_indices = np.array(video_indices)

print(f"Total sequences: {X.shape[0]}")
print(f"Sequence shape: ({X.shape[1]} frames, {X.shape[2]} features)")

# -------------------------------------------------------------------
# Train / validation split (split by video to avoid leakage)
# -------------------------------------------------------------------
unique_videos = np.unique(video_indices)
train_vids, val_vids = train_test_split(unique_videos, test_size=0.2, random_state=42)

train_mask = np.isin(video_indices, train_vids)
val_mask = np.isin(video_indices, val_vids)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# -------------------------------------------------------------------
# Save everything
# -------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "feature_cols.npy"), FEATURE_COLS)

# Save the exact lookback_frames used (since FPS may vary, we need the value from the first video)
joblib.dump({
    'lookback_frames': X_train.shape[1],
    'fps_dict': fps_dict,
    'feature_cols': FEATURE_COLS
}, os.path.join(OUTPUT_DIR, "preprocessing_params.pkl"))

print(f"Preprocessed data saved to {OUTPUT_DIR}")