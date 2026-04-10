# train_lstm.py
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# ============================================================================
# MODIFY HERE: Set your paths
# ============================================================================
# Where the processed data is saved (from data_preprocessing.py)
DATA_DIR = r"C:\Users\srees\LSTM\congestion_prediction\data\processed"  # <-- VERIFY THIS PATH

# Where to save the trained model and scaler
MODEL_DIR = r"C:\Users\srees\LSTM\congestion_prediction\models"         # <-- VERIFY THIS PATH
# ============================================================================

print("="*60)
print("LSTM TRAINING")
print("="*60)
print(f"Data Dir: {DATA_DIR}")
print(f"Model Dir: {MODEL_DIR}")

# Check if data directory exists
if not os.path.exists(DATA_DIR):
    print(f"\n❌ ERROR: Data directory not found at:")
    print(f"   {DATA_DIR}")
    print("\n   Did you run data_preprocessing.py first?")
    exit(1)

# CPU optimization
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

# -------------------------------------------------------------------
# Load preprocessed data (FIXED PATHS)
# -------------------------------------------------------------------
print("\n[1/5] Loading preprocessed data...")

# Check if files exist
required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "preprocessing_params.pkl"]
for file in required_files:
    file_path = os.path.join(DATA_DIR, file)
    if not os.path.exists(file_path):
        print(f"\n❌ ERROR: Missing file: {file_path}")
        print("   Make sure you ran data_preprocessing.py successfully.")
        exit(1)

# Load the data (FROM DATA_DIR, NOT MODEL_DIR)
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

params = joblib.load(os.path.join(DATA_DIR, "preprocessing_params.pkl"))
lookback_frames = params['lookback_frames']
feature_cols = params['feature_cols']

print(f"      X_train shape: {X_train.shape}")
print(f"      y_train shape: {y_train.shape}")
print(f"      X_val shape: {X_val.shape}")
print(f"      Features: {feature_cols}")

# -------------------------------------------------------------------
# Normalize features
# -------------------------------------------------------------------
print("\n[2/5] Normalizing features...")
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, len(feature_cols))
scaler.fit(X_train_flat)

X_train_norm = scaler.transform(X_train_flat).reshape(X_train.shape)
X_val_norm = scaler.transform(X_val.reshape(-1, len(feature_cols))).reshape(X_val.shape)

print(f"      Mean: {scaler.mean_}")
print(f"      Std: {scaler.scale_}")

# -------------------------------------------------------------------
# Build LSTM model (optimized for CPU)
# -------------------------------------------------------------------
print("\n[3/5] Building LSTM model...")
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True),
        input_shape=(lookback_frames, len(feature_cols))
    ),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

# Class weights for imbalanced data
class_weight = None
if np.mean(y_train) < 0.3:
    class_weight = {0: 1.0, 1: 2.0}
    print(f"\n      Using class weights (imbalance detected): {class_weight}")

# -------------------------------------------------------------------
# Train
# -------------------------------------------------------------------
print("\n[4/5] Training model (this may take 30-60 mins on CPU)...")
print("      Training will stop early if validation loss doesn't improve.")
print("")

history = model.fit(
    X_train_norm, y_train,
    validation_data=(X_val_norm, y_val),
    epochs=25,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ],
    verbose=1
)

# -------------------------------------------------------------------
# Save artifacts
# -------------------------------------------------------------------
print("\n[5/5] Saving model and scaler...")
os.makedirs(MODEL_DIR, exist_ok=True)

model.save(os.path.join(MODEL_DIR, "lstm_congestion.h5"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump({
    'lookback_frames': lookback_frames,
    'feature_cols': feature_cols
}, os.path.join(MODEL_DIR, "inference_params.pkl"))

# Print final metrics
final_epoch = len(history.history['loss'])
print(f"\n✅ TRAINING COMPLETE!")
print(f"   Epochs completed: {final_epoch}")
print(f"   Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"   Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"   Final validation AUC: {history.history['val_auc'][-1]:.4f}")
print(f"\n   Model saved to: {MODEL_DIR}")
print("="*60)