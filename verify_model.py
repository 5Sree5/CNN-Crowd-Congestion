# verify_model.py
import numpy as np
import tensorflow as tf
import joblib
import os

# ============================================================================
# SET YOUR PATHS
# ============================================================================
MODEL_DIR = r"C:\Users\srees\LSTM\congestion_prediction\models"
DATA_DIR = r"C:\Users\srees\LSTM\congestion_prediction\data\processed"
# ============================================================================

print("="*60)
print("MODEL VERIFICATION")
print("="*60)

# Load model and scaler
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_congestion.h5"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
params = joblib.load(os.path.join(MODEL_DIR, "inference_params.pkl"))

# Load validation data
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

# Normalize
X_val_flat = X_val.reshape(-1, len(params['feature_cols']))
X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)

# Predict
print("\n[1] Running predictions on validation set...")
y_pred_prob = model.predict(X_val_norm, verbose=0).flatten()
y_pred_class = (y_pred_prob >= 0.5).astype(int)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_val, y_pred_class)
precision = precision_score(y_val, y_pred_class)
recall = recall_score(y_val, y_pred_class)
f1 = f1_score(y_val, y_pred_class)
cm = confusion_matrix(y_val, y_pred_class)

print("\n" + "="*60)
print("📊 VALIDATION METRICS")
print("="*60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"Precision: {precision:.4f} (when we predict congestion, we're right {precision*100:.1f}% of the time)")
print(f"Recall:    {recall:.4f} (we catch {recall*100:.1f}% of actual congestion events)")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0,0]:5d}  (No congestion, predicted correctly)")
print(f"  False Positives: {cm[0,1]:5d}  (False alarm)")
print(f"  False Negatives: {cm[1,0]:5d}  (Missed congestion)")
print(f"  True Positives:  {cm[1,1]:5d}  (Congestion detected correctly)")

# -------------------------------------------------------------------
# Test with synthetic samples
# -------------------------------------------------------------------
print("\n" + "="*60)
print("🧪 SYNTHETIC SAMPLE TESTS")
print("="*60)

lookback = params['lookback_frames']
n_features = len(params['feature_cols'])

# Test Case 1: Low density, high speed → Should predict NO congestion
low_crowd = np.tile([1.5, 0.2, 0.0, 0.0], (lookback, 1))  # speed=1.5, density=0.2
low_crowd_norm = scaler.transform(low_crowd).reshape(1, lookback, n_features)
prob1 = model.predict(low_crowd_norm, verbose=0)[0,0]
print(f"\n✅ Low crowd (speed=1.5, density=0.2):")
print(f"   Congestion probability: {prob1:.4f} → {'CONGESTION' if prob1 >= 0.5 else 'NO CONGESTION'}")

# Test Case 2: High density, low speed → Should predict CONGESTION
high_crowd = np.tile([0.3, 2.5, 0.0, 0.0], (lookback, 1))  # speed=0.3, density=2.5
high_crowd_norm = scaler.transform(high_crowd).reshape(1, lookback, n_features)
prob2 = model.predict(high_crowd_norm, verbose=0)[0,0]
print(f"\n🚨 High crowd (speed=0.3, density=2.5):")
print(f"   Congestion probability: {prob2:.4f} → {'CONGESTION' if prob2 >= 0.5 else 'NO CONGESTION'}")

# Test Case 3: Increasing trend (density rising, speed dropping)
trend = np.zeros((lookback, n_features))
for i in range(lookback):
    progress = i / lookback
    trend[i, 0] = 1.2 - 0.8 * progress  # speed decreasing
    trend[i, 1] = 0.3 + 1.5 * progress  # density increasing
    if i > 0:
        trend[i, 2] = trend[i, 0] - trend[i-1, 0]  # speed_delta
        trend[i, 3] = trend[i, 1] - trend[i-1, 1]  # density_delta
trend_norm = scaler.transform(trend).reshape(1, lookback, n_features)
prob3 = model.predict(trend_norm, verbose=0)[0,0]
print(f"\n📈 Increasing crowd trend:")
print(f"   Congestion probability: {prob3:.4f} → {'CONGESTION' if prob3 >= 0.5 else 'NO CONGESTION'}")

# -------------------------------------------------------------------
# Final verdict
# -------------------------------------------------------------------
print("\n" + "="*60)
print("✅ VERDICT")
print("="*60)

if accuracy >= 0.80 and prob2 > 0.7 and prob1 < 0.3:
    print("✅ MODEL IS WORKING CORRECTLY!")
    print("   - Good accuracy on validation set")
    print("   - Correctly identifies high vs low crowd scenarios")
    print("   - Ready for real-time inference")
else:
    print("⚠️  Model needs attention - check the metrics above")

print("="*60)