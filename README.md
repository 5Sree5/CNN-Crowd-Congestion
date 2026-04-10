# Congestion Prediction

This project contains a simple architecture for processing crowd data, training an LSTM model, and running real-time inference in a live video stream.

## Structure

```
congestion_prediction/
│
├── data/
│   └── ucsd_crowd.csv
├── models/
├── data_preprocessing.py
├── train_lstm.py
├── realtime_inference.py
├── requirements.txt
└── README.md
```

## Setup

1. Activate the virtual environment (Windows):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Place your dataset in `data/ucsd_crowd.csv`.

## Usage

- `data_preprocessing.py`: Convert raw CSV into LSTM sequence data.
- `train_lstm.py`: Train an LSTM model and save artifacts to `models/`.
- `realtime_inference.py`: Run live prediction from webcam or camera stream.

## Notes

- Update the preprocessing logic to match your dataset column names.
- Replace the YOLO placeholder in `realtime_inference.py` with your YOLO detection code.
- Save your model and scaler artifacts into the `models/` folder.
