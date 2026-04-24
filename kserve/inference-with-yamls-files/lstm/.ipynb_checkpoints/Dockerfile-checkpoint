# Dockerfile for LSTM training with MinIO and MLflow
# This Dockerfile sets up a Python environment with necessary libraries for training LSTM models
# using data stored in MinIO and tracking experiments with MLflow.
FROM python:3.11-slim
WORKDIR /app
COPY train_lstm_minio_mlflow.py .
RUN pip install torch pandas numpy scikit-learn mlflow s3fs optuna matplotlib joblib
CMD ["python", "train_lstm_minio_mlflow.py"]

