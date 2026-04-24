import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import matplotlib.pyplot as plt
import optuna

# --- Config ---
S3_URL = "s3://cpu-demo/preprocessed.csv"
MLFLOW_TRACKING_URI = "http://mlflow.isiath.duckdns.org:8082"
MLFLOW_S3_ENDPOINT_URL = "http://minio.isiath.duckdns.org:8082"
AWS_ACCESS_KEY_ID = "admin"
AWS_SECRET_ACCESS_KEY = "password"
MLFLOW_EXPERIMENT = "cpu-forecast"
MODEL_NAME = "cpu-forecast-lstm"
WINDOW_SIZE = 5

# --- ENV Setup (for mlflow + s3fs) ---
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# --- 1. Load Data ---
df = pd.read_csv(S3_URL)
series = df['cpu_pct'].values.astype(float)   # use your real column name!

# --- 2. Sliding Window ---
def make_windows(series, window=5, horizon=1):
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i:i+window])
        y.append(series[i+window: i+window+horizon])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y).reshape(-1, horizon)
    return X, y

X, y = make_windows(series, window=WINDOW_SIZE, horizon=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)

# Simple train/test split
split = int(len(X)*0.8)
Xtr, Xte = X_scaled[:split], X_scaled[split:]
ytr, yte = y[:split], y[split:]

# --- 3. Model ---
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model(hidden=32, layers=1, dropout=0.0, lr=0.001, batch=32, epochs=20):
    device = torch.device("cpu")
    model = LSTMForecaster(hidden_size=hidden, num_layers=layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()
    return model

# --- 4. HPO + MLflow tracking ---
def objective(trial):
    hidden = trial.suggest_int("hidden", 16, 128, step=16)
    layers = trial.suggest_int("layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch = trial.suggest_categorical("batch", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 15, 35)

    with mlflow.start_run(nested=True):
        mlflow.log_params({"hidden": hidden, "layers": layers, "dropout": dropout, "lr": lr, "batch": batch, "epochs": epochs, "window": WINDOW_SIZE})
        model = train_model(hidden, layers, dropout, lr, batch, epochs)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte, dtype=torch.float32))
            mse = mean_squared_error(yte, pred.numpy())
        mlflow.log_metric("mse", mse)
        return mse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=8)  # few trials for demo

best = study.best_params
with mlflow.start_run(run_name="best_run") as run:
    mlflow.log_params({**best, "window": WINDOW_SIZE})
    model = train_model(best["hidden"], best["layers"], best["dropout"], best["lr"], best["batch"], best["epochs"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte, dtype=torch.float32))
        mse = mean_squared_error(yte, pred.numpy())
    mlflow.log_metric("mse", mse)
    # Save scaler
    joblib.dump(scaler, "scaler.pkl")
    mlflow.log_artifact("scaler.pkl")
    # Save prediction plot
    plt.figure(figsize=(10,4))
    plt.plot(yte, label="True")
    plt.plot(pred.numpy(), label="Predicted")
    plt.legend(); plt.title("Forecasting Test vs Prediction")
    plt.savefig("forecast_plot.png"); plt.close()
    mlflow.log_artifact("forecast_plot.png")
    # Save the model
    mlflow.pytorch.log_model(model, artifact_path="model")
    # Register model
    result = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=MODEL_NAME)
    # Add tag for this run/model version
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(name=MODEL_NAME, version=result.version, key="stage", value="demo")
    print("Done. Run:", run.info.run_id, "Version:", result.version, "MSE:", mse)
