# 1. Imports & MinIO client creation adn Download Function

import numpy as np
import io
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from minio import Minio
import datetime

tracking_timestamp =  datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S-")

# Function to download numpy arrays from MinIO
def download_numpy_from_minio(minio_client, bucket, object_name):
    try:
        with minio_client.get_object(bucket, object_name) as response:
            arr = np.load(io.BytesIO(response.read()))
            print(f"Downloaded: s3://{bucket}/{object_name} shape={arr.shape}")
            return arr
    except Exception as e:
        print(f"Error: {e}")
# Minio client
minio_client = Minio(
    "minio-service.kubeflow.svc.cluster.local:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False,
)
# SET MLflow URI in the k8s cluster
# This line must be placed before any mlflow.start_run()
mlflow.set_tracking_uri("http://sunrise-mlflow-tracking.mlflow.svc.cluster.local:5080")
mlflow.set_experiment("k8s-cpu-forecasting")


# 2: Load Train/Val Sets From MinIO
bucket_name = "k8s-resources-forecast"
object_names = {
    "X_train": "data/k8s-preprocessed/node-1-X_train/X_train.npy",
    "y_train": "data/k8s-preprocessed/node-1-y_train/y_train.npy",
    "X_val":   "data/k8s-preprocessed/node-1-X_test/X_test.npy",
    "y_val":   "data/k8s-preprocessed/node-1-y_test/y_test.npy",
}

X_train = download_numpy_from_minio(minio_client, bucket_name, object_names["X_train"])
y_train = download_numpy_from_minio(minio_client, bucket_name, object_names["y_train"])
X_val   = download_numpy_from_minio(minio_client, bucket_name, object_names["X_val"])
y_val   = download_numpy_from_minio(minio_client, bucket_name, object_names["y_val"])

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape, "y_val shape:", y_val.shape)


## 3: Build PyTorch DataLoaders
BATCH_SIZE = 32

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
val_dataset   = TensorDataset(torch.tensor(X_val,   dtype=torch.float32),
                              torch.tensor(y_val,   dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)


# 4: Define LSTM Model
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(1)  # (batch, horizon, 1)

# 5: Training Loop With Early Stopping and MLflow Logging

def train_model_with_early_stopping(
    train_loader, val_loader, input_size=1, hidden_size=64, num_layers=2,
    lr=0.001, epochs=35, patience=5, dropout=0.0, model_name="node-1-cpu-pct-forecast", run_name=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(input_size, hidden_size, num_layers, output_size=1, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_model = None
    wait = 0
    train_losses, val_losses = [], []

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "input_size": input_size, "hidden_size": hidden_size,
            "num_layers": num_layers, "lr": lr, "epochs": epochs,
            "batch_size": BATCH_SIZE, "dropout": dropout, "patience": patience
        })

        for epoch in range(epochs):
            model.train()
            running_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_running_loss = 0
            all_pred, all_true = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    val_running_loss += loss.item()
                    all_pred.append(out.cpu().numpy())
                    all_true.append(yb.cpu().numpy())
            val_loss = val_running_loss / len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    mlflow.log_metric("epoch_actual", epoch + 1)  # real epoch that run
                    print("Early stopping triggered!")
                    break

        # Load best
        if best_model: model.load_state_dict(best_model)

        # Final metrics
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb.to(device)).cpu().numpy()
                preds.append(out)
                targets.append(yb.cpu().numpy())
        preds = np.concatenate(preds).reshape(-1)
        targets = np.concatenate(targets).reshape(-1)

        mae  = mean_absolute_error(targets, preds)
        #rmse = mean_squared_error(targets, preds, squared=False) #  scikit-learn new version has this
        rmse = np.sqrt(mean_squared_error(targets, preds))
        r2   = r2_score(targets, preds)
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_r2", r2)

        # --- Plots
        plt.figure(figsize=(10,4))
        plt.plot(targets, label="True")
        plt.plot(preds, label="Predicted")
        plt.legend(); plt.title("True vs. Predicted CPU% (Validation)")
        #most KFP v2 components, the working directory for your step is /tmp, which is writeable.
        plt.tight_layout(); plt.savefig("/tmp/true_vs_pred.png"); plt.close()
        mlflow.log_artifact("/tmp/true_vs_pred.png")

        plt.figure(figsize=(10,4))
        plt.plot(preds - targets)
        plt.title("Residuals Over Time"); plt.xlabel("Time"); plt.ylabel("Residual (Pred - True)")
        #most KFP v2 components, the working directory for your step is /tmp, which is writeable.
        plt.tight_layout(); plt.savefig("/tmp/residuals.png"); plt.close()
        mlflow.log_artifact("/tmp/residuals.png")

        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Learning Curve")
        #most KFP v2 components, the working directory for your step is /tmp, which is writeable.
        plt.legend(); plt.tight_layout(); plt.savefig("/tmp/learning_curve.png"); plt.close()
        mlflow.log_artifact("/tmp/learning_curve.png")

        # --- (NEW) Infer signature and log model properly ---
        sample_input_t = torch.tensor(X_val[:1], dtype=torch.float32)
        with torch.no_grad():
            sample_output_np = model(sample_input_t).detach().cpu().numpy()
        input_example_np = sample_input_t.cpu().numpy()
        
        from mlflow.models import infer_signature
        signature = infer_signature(input_example_np, sample_output_np)
        
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            input_example=input_example_np,
            signature=signature,
        )
        print("Model + artifacts logged in MLflow (with input_example and signature).")
       

    return model, (mae, rmse, r2)

# 6: Run the Training & Logging
EPOCHS = 35
PATIENCE = 5
#tracking_timestamp =  datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S-") 
model, metrics = train_model_with_early_stopping(
    train_loader, val_loader,
    input_size=X_train.shape[-1],
    hidden_size=64,
    num_layers=2,
    lr=0.001,
    epochs=EPOCHS,
    patience=PATIENCE,
    dropout=0.1,
    model_name= tracking_timestamp + "cpu-node-1-pct-model",
    run_name= tracking_timestamp + "cpu-node-1-forecast" 
)

print(f"Final MAE: {metrics[0]:.4f} | RMSE: {metrics[1]:.4f} | R2: {metrics[2]:.4f}")

