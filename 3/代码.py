from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(r"D:\PRML\3")
TRAIN_PATH = BASE_DIR / "LSTM-Multivariate_pollution.csv"
TEST_PATH = BASE_DIR / "pollution_test_data1.csv"

HOLDOUT_OUTPUT = BASE_DIR / "holdout_predictions.csv"
TEST_OUTPUT = BASE_DIR / "pollution_test_predictions.csv"
MODEL_OUTPUT = BASE_DIR / "lstm_pollution_model.pt"

TRAIN_LOSS_FIG = BASE_DIR / "training_loss_curve.png"
HOLDOUT_FIG = BASE_DIR / "holdout_prediction_curve.png"
TEST_FIG = BASE_DIR / "test_prediction_curve.png"
SCATTER_FIG = BASE_DIR / "prediction_scatter.png"
ERROR_HIST_FIG = BASE_DIR / "prediction_error_histogram.png"

SEQ_LEN = 24
HORIZON = 1
BATCH_SIZE = 64
EPOCHS = 15
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.8
SEED = 42


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx]


class PollutionLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def load_train_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    numeric_cols = ["pollution", "dew", "temp", "press", "wnd_spd", "snow", "rain"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pollution"] = df["pollution"].replace(0, np.nan)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    df["wnd_dir"] = df["wnd_dir"].fillna("NA")

    wind_dummies = pd.get_dummies(df["wnd_dir"], prefix="wnd")
    return pd.concat([df, wind_dummies], axis=1)


def load_test_data(path: Path, wind_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = ["pollution", "dew", "temp", "press", "wnd_spd", "snow", "rain"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    df["wnd_dir"] = df["wnd_dir"].fillna("NA")

    wind_dummies = pd.get_dummies(df["wnd_dir"], prefix="wnd")
    for col in wind_columns:
        if col not in wind_dummies.columns:
            wind_dummies[col] = 0
    wind_dummies = wind_dummies[wind_columns]
    return pd.concat([df, wind_dummies], axis=1)


def build_sequences(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    seq_len: int,
    horizon: int,
    start_index: int,
    end_index: int,
):
    sequences = []
    targets = []
    indices = []

    for current_idx in range(start_index + seq_len, end_index - horizon + 1):
        seq_start = current_idx - seq_len
        target_idx = current_idx + horizon - 1
        sequences.append(feature_array[seq_start:current_idx])
        targets.append(target_array[target_idx])
        indices.append(target_idx)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(targets, dtype=np.float32),
        np.array(indices, dtype=np.int64),
    )


def inverse_pollution(values: np.ndarray, scaler: StandardScaler, pollution_idx: int) -> np.ndarray:
    return values * scaler.scale_[pollution_idx] + scaler.mean_[pollution_idx]


def evaluate_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    non_zero_mask = np.abs(actual) > 1e-8
    if np.any(non_zero_mask):
        mape = float(
            np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
        )
    else:
        mape = float("nan")
    return rmse, mae, mape


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            outputs = model(features).cpu().numpy().ravel()
            predictions.append(outputs)
            actuals.append(targets.numpy().ravel())

    return np.concatenate(predictions), np.concatenate(actuals)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_state = None
    best_val_loss = float("inf")
    train_history = []
    val_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                val_losses.append(criterion(outputs, targets).item())

        train_loss = float(np.mean(epoch_losses))
        val_loss = float(np.mean(val_losses))
        train_history.append(train_loss)
        val_history.append(val_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS} - train_loss={train_loss:.4f} - val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_history, val_history


def plot_training_history(train_history: list[float], val_history: list[float]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    epochs = range(1, len(train_history) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(epochs, train_history, marker="o", label="Train Loss")
    ax.plot(epochs, val_history, marker="s", label="Validation Loss")
    ax.set_title("LSTM Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(TRAIN_LOSS_FIG, dpi=180)
    plt.close(fig)


def plot_prediction_curves(holdout_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    recent = holdout_df.tail(168).copy()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(recent["date"], recent["actual_pollution"], label="Actual PM2.5", linewidth=1.6)
    ax.plot(recent["date"], recent["predicted_pollution"], label="Predicted PM2.5", linewidth=1.6)
    ax.set_title("Holdout Set Prediction Comparison (Last 168 Hours)")
    ax.set_xlabel("Time")
    ax.set_ylabel("PM2.5")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(HOLDOUT_FIG, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(test_df["test_row_index"], test_df["actual_pollution"], label="Actual PM2.5", linewidth=1.6)
    ax.plot(test_df["test_row_index"], test_df["predicted_pollution"], label="Predicted PM2.5", linewidth=1.6)
    ax.set_title("External Test Set Prediction Comparison")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("PM2.5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(TEST_FIG, dpi=180)
    plt.close(fig)


def plot_scatter_and_error(test_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(test_df["actual_pollution"], test_df["predicted_pollution"], alpha=0.65, s=22)
    max_value = max(test_df["actual_pollution"].max(), test_df["predicted_pollution"].max())
    ax.plot([0, max_value], [0, max_value], linestyle="--", linewidth=1.2, color="red")
    ax.set_title("Actual vs Predicted PM2.5")
    ax.set_xlabel("Actual PM2.5")
    ax.set_ylabel("Predicted PM2.5")
    fig.tight_layout()
    fig.savefig(SCATTER_FIG, dpi=180)
    plt.close(fig)

    errors = test_df["predicted_pollution"] - test_df["actual_pollution"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(errors, bins=25, color="#4C72B0", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(ERROR_HIST_FIG, dpi=180)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = load_train_data(TRAIN_PATH)
    wind_columns = [
        col
        for col in train_df.columns
        if col.startswith("wnd_") and col not in {"wnd_dir", "wnd_spd"}
    ]
    feature_columns = ["pollution", "dew", "temp", "press", "wnd_spd", "snow", "rain"] + wind_columns
    pollution_idx = feature_columns.index("pollution")

    split_idx = int(len(train_df) * TRAIN_RATIO)
    scaler = StandardScaler()
    scaler.fit(train_df.iloc[:split_idx][feature_columns])
    scaled_features = scaler.transform(train_df[feature_columns])
    scaled_target = scaled_features[:, pollution_idx]

    x_train, y_train, _ = build_sequences(
        scaled_features,
        scaled_target,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        start_index=0,
        end_index=split_idx,
    )
    x_val, y_val, val_indices = build_sequences(
        scaled_features,
        scaled_target,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        start_index=split_idx - SEQ_LEN,
        end_index=len(train_df),
    )

    train_loader = DataLoader(TimeSeriesDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = PollutionLSTM(
        input_size=len(feature_columns),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(device)

    train_history, val_history = train_model(model, train_loader, val_loader, device)

    holdout_pred_scaled, holdout_actual_scaled = predict(model, val_loader, device)
    holdout_pred = inverse_pollution(holdout_pred_scaled, scaler, pollution_idx)
    holdout_actual = inverse_pollution(holdout_actual_scaled, scaler, pollution_idx)

    holdout_results = pd.DataFrame(
        {
            "date": train_df.iloc[val_indices]["date"].to_numpy(),
            "actual_pollution": holdout_actual,
            "predicted_pollution": holdout_pred,
        }
    )
    holdout_results.to_csv(HOLDOUT_OUTPUT, index=False, encoding="utf-8-sig")

    holdout_rmse, holdout_mae, holdout_mape = evaluate_metrics(holdout_actual, holdout_pred)
    print("\nHoldout metrics")
    print(f"RMSE: {holdout_rmse:.3f}")
    print(f"MAE : {holdout_mae:.3f}")
    print(f"MAPE: {holdout_mape:.2f}%")

    test_df = load_test_data(TEST_PATH, wind_columns)
    combined_features = pd.concat(
        [train_df[feature_columns], test_df[feature_columns]],
        axis=0,
        ignore_index=True,
    )
    combined_scaled = scaler.transform(combined_features)
    combined_target = combined_scaled[:, pollution_idx]

    test_start = len(train_df)
    x_test, y_test, test_indices = build_sequences(
        combined_scaled,
        combined_target,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        start_index=test_start - SEQ_LEN,
        end_index=len(combined_scaled),
    )

    test_loader = DataLoader(TimeSeriesDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    test_pred_scaled, test_actual_scaled = predict(model, test_loader, device)
    test_pred = inverse_pollution(test_pred_scaled, scaler, pollution_idx)
    test_actual = inverse_pollution(test_actual_scaled, scaler, pollution_idx)

    test_results = pd.DataFrame(
        {
            "test_row_index": test_indices - len(train_df),
            "actual_pollution": test_actual,
            "predicted_pollution": test_pred,
        }
    )
    test_results.to_csv(TEST_OUTPUT, index=False, encoding="utf-8-sig")

    test_rmse, test_mae, test_mape = evaluate_metrics(test_actual, test_pred)
    print("\nExternal test metrics")
    print(f"RMSE: {test_rmse:.3f}")
    print(f"MAE : {test_mae:.3f}")
    print(f"MAPE: {test_mape:.2f}%")

    plot_training_history(train_history, val_history)
    plot_prediction_curves(holdout_results, test_results)
    plot_scatter_and_error(test_results)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        MODEL_OUTPUT,
    )

    print(f"\nSaved model: {MODEL_OUTPUT}")
    print(f"Saved holdout predictions: {HOLDOUT_OUTPUT}")
    print(f"Saved test predictions: {TEST_OUTPUT}")
    print("Saved figures:")
    print(f" - {TRAIN_LOSS_FIG}")
    print(f" - {HOLDOUT_FIG}")
    print(f" - {TEST_FIG}")
    print(f" - {SCATTER_FIG}")
    print(f" - {ERROR_HIST_FIG}")


if __name__ == "__main__":
    main()
