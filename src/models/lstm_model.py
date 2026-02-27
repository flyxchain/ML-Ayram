"""
src/models/lstm_model.py
Modelo LSTM con PyTorch para clasificación de señales forex.
Captura patrones secuenciales que XGBoost no puede ver.
Clasifica: +1 (long ganador), 0 (neutral), -1 (long perdedor)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sqlalchemy import create_engine
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
MODELS_DIR   = Path("models/saved")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"LSTM usando: {DEVICE}")

SEQUENCE_LEN = 60   # velas de contexto hacia atrás
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}

FEATURE_COLS = [
    "ema_20", "ema_50", "ema_200",
    "macd_line", "macd_signal", "macd_hist",
    "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7",
    "stoch_k", "stoch_d",
    "williams_r", "roc_10", "cci_20",
    "atr_14", "atr_7",
    "bb_width", "bb_pct",
    "kc_width", "kc_pct",
    "dc_width",
    "volume_ratio_20",
    "price_vs_sh", "price_vs_sl",
    "trend_direction",
    "body_size", "upper_wick", "lower_wick", "is_bullish",
    "log_return_1", "log_return_5", "log_return_10",
    "close_vs_ema20", "close_vs_ema50", "close_vs_ema200",
    "hour_of_day", "day_of_week",
    "is_london", "is_newyork", "is_overlap",
    "htf_trend", "htf_rsi", "htf_adx",
]


# ── Dataset ──────────────────────────────────────────────────────────────────

class ForexSequenceDataset(Dataset):
    """
    Crea secuencias de longitud SEQUENCE_LEN para el LSTM.
    Cada muestra es una ventana de velas consecutivas.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQUENCE_LEN):
        self.X       = torch.FloatTensor(X)
        self.y       = torch.LongTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]          # (seq_len, n_features)
        label = self.y[idx + self.seq_len]                 # etiqueta de la vela siguiente
        return x_seq, label


# ── Arquitectura ─────────────────────────────────────────────────────────────

class ForexLSTM(nn.Module):
    def __init__(
        self,
        input_size:   int,
        hidden_size:  int  = 128,
        num_layers:   int  = 2,
        dropout:      float = 0.3,
        num_classes:  int  = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            dropout      = dropout if num_layers > 1 else 0,
            batch_first  = True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)                         # (batch, seq_len, hidden)

        # Attention: pondera las velas más relevantes
        attn_scores  = self.attention(lstm_out)            # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)   # (batch, seq_len, 1)
        context      = (lstm_out * attn_weights).sum(dim=1) # (batch, hidden)

        logits = self.classifier(context)                  # (batch, num_classes)
        return logits


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def load_dataset(pair: str, timeframe: str):
    """Carga y preprocesa datos desde la BD."""
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql(
        f"""SELECT * FROM features_computed
            WHERE pair='{pair}' AND timeframe='{timeframe}'
            AND label IS NOT NULL
            ORDER BY timestamp""",
        engine
    )
    if df.empty:
        raise ValueError(f"Sin datos etiquetados para {pair} {timeframe}")

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(df[available].median()).values.astype(np.float32)
    y = df["label"].map(LABEL_MAP).values.astype(np.int64)

    logger.info(f"Dataset {pair} {timeframe}: {len(X)} filas, {len(available)} features")
    return X, y, available


def train_lstm(
    pair:        str,
    timeframe:   str,
    hidden_size: int   = 128,
    num_layers:  int   = 2,
    dropout:     float = 0.3,
    lr:          float = 1e-3,
    epochs:      int   = 50,
    batch_size:  int   = 128,
    seq_len:     int   = SEQUENCE_LEN,
    val_split:   float = 0.2,
    patience:    int   = 10,
) -> tuple:
    """
    Entrena el LSTM. Respeta el orden temporal (sin shuffle).
    Usa early stopping basado en F1 de validación.
    Devuelve (model, scaler, metrics, feature_cols).
    """
    X_raw, y, feature_cols = load_dataset(pair, timeframe)

    # Normalizar con StandardScaler (fit solo en train)
    split_idx = int(len(X_raw) * (1 - val_split))
    scaler    = StandardScaler()
    X_train   = scaler.fit_transform(X_raw[:split_idx])
    X_val     = scaler.transform(X_raw[split_idx:])
    y_train   = y[:split_idx]
    y_val     = y[split_idx:]

    train_ds = ForexSequenceDataset(X_train, y_train, seq_len)
    val_ds   = ForexSequenceDataset(X_val,   y_val,   seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Peso de clases para desbalanceo
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(1.0 / (class_counts + 1)).to(DEVICE)

    model     = ForexLSTM(len(feature_cols), hidden_size, num_layers, dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_f1      = 0.0
    best_state   = None
    no_improve   = 0
    history      = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validación
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                logits = model(X_b)
                val_loss += criterion(logits, y_b).item()
                preds    = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_b.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        scheduler.step(val_loss)
        history.append({"epoch": epoch+1, "val_f1": val_f1, "val_loss": val_loss})

        if (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} — val_f1: {val_f1:.4f}  val_loss: {val_loss/len(val_dl):.4f}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping en epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    logger.success(f"  ✅ Mejor val F1: {best_f1:.4f}")

    metrics = {
        "best_val_f1": float(best_f1),
        "epochs_run":  epoch + 1,
        "history":     history[-10:],  # últimas 10 épocas
    }
    return model, scaler, metrics, feature_cols


def save_model(
    model:        ForexLSTM,
    scaler:       StandardScaler,
    pair:         str,
    timeframe:    str,
    metrics:      dict,
    feature_cols: list,
) -> Path:
    """Guarda modelo PyTorch + scaler + metadatos."""
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M")
    name = f"lstm_{pair}_{timeframe}_{ts}"
    path = MODELS_DIR / f"{name}.pt"

    torch.save({
        "model_state":   model.state_dict(),
        "model_config": {
            "input_size":  model.lstm.input_size,
            "hidden_size": model.hidden_size,
            "num_layers":  model.num_layers,
        },
        "scaler_mean":  scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_cols": feature_cols,
        "label_map":    LABEL_MAP,
        "metrics":      metrics,
        "pair":         pair,
        "timeframe":    timeframe,
        "trained_at":   ts,
    }, str(path))

    logger.success(f"Modelo LSTM guardado: {path}")
    return path


def load_model(pair: str, timeframe: str) -> tuple:
    """Carga el modelo LSTM más reciente para un par/timeframe."""
    pattern = f"lstm_{pair}_{timeframe}_*.pt"
    files   = sorted(MODELS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No hay modelo LSTM guardado para {pair} {timeframe}")

    checkpoint = torch.load(str(files[-1]), map_location=DEVICE)
    cfg   = checkpoint["model_config"]
    model = ForexLSTM(
        input_size  = cfg["input_size"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    scaler        = StandardScaler()
    scaler.mean_  = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]

    feature_cols = checkpoint["feature_cols"]
    logger.info(f"Modelo LSTM cargado: {files[-1]}")
    return model, scaler, feature_cols


def predict(
    model:        ForexLSTM,
    scaler:       StandardScaler,
    feature_cols: list,
    df:           pd.DataFrame,
    seq_len:      int = SEQUENCE_LEN,
) -> np.ndarray:
    """
    Genera predicciones para un DataFrame.
    Devuelve array con valores -1, 0, +1.
    Requiere al menos seq_len filas.
    """
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(df[available].median()).values.astype(np.float32)
    X = scaler.transform(X)

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(seq_len, len(X)):
            seq    = torch.FloatTensor(X[i - seq_len:i]).unsqueeze(0).to(DEVICE)
            logits = model(seq)
            pred   = logits.argmax(dim=1).item()
            preds.append(LABEL_MAP_INV[pred])

    # Las primeras seq_len filas no tienen predicción
    result = np.full(len(df), np.nan)
    result[seq_len:] = preds
    return result


def predict_proba(
    model:        ForexLSTM,
    scaler:       StandardScaler,
    feature_cols: list,
    df:           pd.DataFrame,
    seq_len:      int = SEQUENCE_LEN,
) -> pd.DataFrame:
    """
    Devuelve probabilidades por clase.
    Columnas: prob_short, prob_neutral, prob_long
    """
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(df[available].median()).values.astype(np.float32)
    X = scaler.transform(X)

    probas = []
    model.eval()
    with torch.no_grad():
        for i in range(seq_len, len(X)):
            seq    = torch.FloatTensor(X[i - seq_len:i]).unsqueeze(0).to(DEVICE)
            logits = model(seq)
            proba  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probas.append(proba)

    result = pd.DataFrame(probas, columns=["prob_short", "prob_neutral", "prob_long"])
    return result


def train_and_save(
    pair:      str,
    timeframe: str,
    **kwargs,
) -> ForexLSTM:
    """Pipeline completo: carga → entrena → guarda."""
    model, scaler, metrics, feature_cols = train_lstm(pair, timeframe, **kwargs)
    save_model(model, scaler, pair, timeframe, metrics, feature_cols)
    return model


if __name__ == "__main__":
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
    for pair in pairs:
        try:
            train_and_save(pair, "H1", epochs=50, patience=10)
        except Exception as e:
            logger.error(f"Error {pair}: {e}")
