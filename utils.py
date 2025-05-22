import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

def simulate_harmonic_oscillator(timesteps=1000, dt=0.01, omega=2.0):
    x = np.zeros(timesteps)
    v = np.zeros(timesteps)
    x[0] = 1.0
    for t in range(1, timesteps):
        a = -omega**2 * x[t-1]
        v[t] = v[t-1] + a * dt
        x[t] = x[t-1] + v[t] * dt
    return x

def inject_perturbations(x, num_anomalies=10, severity=2.0):
    x_anomalous = x.copy()
    anomaly_indices = np.random.choice(len(x), num_anomalies, replace=False)
    for idx in anomaly_indices:
        if idx < len(x) - 1:
            x_anomalous[idx] += severity * np.random.randn()
    return x_anomalous, anomaly_indices

def create_rolling_windows(data, window_size):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
    return np.array(X)
    
def prepare_data(data, window_size):
    windows = create_rolling_windows(data, window_size)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(windows)
    return scaled[..., np.newaxis], scaler

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size = x.size(0)
        _, (h_n, _) = self.encoder(x)
        decoder_input = torch.zeros((batch_size, self.seq_len, 1), device=x.device)
        decoder_out, _ = self.decoder(decoder_input, (h_n, torch.zeros_like(h_n)))
        out = self.output_layer(decoder_out)
        return out

def create_rolling_dataset(x, window_size=20):
    X = []
    for i in range(len(x) - window_size):
        X.append(x[i:i+window_size])
    return np.array(X)


def default_layout(title, xaxis="Time", yaxis="Amplitude"):
    return dict(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        template="plotly_white",
        legend=dict(x=0, y=1.1, orientation="h"),
    )

import numpy as np

def tune_threshold_f1(errors, true_anomaly_idxs, window_size, num_thresholds=100):
    thresholds = np.linspace(np.min(errors), np.max(errors), num_thresholds)
    results = []
    best_threshold = None
    best_f1 = -1
    best_metrics = {}

    window_centers = np.arange(window_size // 2, len(errors) + window_size // 2)

    for t in thresholds:
        preds = errors > t
        detected_idxs = window_centers[preds]

        # Match detected to true using proximity
        true_positives = [
            idx for idx in true_anomaly_idxs
            if any(abs(idx - d) <= window_size // 2 for d in detected_idxs)
        ]

        false_positives = [
            d for d in detected_idxs
            if not any(abs(d - idx) <= window_size // 2 for idx in true_anomaly_idxs)
        ]

        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(true_anomaly_idxs) - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "detected_idxs": detected_idxs
        })

        best = max(results, key=lambda x: x["f1"])

        #if f1 > best_f1:
        #    best_f1 = f1
        #    best_threshold = t
        #    best_metrics = {
        #        "threshold": t,
        #        "precision": precision,
        #        "recall": recall,
        #        "f1": f1,
        #        "tp": tp,
        #        "fp": fp,
        #        "fn": fn,
        #        "detected_idxs": detected_idxs,
        #    }

    return best, results
