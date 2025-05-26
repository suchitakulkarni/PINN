import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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

    return best, results

def second_derivative(x, dt=1.0):
    return (x[:, 2:, 0] - 2 * x[:, 1:-1, 0] + x[:, :-2, 0]) / dt**2

def sho_physics_loss(x, dt=1.0, omega=1.0):
    d2x = second_derivative(x, dt)
    x_trimmed = x[:, 1:-1, 0]  # trim to match d2x shape
    physics_residual = d2x + (omega**2) * x_trimmed
    return torch.mean(physics_residual**2)

def combined_loss(x, recon_x, alpha=0.5, dt=1.0, omega=1.0):
    mse = torch.nn.functional.mse_loss(recon_x, x)
    physics = sho_physics_loss(x, dt, omega)
    return mse + alpha * physics


def combined_loss_tuned(x, recon_x, dt=1.0, omega=1.0):
    alpha_range = np.linspace(0, 1, 100)
    losses = []

    for alpha in alpha_range:
        loss = combined_loss(x, recon_x, alpha=alpha, dt=dt, omega=omega)
        losses.append(loss.item())  # Assuming loss is a torch scalar

    best_alpha = alpha_range[np.argmin(losses)]
    best_loss = min(losses)

    return best_alpha, best_loss, alpha_range, losses

def plot_window_construction(x, x_anom, window_size, anomaly_idxs, mode="train"):
    signal = x if mode == "train" else x_anom
    num_windows = len(signal) - window_size + 1

    st.write(f"### {'Clean Training' if mode == 'train' else 'Anomalous Testing'} Data (with Sliding Windows)")

    window_slider = st.slider("Window index", 0, num_windows - 1, 0, key=f"window_slider_{mode}")

    # Plot signal
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=signal,
        mode="lines",
        name="Signal",
        line=dict(color="gray")
    ))

    # Highlight window
    i = window_slider
    fig.add_trace(go.Scatter(
        x=list(range(i, i + window_size)),
        y=signal[i:i + window_size],
        mode="lines+markers",
        name=f"Window {i}",
        line=dict(color="blue" if mode == "train" else "red", width=3)
    ))

    # Optionally show anomaly markers
    if mode == "test" and anomaly_idxs is not None:
        fig.add_trace(go.Scatter(
            x=anomaly_idxs,
            y=signal[anomaly_idxs],
            mode="markers",
            marker=dict(color="black", size=8, symbol="x"),
            name="Anomalies"
        ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Signal",
        title="Sliding Window Visualization"
    )

    st.plotly_chart(fig, use_container_width=True)