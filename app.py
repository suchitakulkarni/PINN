import streamlit as st
import numpy as np
import pandas as pd
import torch
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from model import LSTMAutoencoder
from utils import *

st.set_page_config(page_title="Anomaly Detection in Harmonic Oscillator", layout="wide")

# Sidebar Parameters
st.sidebar.title("Simulation Settings")
st.sidebar.markdown(":red[Signal generation settings]")
omega = st.sidebar.slider("Oscillation period", 0.1, 100.0, 2.0, help="Controls how fast the signal oscillates")
timesteps = st.sidebar.slider("Time steps", 500, 3000, 1000, help="Controls how long the signal oscillates")
num_anomalies = st.sidebar.slider("Anomalies", 5, 50, 10, help="Controls the number of injected anomalies")
severity = st.sidebar.slider("Perturbation Severity", 0.5, 5.0, 2.0, help="Controls the height or strength of the anomalies")
st.sidebar.markdown(":red[Hyperparameters]")
window_size = st.sidebar.slider("Window Size", 10, 50, 20, \
    help="number of consecutive time steps fed into the LSTM as a single input sequence, optimal window size captures complete oscillation cycle and their natural variation.")
hidden_dim = st.sidebar.slider("Hidden Dim", 4, 128, 16,\
    help="number of neurons in each LSTM layer, too small hidden dimension forces strong compression and learns only essential features.")
epochs = st.sidebar.slider("Epochs", 1, 100, 30, \
    help="number of times the model sees the entire training dataset, too small leads to underfitting and too large gives overfitting.")
learning_rate = st.sidebar.number_input("Learning Rate", 1e-5, 1e-1, 1e-3, format="%.5f",\
    help="step size for weight updates during training, too large learning rate may lead to wild oscillations in training loss, bad convergence or overshooting optimal weight. Too small learning rate leads to very slow learning, needs many epochs and may get stuck in local minima.")

st.title("LSTM Autoencoder: Anomaly Detection in a Simulated Harmonic Oscillator")

# Simulation
x = simulate_harmonic_oscillator(timesteps=timesteps, omega=omega)
x_anom, anomaly_idxs = inject_perturbations(x, num_anomalies, severity)

# Rolling window for clean training data
X_windows_clean = create_rolling_dataset(x, window_size=window_size)

# Rolling window for anomalous testing data
X_windows_anom = create_rolling_dataset(x_anom, window_size=window_size)

st.markdown("Generated anomalous signal")
# Update plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    y=x_anom,
    mode='lines',
    name='Anomalous Signal',
    line=dict(color='blue')))

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Amplitude",
    template="plotly_white",
    margin=dict(t=20, b=40, l=40, r=20),  # Reduce top margin especially
    xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
    yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True)
)

st.plotly_chart(fig, use_container_width=True)

# Normalize based only on clean data
scaler = StandardScaler()
X_windows_clean_scaled = scaler.fit_transform(X_windows_clean)
X_windows_anom_scaled = scaler.transform(X_windows_anom)

# Reshape
X_train = X_windows_clean_scaled[..., np.newaxis]
X_test = X_windows_anom_scaled[..., np.newaxis]
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=32, shuffle=True)

# Model
model = LSTMAutoencoder(seq_len = window_size, hidden_dim = hidden_dim)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

st.markdown("Training Loss")
loss_placeholder = st.empty()
loss_data = pd.DataFrame(columns=["Epoch", "Loss"])

fig = go.Figure()
fig.update_layout(
    xaxis_title="Epoch",
    yaxis_title="Loss",
    template="plotly_white",
    margin=dict(t=20, b=40, l=40, r=20),
    xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
    yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True)
)

# Training loop
losses = []
# Initialize empty DataFrame for dynamic updates
loss_df = pd.DataFrame({"Epoch": [], "Loss": []})

# Add progress bar
progress = st.progress(0)
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        seq = batch[0]
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, seq)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    new_row = pd.DataFrame({"Epoch": [epoch], "Loss": [loss.item()]})
    loss_df = pd.concat([loss_df, new_row], ignore_index=True)
    # Update the chart in-place

    # Update plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=loss_df["Epoch"],
        y=loss_df["Loss"],
        mode="lines+markers",
        name="Training Loss"
    ))
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        margin=dict(t=20, b=40, l=40, r=20),
        xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True)
    )

    loss_placeholder.plotly_chart(fig, use_container_width=True)

    # Optional: update progress bar
    progress.progress((epoch + 1) / epochs)

# Convert test data
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    recon = model(X_test_tensor).numpy()

errors = np.mean((X_test - recon)**2, axis=(1, 2))

# ======= Validation with Injected Anomalies =======

# Re-simulate a clean + anomalous signal for validation
x_val = simulate_harmonic_oscillator(timesteps=timesteps)
x_val_anom, val_anomaly_idxs = inject_perturbations(x_val, num_anomalies, severity)
X_val = create_rolling_dataset(x_val_anom, window_size=window_size)

# Scale using previously fitted scaler
X_val_scaled = scaler.transform(X_val)
X_val_lstm = X_val_scaled[..., np.newaxis]
X_val_tensor = torch.tensor(X_val_lstm, dtype=torch.float32)

# Inference
model.eval()
with torch.no_grad():
    recon_val = model(X_val_tensor).numpy()

val_errors = np.mean((X_val_lstm - recon_val) ** 2, axis=(1, 2))

best_result, threshold_metrics = tune_threshold_f1(val_errors, val_anomaly_idxs, window_size)
threshold = best_result["threshold"]
detected_idxs = best_result["detected_idxs"]
precision = best_result["precision"]
recall = best_result["recall"]

# Convert to DataFrame
df_metrics = pd.DataFrame(threshold_metrics)

# Plot Precision, Recall, F1
fig = px.line(df_metrics, x="threshold", y=["precision", "recall", "f1"],
              title="Threshold Tuning: Precision / Recall / F1",
              labels={"value": "Score", "threshold": "Threshold"},
              template="plotly_white")

# Add vertical line for best threshold
fig.add_vline(x=best_result["threshold"], line_dash="dash", line_color="red",
              annotation_text="Best Threshold", annotation_position="top right")

fig.update_layout(
        xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True))

st.subheader("Threshold Optimization")
st.plotly_chart(fig, use_container_width=True)

# Calculate optimal threshold via percentile
val_anomalies = val_errors > threshold

# Map rolling windows to time indices
window_centers = np.arange(window_size // 2, len(x_val_anom) - window_size // 2)

# Match window-level anomalies to injected anomalies
detected_idxs = window_centers[val_anomalies]
true_positives = [idx for idx in val_anomaly_idxs if any(abs(idx - d) <= window_size // 2 for d in detected_idxs)]
false_negatives = [idx for idx in val_anomaly_idxs if idx not in true_positives]
precision = len(true_positives) / len(detected_idxs) if len(detected_idxs) > 0 else 0
recall = len(true_positives) / len(val_anomaly_idxs) if len(val_anomaly_idxs) > 0 else 0

st.subheader("Anomaly detection results")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.4, 0.3, 0.3],
    vertical_spacing=0.1,
    subplot_titles=(
        "Anomalous Signal",
        "Reconstruction Error",
        "Detected Anomalies",
    )
)

# --- 1. Signal plot ---
fig.add_trace(go.Scatter(
    y=x_anom,
    mode='lines',
    name='Anomalous Signal',
    line=dict(color='blue')
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=anomaly_idxs,
    y=x_anom[anomaly_idxs],
    mode='markers',
    name='Injected Anomalies',
    marker=dict(color='red', size=8, symbol='x')
), row=1, col=1)

# --- 2. Error plot ---
fig.add_trace(go.Scatter(
    y=errors,
    mode='lines',
    name='Reconstruction Error',
    line=dict(color='orange')
), row=2, col=1)

fig.add_trace(go.Scatter(
    y=[threshold] * len(errors),
    mode='lines',
    name='Threshold',
    line=dict(color='red', dash='dash')
), row=2, col=1)

# --- 3. Anomaly indicator (binary) ---
anomaly_indicator = np.zeros_like(errors)
anomaly_indicator[val_anomalies] = 1

fig.add_trace(go.Scatter(
    y=anomaly_indicator,
    mode='lines+markers',
    name='Detected Anomalies',
    line=dict(color='green'),
    marker=dict(symbol='circle')
), row=3, col=1)

# --- Layout ---
fig.update_layout(
    #margin=dict(t=500),
    height=800, 
    showlegend=True,
    template="plotly_white",
    hovermode='x unified',  # Sync hover across subplots
    xaxis=dict(showline=True, linewidth=2, linecolor="black", mirror=True),   # xaxis for row=1
    yaxis=dict(showline=True, linewidth=2, linecolor="black", mirror=True),   # yaxis for row=1
    xaxis2=dict(showline=True, linewidth=2, linecolor="black", mirror=True),  # xaxis for row=2
    yaxis2=dict(showline=True, linewidth=2, linecolor="black", mirror=True),  # yaxis for row=2
    xaxis3=dict(showline=True, linewidth=2, linecolor="black", mirror=True),  # xaxis for row=3
    yaxis3=dict(showline=True, linewidth=2, linecolor="black", mirror=True),  # yaxis for row=3
)

# Adjust subplot title properties
fig.update_annotations(
    font_size=14,
    font_color="black",
    yshift=10  # Move titles up slightly
)

fig.update_xaxes(title_text="Time", row=3, col=1)
fig.update_yaxes(title_text="Amplitude", row=1, col=1)
fig.update_yaxes(title_text="Error", row=2, col=1)
fig.update_yaxes(title_text="Anomaly Flag", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)


st.markdown(f"""
- **Validation Detected:** `{len(detected_idxs)}`
- **True Injected:** `{len(val_anomaly_idxs)}`
- **True Positives:** `{best_result['tp']}`
- **False Positives:** `{best_result['fp']}`
- **Precision:** `{precision:.2f}`
- **Recall:** `{recall:.2f}`
- **F1 Score:** `{best_result['f1']:.2f}`
""")