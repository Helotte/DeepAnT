import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from deepant.trainer import AnomalyDetector
from deepant.model import DeepAntPredictor
from utils.data_utils import DataModule

def load_full_dataset(file_path, window_size, device):
    # Load the dataset
    data = pd.read_csv(file_path, parse_dates=["summary_time"], index_col="summary_time")
    
    # Extract feature values and scale them
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values)

    # Generate sliding windows
    data_x = [data_scaled[i - window_size:i] for i in range(window_size, len(data_scaled))]
    data_x = np.array(data_x)

    # Prepare DataLoader
    dataset = DataModule(data_x, data_x[:, -1, :], device)
    loader = DataLoader(dataset, batch_size=data_x.shape[0], shuffle=False)
    return loader, data.index[window_size:]  # Return timestamps starting from the first complete window

def calculate_anomaly_scores(predictions, ground_truth):
    return [np.linalg.norm(pred - gt) for pred, gt in zip(predictions, ground_truth)]

def visualize_anomalies(timestamps, anomaly_scores, threshold, anomalies, output_path):
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, anomaly_scores, label="Anomaly Scores", color='blue', linestyle='-', linewidth=1)
    plt.scatter([timestamps[i] for i in anomalies], [anomaly_scores[i] for i in anomalies], color='red', label='Anomaly Points', zorder=5)
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label="Threshold")
    plt.title("Anomaly Scores and Threshold on the Full Dataset", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def main():
    # Configurations
    checkpoint_path = "experiment/new_york_port/best_model.ckpt"
    data_path = "data/NewYork_port/NewYork_port_data_2020_2024.csv"
    output_path = "experiment/new_york_port/full_dataset_anomaly_plot.png"
    anomaly_timestamps_path = "experiment/new_york_port/full_dataset_anomaly_timestamps.json"
    window_size = 25  # Make sure this matches your training window size
    threshold_std_rate = 3  # Define your threshold multiplier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, timestamps = load_full_dataset(data_path, window_size, device)
    
    model = AnomalyDetector.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=DeepAntPredictor(feature_dim=loader.dataset.data_x.shape[-1], window_size=window_size).to(device),
        lr=1e-3  # Learning rate is not used for inference but is needed for initialization
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = []
        for batch in loader:
            x, _ = batch
            predictions.append(model(x).cpu().numpy())
        predictions = np.concatenate(predictions)

    # Calculate anomaly scores
    ground_truth = loader.dataset.data_y
    anomaly_scores = calculate_anomaly_scores(predictions, ground_truth)

    # Calculate dynamic threshold
    threshold = np.mean(anomaly_scores) + threshold_std_rate * np.std(anomaly_scores)
    
    # Identify anomalies
    anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold]
    anomaly_timestamps = [str(timestamps[i]) for i in anomalies]  # Convert timestamps to string for JSON compatibility
    print("Anomalies detected at timestamps:", [timestamps[i] for i in anomalies])
    
    with open(anomaly_timestamps_path, "w") as json_file:
        json.dump({"anomaly_timestamps": anomaly_timestamps}, json_file)

    # Save anomaly indices
    with open(os.path.join("experiment/new_york_port", "full_dataset_anomalies_indices.json"), "w") as json_file:
        json.dump({"anomalies": [int(i) for i in anomalies]}, json_file)

    # Visualize anomalies
    visualize_anomalies(timestamps, anomaly_scores, threshold, anomalies, output_path)

if __name__ == "__main__":
    main()
