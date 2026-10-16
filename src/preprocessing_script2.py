# -*- coding: utf-8 -*-
"""
Created on Thu May 15 22:54:10 2025

@author: Burningirl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def convert_time_to_seconds(t):
    try:
        if isinstance(t, pd.Timestamp):
            t = t.time()
        elif isinstance(t, str):
            t = pd.to_datetime(t).time()
        return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
    except Exception as e:
        return np.nan

def remove_outliers(signal, window_size, threshold_factor=5):
    median = pd.Series(signal).rolling(window=window_size, center=True).median()
    deviation = np.abs(signal - median)
    mad = pd.Series(signal).rolling(window=window_size, center=True).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    threshold = threshold_factor * mad
    cleaned_signal = np.copy(signal)
    outliers = deviation > threshold
    cleaned_signal[outliers] = median[outliers]
    return cleaned_signal

def load_and_preprocess(filepath):
    df = pd.read_excel(filepath, header=None)

    try:
        param_start = df[df.apply(lambda row: row.astype(str).str.contains("Item").any(), axis=1)].index[0]
    except IndexError:
        return None, None, None

    signal_df = df.iloc[1:param_start, [0, 1]]
    signal_df.columns = ['time', 'signal']
    signal_df['time'] = signal_df['time'].apply(convert_time_to_seconds)
    signal_df['signal'] = pd.to_numeric(signal_df['signal'], errors='coerce')
    signal_df = signal_df.dropna()

    params_df = df.iloc[param_start + 1:, [0, 1]]
    params_df.columns = ['key', 'value']
    params = dict(zip(params_df['key'], params_df['value']))

    signal = signal_df['signal'].values
    time = signal_df['time'].values

    cleaned_signal = remove_outliers(signal, window_size=200)
    window_length = 5
    if len(cleaned_signal) < window_length:
        window_length = len(cleaned_signal) if len(cleaned_signal) % 2 == 1 else len(cleaned_signal) - 1

    if window_length > 1:
        filtered_signal = savgol_filter(cleaned_signal, window_length=window_length, polyorder=2)
    else:
        filtered_signal = cleaned_signal

    derivative = np.gradient(filtered_signal, time)

    return time, filtered_signal, derivative, params

# Plotting section (example usage)
if __name__ == "__main__":
    filepath = "C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/raw_files/PORH050.xlsx"  # Replace with your actual file path
    time, signal, derivative, params = load_and_preprocess(filepath)

    if time is not None:
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
        # Plot filtered signal
        axs[0].plot(time, signal, label='Filtered Signal', color='blue', linewidth=2)
        axs[0].set_ylabel("Amplitude")
        axs[0].set_title("Filtered Signal")
        axs[0].legend()
        axs[0].grid(True)
    
        # Plot derivative
        axs[1].plot(time, derivative, label='Derivative', color='orange', linestyle='--', linewidth=2)
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Derivative")
        axs[1].set_title("Signal Derivative")
        axs[1].legend()
        axs[1].grid(True)
    
        plt.tight_layout()
        plt.show()
    else:
        print("Failed to load or preprocess the data.")

