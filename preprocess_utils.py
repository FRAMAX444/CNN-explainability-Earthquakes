import h5py
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
import librosa

FS = 100
NPERSEG = 64
NOVERLAP = 48

def compute_spectrograms(waveforms, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP):
    """
    Compute three-channel spectrogram and log-normalized spectrogram.
    """
    specs = []
    for i in range(3):
        f, t, Sxx = spectrogram(waveforms[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        specs.append(Sxx)
    specs = np.stack(specs, axis=0)  # (3, F, T)
    return specs

def compute_log_spectrograms(waveforms, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP):
    """
    Compute three-channel spectrogram and log-normalized spectrogram.
    """
    specs = []
    for i in range(3):
        f, t, Sxx = spectrogram(waveforms[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        specs.append(Sxx)
    specs = np.stack(specs, axis=0)  # (3, F, T)
    log_specs = np.log1p(specs)
    return log_specs

def extract_features(waveforms, fs=FS):
    """
    Extract additional features to help classify pre- vs post-events.

    Features:
    - Max and RMS amplitude per channel
    - Spectral centroid and bandwidth (mean over time)
    - Zero-crossing rate (mean)
    - Energy ratio in a specific frequency band (e.g., 5–10 Hz)
    """
    features = {}

    # Basic amplitude features
    for i in range(3):
        ch_data = waveforms[i]
        features[f'max_amp_ch{i}'] = np.max(np.abs(ch_data))
        features[f'rms_amp_ch{i}'] = np.sqrt(np.mean(ch_data**2))

    # Advanced spectral and temporal features using librosa
    # For each channel, compute spectral centroid, bandwidth, and zero-crossing rate
    for i in range(3):
        ch_data = waveforms[i]

        # Spectral centroid & bandwidth
        centroid = librosa.feature.spectral_centroid(y=ch_data, sr=fs)
        bandwidth = librosa.feature.spectral_bandwidth(y=ch_data, sr=fs)
        # Take mean over time frames
        features[f'spectral_centroid_ch{i}'] = np.mean(centroid)
        features[f'spectral_bandwidth_ch{i}'] = np.mean(bandwidth)

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=ch_data)
        features[f'zcr_ch{i}'] = np.mean(zcr)

        # Energy ratio in a band (e.g., 5–10 Hz)
        # Compute FFT
        fft_vals = np.fft.rfft(ch_data)
        freqs = np.fft.rfftfreq(len(ch_data), 1/fs)
        
        # Total energy
        total_energy = np.sum(np.abs(fft_vals)**2)
        
        # Frequency band energy: 5–10 Hz
        band_mask = (freqs >= 5) & (freqs <= 10)
        band_energy = np.sum(np.abs(fft_vals[band_mask])**2)
        
        # Ratio of band energy to total energy
        band_ratio = band_energy / (total_energy + 1e-12)  # Avoid division by zero
        features[f'band_5_10_energy_ratio_ch{i}'] = band_ratio

    return features

