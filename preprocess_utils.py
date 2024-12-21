import h5py
import numpy as np
import pandas as pd
import scipy

FS = 100
NPERSEG = 64
NOVERLAP = 48

def compute_spectrograms(waveforms, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP):
    """
    Compute three-channel spectrogram and log-normalized spectrogram.
    """
    specs = []
    for i in range(3):
        f, t, Sxx = scipy.signal.spectrogram(waveforms[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        specs.append(Sxx)
    specs = np.stack(specs, axis=0)  # (3, F, T)
    return specs

def compute_log_spectrograms(waveforms, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP):
    """
    Compute three-channel spectrogram and log-normalized spectrogram.
    """
    specs = []
    for i in range(3):
        f, t, Sxx = scipy.signal.spectrogram(waveforms[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        specs.append(Sxx)
    specs = np.stack(specs, axis=0)  # (3, F, T)
    log_specs = np.log1p(specs)
    return log_specs

import numpy as np
import scipy.signal
import scipy.stats

def extract_waveform_features(waveform):
    """
    Extract features from a 3-channel seismic waveform.
    Input: waveform array of shape (3, samples)
    Returns: dictionary of features with descriptive names.
    """
    features = {}
    
    for i in range(3):
        channel = waveform[i]
        channel_id = f"c{i}"  # Channel identifier (e.g., c1, c2, c3)
        
        # Basic statistics
        features[f"f1_{channel_id}_maximum_amplitude"] = np.max(channel)
        features[f"f2_{channel_id}_minimum_amplitude"] = np.min(channel)
        features[f"f3_{channel_id}_mean_amplitude"] = np.mean(channel)
        features[f"f4_{channel_id}_std_dev_amplitude"] = np.std(channel)
        features[f"f5_{channel_id}_median_amplitude"] = np.median(channel)
        features[f"f6_{channel_id}_signal_range"] = np.ptp(channel)  # Range
        
        # Energy features
        features[f"f7_{channel_id}_total_energy"] = np.sum(channel**2)
        features[f"f8_{channel_id}_root_mean_square"] = np.sqrt(np.mean(channel**2))
        features[f"f9_{channel_id}_absolute_mean_amplitude"] = np.mean(np.abs(channel))
        
        # Spectral features
        freqs, psd = scipy.signal.welch(channel)  # Power spectral density
        features[f"f10_{channel_id}_peak_spectral_power"] = np.max(psd)
        features[f"f11_{channel_id}_dominant_frequency"] = freqs[np.argmax(psd)]
        features[f"f12_{channel_id}_mean_spectral_power"] = np.mean(psd)
        features[f"f13_{channel_id}_spectral_entropy"] = scipy.stats.entropy(psd / np.sum(psd))
        features[f"f14_{channel_id}_spectral_flatness"] = np.exp(np.mean(np.log(psd))) / np.mean(psd)
        
        # Zero crossings and peaks
        zero_crosses = np.where(np.diff(np.signbit(channel)))[0]
        features[f"f15_{channel_id}_num_zero_crossings"] = len(zero_crosses)
        peaks = len(scipy.signal.find_peaks(channel)[0])
        features[f"f16_{channel_id}_num_peaks"] = peaks
        
        # Higher-order statistics
        features[f"f17_{channel_id}_kurtosis"] = scipy.stats.kurtosis(channel)
        features[f"f18_{channel_id}_skewness"] = scipy.stats.skew(channel)
        features[f"f19_{channel_id}_variance"] = np.var(channel)
        features[f"f20_{channel_id}_iqr"] = np.percentile(channel, 75) - np.percentile(channel, 25)
        
        # Shape-based features
        rise_time = np.argmax(channel) if np.argmax(channel) > 0 else 0
        fall_time = len(channel) - np.argmax(channel) if np.argmax(channel) < len(channel) else 0
        features[f"f21_{channel_id}_rise_time"] = rise_time
        features[f"f22_{channel_id}_fall_time"] = fall_time
        features[f"f23_{channel_id}_peak_to_peak_amplitude"] = np.max(channel) - np.min(channel)
        
    
    return features

