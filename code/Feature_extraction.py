import numpy as np
import pandas as pd
from scipy.fft import fft

# Assuming 'eeg_data' is a DataFrame with each column representing an EEG channel
eeg_data = pd.read_csv('path_to_your_eeg_data.csv')

# Define a function to apply FFT and extract features for each channel
def extract_fft_features(eeg_channel_data):
    # Apply FFT on the channel data
    fft_result = fft(eeg_channel_data)
    
    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(fft_result)
    
    # Compute the power spectrum
    power_spectrum = np.square(magnitude_spectrum)
    
    # Optionally, compute the log power spectrum for better visualization
    log_power_spectrum = np.log(power_spectrum + 1)
    
    # Return the desired feature, e.g., magnitude spectrum, power spectrum, etc.
    return log_power_spectrum

# Apply the feature extraction function to each channel
fft_features = eeg_data.apply(extract_fft_features, axis=0)

# Now 'fft_features' contains the extracted features for each channel
