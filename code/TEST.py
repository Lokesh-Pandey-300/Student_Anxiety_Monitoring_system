# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.signal import welch

# Load your dataset
# This should be a preprocessed dataset with relevant features for anxiety detection
data = pd.read_csv('C:\Lokesh\ML_Project\code\emotions.csv')

# Define a function to extract power spectral density features from EEG data
def extract_beta_features(eeg_data):
    # Define the beta frequency range
    beta_range = (13, 30)
    
    # Calculate the power spectral density using Welch's method
    freqs, psd = welch(eeg_data, fs=256, nperseg=512)
    
    # Find the indices corresponding to the beta range
    idx_beta = np.logical_and(freqs >= beta_range[0], freqs <= beta_range[1])
    
    # Sum the power spectral density within the beta range to get the beta power
    beta_power = np.sum(psd[idx_beta], axis=0)
    
    return beta_power

# Apply the feature extraction function to your EEG data column(s)
# Assuming 'eeg_column' is the name of the column containing raw EEG data
data['beta_power'] = data['eeg_column'].apply(extract_beta_features)

# Now 'beta_power' is a feature that can be used for anxiety classification

# Split the dataset into features and target variable
X = data.drop('anxiety_label', axis=1)  # Features
y = data['anxiety_label']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')
