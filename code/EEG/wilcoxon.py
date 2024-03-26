import pandas as pd
import chardet
from scipy.stats import wilcoxon

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Function to read CSV file with manual inspection
def read_csv_with_inspection(file_path, encoding):
    try:
        data = pd.read_csv(file_path, encoding=encoding)
        return data
    except UnicodeDecodeError:
        with open(file_path, 'rb') as f:
            content = f.read()
            print(f"Error decoding file with encoding {encoding}")
            print("Printing first 100 characters of the file content:")
            print(content[:100])
            raise

# Load your data from CSV files
file1 = r"C:\Users\hp\OneDrive\Desktop\Aditya\Relax_Aditya.csv"
file2 = r"C:\Users\hp\OneDrive\Desktop\Aditya\Aditya_PS_Moderate.csv"

# Detect encoding
encoding1 = detect_encoding(file1)
encoding2 = detect_encoding(file2)

# Load data with manual inspection
data1 = read_csv_with_inspection(file1, encoding1)
data2 = read_csv_with_inspection(file2, encoding2)

# List of waves, electrodes, and combinations
waves = ["Alpha", "BetaL", "BetaH", "Gamma", "Theta"]
electrodes = ["F4", "FC6", "T8", "P8", "O2", "O1", "P7", "T7", "FC5", "F3", "AF3", "F7", "F8", "AF4"]

# Perform Wilcoxon signed-rank test for each wave and electrode
for wave in waves:
    for electrode in electrodes:
        column_name1 = f"POW.{electrode}.{wave}"
        column_name2 = f"POW.{electrode}.{wave}"

        # Check for missing values
        if data1[column_name1].isnull().any() or data2[column_name2].isnull().any():
            print(f"Warning: Missing values found in {column_name1} or {column_name2}. Skipping test.")
            continue

        # Ensure equal sample lengths
        min_length = min(len(data1[column_name1]), len(data2[column_name2]))
        trimmed_data1 = data1[column_name1][:min_length]
        trimmed_data2 = data2[column_name2][:min_length]

        # Perform Wilcoxon signed-rank test
        statistic, p_value = wilcoxon(trimmed_data1, trimmed_data2)

        # Output the results for each combination of wave and electrode
        print(f"Wave: {wave}, Electrode: {electrode}")
        print(f"Wilcoxon Statistic: {statistic}")
        print(f"P-value: {p_value}")

        # Check the p-value to determine significance
        alpha = 0.05
        if p_value < alpha:
            print("The difference is statistically significant.")
        else:
            print("The difference is not statistically significant.")
        print("=" * 30)
