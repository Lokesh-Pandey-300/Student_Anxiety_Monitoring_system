import pandas as pd
import chardet
from scipy.stats import wilcoxon, mannwhitneyu

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

# Perform both Wilcoxon signed-rank test and Mann-Whitney U Test for each wave and all electrodes
for wave in waves:
    # Initialize lists to store results for each electrode
    wilcoxon_statistics = []
    wilcoxon_p_values = []
    mannwhitneyu_statistics = []
    mannwhitneyu_p_values = []

    for electrode in electrodes:
        column_name1 = f"POW.{electrode}.{wave}"
        column_name2 = f"POW.{electrode}.{wave}"

        # Check for missing values
        if data1[column_name1].isnull().any() or data2[column_name2].isnull().any():
            print(f"Warning: Missing values found in {column_name1} or {column_name2}. Skipping tests for {wave} and {electrode}.")
            continue

        # Ensure equal sample lengths
        min_length = min(len(data1[column_name1]), len(data2[column_name2]))
        trimmed_data1 = data1[column_name1][:min_length]
        trimmed_data2 = data2[column_name2][:min_length]

        # Perform Wilcoxon signed-rank test with zero_method='zsplit'
        wilcoxon_statistic, wilcoxon_p_value = wilcoxon(trimmed_data1, trimmed_data2, zero_method='zsplit')
        
        # Perform Mann-Whitney U Test
        mannwhitneyu_statistic, mannwhitneyu_p_value = mannwhitneyu(trimmed_data1, trimmed_data2)

        # Append results for each electrode
        wilcoxon_statistics.append(wilcoxon_statistic)
        wilcoxon_p_values.append(wilcoxon_p_value)
        mannwhitneyu_statistics.append(mannwhitneyu_statistic)
        mannwhitneyu_p_values.append(mannwhitneyu_p_value)

    # Output the results for each wave
    print(f"Wave: {wave}")
    print(f"Wilcoxon Statistics: {wilcoxon_statistics}")
    print(f"Wilcoxon P-values: {wilcoxon_p_values}")
    print(f"Mann-Whitney U Statistics: {mannwhitneyu_statistics}")
    print(f"Mann-Whitney U P-values: {mannwhitneyu_p_values}")
    print("=" * 30)
