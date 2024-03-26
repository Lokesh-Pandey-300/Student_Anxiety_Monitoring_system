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

# Create empty dictionaries to store results
wilcoxon_results = {}
mannwhitneyu_results = {}

# Perform Wilcoxon signed-rank test (paired) and Mann-Whitney U Test for each wave and electrode
for wave in waves:
    wave_wilcoxon_values = []
    wave_mannwhitneyu_values = []

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
        statistic_wilcoxon, p_value_wilcoxon = wilcoxon(trimmed_data1, trimmed_data2)

        # Perform Mann-Whitney U Test (Wilcoxon Rank-Sum Test)
        statistic_mannwhitneyu, p_value_mannwhitneyu = mannwhitneyu(trimmed_data1, trimmed_data2, alternative='two-sided')

        # Store results in dictionaries
        key = (wave, electrode)
        wilcoxon_results[key] = {'statistic': statistic_wilcoxon, 'p_value': p_value_wilcoxon}
        mannwhitneyu_results[key] = {'statistic': statistic_mannwhitneyu, 'p_value': p_value_mannwhitneyu}

        # Append p-values for each electrode
        wave_wilcoxon_values.append(p_value_wilcoxon)
        wave_mannwhitneyu_values.append(p_value_mannwhitneyu)

    # Calculate and print average values for each wave across electrodes
    print(f"\nAverage values for Wave {wave}:")
    average_wilcoxon_p_value = sum(wave_wilcoxon_values) / len(wave_wilcoxon_values)
    average_mannwhitneyu_p_value = sum(wave_mannwhitneyu_values) / len(wave_mannwhitneyu_values)
    print(f"Average Wilcoxon P-value: {average_wilcoxon_p_value}")
    print(f"Average Mann-Whitney U P-value: {average_mannwhitneyu_p_value}")
    print("=" * 20)
