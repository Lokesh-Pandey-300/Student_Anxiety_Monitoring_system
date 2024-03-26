import os
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
file1 = r"C:\Users\hp\OneDrive\Desktop\states\pradeep\Pradeep_Relax.csv"
file2 = r"C:\Users\hp\OneDrive\Desktop\states\pradeep\Pradeep_Scary.csv"

# Detect encoding
encoding1 = detect_encoding(file1)
encoding2 = detect_encoding(file2)

# Load data with manual inspection
data1 = read_csv_with_inspection(file1, encoding1)
data2 = read_csv_with_inspection(file2, encoding2)

# List of waves, electrodes, and combinations
waves = ["Alpha", "BetaL", "BetaH", "Gamma", "Theta"]
electrodes = ["F4", "FC6", "T8", "P8", "O2", "O1", "P7", "T7", "FC5", "F3", "AF3", "F7", "F8", "AF4"]

# Create empty lists to store results
wilcoxon_p_values = []
wilcoxon_statistic_values = []
mannwhitneyu_p_values = []
mannwhitneyu_statistic_values = []
file_names = []

# Specify the full path for the output CSV file
output_csv_path = r"C:\Lokesh\ML_Project\code\EEG\average_results.csv"

# Perform Wilcoxon signed-rank test (paired) and Mann-Whitney U Test for each wave and electrode
for wave in waves:
    wave_wilcoxon_values = []
    wave_mannwhitneyu_values = []
    wave_wilcoxon_statistic_values = []
    wave_mannwhitneyu_statistic_values = []

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

        # Append p-values and statistical values for each electrode
        wave_wilcoxon_values.append(p_value_wilcoxon)
        wave_mannwhitneyu_values.append(p_value_mannwhitneyu)
        wave_wilcoxon_statistic_values.append(statistic_wilcoxon)
        wave_mannwhitneyu_statistic_values.append(statistic_mannwhitneyu)

    # Calculate and append average p-values and statistical values for each wave across electrodes
    average_wilcoxon_p_value = sum(wave_wilcoxon_values) / len(wave_wilcoxon_values)
    average_mannwhitneyu_p_value = sum(wave_mannwhitneyu_values) / len(wave_mannwhitneyu_values)
    average_wilcoxon_statistic = sum(wave_wilcoxon_statistic_values) / len(wave_wilcoxon_statistic_values)
    average_mannwhitneyu_statistic = sum(wave_mannwhitneyu_statistic_values) / len(wave_mannwhitneyu_statistic_values)

    wilcoxon_p_values.append(average_wilcoxon_p_value)
    mannwhitneyu_p_values.append(average_mannwhitneyu_p_value)
    wilcoxon_statistic_values.append(average_wilcoxon_statistic)
    mannwhitneyu_statistic_values.append(average_mannwhitneyu_statistic)
    file_names.append(f"{file1}, {file2}")

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Wave': waves,
    'File Names': file_names,
    'Average Wilcoxon P-value': wilcoxon_p_values,
    'Average Mann-Whitney U P-value': mannwhitneyu_p_values,
    'Average Wilcoxon Statistic': wilcoxon_statistic_values,
    'Average Mann-Whitney U Statistic': mannwhitneyu_statistic_values
})

# Save the updated DataFrame to a CSV file with append mode or create a new one
if os.path.exists(output_csv_path):
    existing_data = pd.read_csv(output_csv_path)
    results_df.to_csv(output_csv_path, index=False, mode='a', header=False)
else:
    results_df.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")
