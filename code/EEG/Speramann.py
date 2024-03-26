import pandas as pd
from scipy.stats import spearmanr

# Load the CSV file into a DataFrame
df = pd.read_csv("C:\Lokesh\ML_Project\code\EEG\average_results.csv")

# Extract the weights and Average Wilcoxon P-value columns for correlation calculation
weights = df['Weights']
wilcoxon_p_values = df['Average Wilcoxon P-value']

# Calculate Spearman's rank correlation coefficient
correlation_coefficient, p_value = spearmanr(weights, wilcoxon_p_values)

print(f"Spearman's rank correlation coefficient: {correlation_coefficient}")
print(f"P-value: {p_value}")
