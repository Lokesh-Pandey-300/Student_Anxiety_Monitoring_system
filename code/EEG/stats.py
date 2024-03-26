import pandas as pd
from scipy import stats

# Load the data
df = pd.read_csv(r'C:\Users\hp\OneDrive\Desktop\states\RELAX.csv')

# Extract the unique last names (Theta, Alpha, BetaL, BetaH, Gamma)
unique_last_names = set(col.split('.')[-1] for col in df.columns)

# Create new columns for each last name
for last_name in unique_last_names:
    columns_with_last_name = [col for col in df.columns if col.endswith(last_name) and pd.api.types.is_numeric_dtype(df[col])]
    if columns_with_last_name:  # Check if there are numeric columns with the given last name
        df[f'Average_{last_name}'] = df[columns_with_last_name].mean(axis=1)
    else:
        print(f"No numeric columns found for {last_name}")

# Now, you have new columns like 'Average_Theta', 'Average_Alpha', 'Average_BetaL', 'Average_BetaH', 'Average_Gamma'
# Use these new columns for your analysis

# Perform statistical tests only on numeric columns
numeric_columns = df.select_dtypes(include='number').columns
t_stat, p_val = stats.ttest_ind(df[numeric_columns], axis=1)
print(f"T-statistic: {t_stat}, P-value: {p_val}")

u_stat, p_val = stats.mannwhitneyu(df[numeric_columns], axis=1)
print(f"U-statistic: {u_stat}, P-value: {p_val}")
