import pandas as pd

# Load the first 10,000 rows (to avoid Excel limitations and processing time)
df = pd.read_csv('raceform_2023_2025.csv', nrows=10000)

# Save as Excel
df.to_excel('raceform_2023_2025.xlsx', index=False)
print(f"Excel file saved with {len(df)} records.")