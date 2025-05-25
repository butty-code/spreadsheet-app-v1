import pandas as pd
import os

# List all CSV files in the current directory
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

print(f"Found {len(csv_files)} CSV files")

for file in csv_files:
    print(f"Processing {file}...")
    
    # Read the CSV file
    try:
        df = pd.read_csv(file)
        original_rows = len(df)
        
        # Take only the first 100 rows
        df_reduced = df.head(100)
        
        # Create a new filename with "_sample" added
        new_filename = file.replace('.csv', '_sample.csv')
        
        # Save the reduced dataframe
        df_reduced.to_csv(new_filename, index=False)
        
        print(f"✓ Created {new_filename} with 100 rows (reduced from {original_rows} rows)")
    except Exception as e:
        print(f"× Error processing {file}: {e}")

print("\nDone! All sample files created.")