import pandas as pd
import os

# List all Excel files in the current directory
excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx') or f.endswith('.xls')]

print(f"Found {len(excel_files)} Excel files")

for file in excel_files:
    print(f"Processing {file}...")
    
    try:
        # Get all sheet names
        excel = pd.ExcelFile(file)
        sheet_names = excel.sheet_names
        
        # Create a new Excel writer (with correct filename)
        new_filename = file.split('.')[0] + '_sample.xlsx'
        writer = pd.ExcelWriter(new_filename, engine='openpyxl')
        
        # Process each sheet
        for sheet in sheet_names:
            print(f"  - Reading sheet: {sheet}")
            df = pd.read_excel(file, sheet_name=sheet)
            original_rows = len(df)
            
            # Take only the first 100 rows
            df_reduced = df.head(100)
            
            # Save to the new Excel file
            df_reduced.to_excel(writer, sheet_name=sheet, index=False)
            
            print(f"    ✓ Reduced from {original_rows} to 100 rows")
        
        # Save the Excel file
        writer.close()
        print(f"✓ Created {new_filename} with sample data")
        
    except Exception as e:
        print(f"× Error processing {file}: {e}")

print("\nDone! All sample files created.")