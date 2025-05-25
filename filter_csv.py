import pandas as pd
import sys
import os

def filter_date_range(input_file, output_file, start_year=2023, end_year=2025):
    """
    Filter a CSV file to only include rows within a specified year range.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the filtered CSV file
        start_year: Starting year for filter (inclusive)
        end_year: Ending year for filter (inclusive)
    """
    print(f"Loading data from {input_file}...")
    
    # Try to determine date column automatically
    try:
        # First load just the header to check column names
        headers = pd.read_csv(input_file, nrows=0).columns.tolist()
        
        # Look for common date column names
        date_columns = [col for col in headers if 'date' in col.lower() 
                       or 'time' in col.lower() 
                       or 'year' in col.lower()
                       or col.lower() == 'dt']
        
        # If no obvious date columns, try the first few rows to detect date formats
        if not date_columns:
            sample = pd.read_csv(input_file, nrows=5)
            for col in sample.columns:
                # Try to convert to datetime to see if it works
                try:
                    pd.to_datetime(sample[col], errors='raise')
                    date_columns.append(col)
                except:
                    pass
    
        # If we found potential date columns, print them for reference
        if date_columns:
            print(f"Detected potential date columns: {date_columns}")
    except Exception as e:
        print(f"Error while trying to detect date columns: {str(e)}")
        date_columns = []
    
    # Load the data in chunks to handle large files
    chunk_size = 100000  # Adjust based on your memory constraints
    chunks = pd.read_csv(input_file, chunksize=chunk_size)
    
    filtered_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}...")
        
        # Ask user for date column if it's the first chunk and we have candidates
        if i == 0 and date_columns:
            print("\nPlease select the date column from the following options:")
            for idx, col in enumerate(date_columns):
                print(f"{idx+1}. {col}")
            print(f"{len(date_columns)+1}. Enter another column name")
            print(f"{len(date_columns)+2}. Show first few rows to help me decide")
            
            choice = input("\nEnter your choice (number): ")
            
            if choice.isdigit() and int(choice) == len(date_columns)+2:
                print("\nHere are the first few rows of the data:")
                print(chunk.head())
                print("\nPlease select the date column from the following options:")
                for idx, col in enumerate(date_columns):
                    print(f"{idx+1}. {col}")
                print(f"{len(date_columns)+1}. Enter another column name")
                choice = input("\nEnter your choice (number): ")
            
            if choice.isdigit() and 1 <= int(choice) <= len(date_columns):
                date_column = date_columns[int(choice)-1]
            elif choice.isdigit() and int(choice) == len(date_columns)+1:
                date_column = input("Enter the date column name: ")
            else:
                date_column = choice
            
            print(f"Using '{date_column}' as the date column.")
        elif i == 0:
            # If no date columns were detected, ask user
            print("\nCouldn't automatically detect date columns.")
            print("Here are the first few rows of the data:")
            print(chunk.head())
            print("\nPlease enter the name of the column containing dates or years:")
            date_column = input()
            print(f"Using '{date_column}' as the date column.")
        
        # Convert the date column to datetime if it's not already
        try:
            # First check if it's a year column (just contains years as integers)
            if chunk[date_column].dtype == 'int64' or chunk[date_column].dtype == 'float64':
                year_values = chunk[date_column].unique()
                if all(1900 <= y <= 2100 for y in year_values if not pd.isna(y)):
                    print(f"Treating '{date_column}' as a year column.")
                    filtered_chunk = chunk[(chunk[date_column] >= start_year) & 
                                         (chunk[date_column] <= end_year)]
                else:
                    # Not years, try as regular dates
                    chunk[date_column] = pd.to_datetime(chunk[date_column], errors='coerce')
                    filtered_chunk = chunk[(chunk[date_column].dt.year >= start_year) & 
                                         (chunk[date_column].dt.year <= end_year)]
            else:
                # Try as regular dates
                chunk[date_column] = pd.to_datetime(chunk[date_column], errors='coerce')
                filtered_chunk = chunk[(chunk[date_column].dt.year >= start_year) & 
                                     (chunk[date_column].dt.year <= end_year)]
        except Exception as e:
            print(f"Error processing dates: {str(e)}")
            print("Continuing with unfiltered data for this chunk...")
            filtered_chunk = chunk
        
        filtered_chunks.append(filtered_chunk)
    
    # Combine all filtered chunks
    filtered_df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"Filtering complete. Original size: {sum(len(chunk) for chunk in chunks)} rows, Filtered size: {len(filtered_df)} rows")
    
    # Save the filtered data
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")
    
    # Also create an Excel version if requested
    excel_output = os.path.splitext(output_file)[0] + '.xlsx'
    save_excel = input(f"Would you like to save as Excel file ({excel_output}) too? (y/n): ")
    if save_excel.lower() == 'y':
        try:
            # If file is too large, save only the first 1M rows
            if len(filtered_df) > 1000000:
                print(f"File too large for Excel. Saving first 1M rows to {excel_output}")
                filtered_df.iloc[:1000000].to_excel(excel_output, index=False)
            else:
                filtered_df.to_excel(excel_output, index=False)
            print(f"Excel file saved to {excel_output}")
        except Exception as e:
            print(f"Error saving Excel file: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_csv.py <input_file.csv> [output_file.csv]")
        print("If output_file is not specified, will use 'filtered_<input_file>'")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Create default output filename
        base_name = os.path.basename(input_file)
        name, ext = os.path.splitext(base_name)
        output_file = f"filtered_{name}_2023-2025{ext}"
    
    filter_date_range(input_file, output_file)