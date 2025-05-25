import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
from utils import get_numeric_columns, get_object_columns

def get_sheet_names(file):
    """Get all sheet names from an Excel file."""
    xls = pd.ExcelFile(file)
    return xls.sheet_names

def excel_to_csv(file, sheet_name):
    """
    Convert an Excel sheet to CSV format for better performance and compatibility.
    
    Args:
        file: The uploaded Excel file
        sheet_name: Name of the sheet to convert
        
    Returns:
        Pandas DataFrame from the converted CSV
    """
    # Read the Excel sheet
    df = pd.read_excel(file, sheet_name=sheet_name)
    
    # Convert to CSV (in memory)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Read back as CSV
    processed_df = pd.read_csv(csv_buffer)
    
    return processed_df

def process_excel_file(file):
    """
    Process an Excel file by converting all sheets to CSV format.
    
    Args:
        file: The uploaded Excel file
        
    Returns:
        Dictionary with sheet names as keys and DataFrames as values
    """
    sheet_names = get_sheet_names(file)
    all_dfs = {}
    
    with st.spinner(f"Converting {len(sheet_names)} sheets to optimized format..."):
        for sheet in sheet_names:
            # Convert the sheet to CSV format
            all_dfs[sheet] = excel_to_csv(file, sheet)
            
    return all_dfs, sheet_names

def filter_dataframe(df):
    """Create UI for filtering the dataframe."""
    filtered_df = df.copy()
    
    st.markdown("### Filter Data")
    st.write("Apply filters to your data:")
    
    # Get column lists by type
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_object_columns(df)
    
    # Numeric column filters
    if numeric_columns:
        st.write("#### Numeric Filters")
        
        # Allow user to select which numeric columns to filter
        selected_numeric = st.multiselect(
            "Select numeric columns to filter",
            options=numeric_columns
        )
        
        for column in selected_numeric:
            # Safely convert to float and handle NaN/None values
            col_data = filtered_df[column].dropna()
            
            if col_data.empty:
                st.warning(f"Column '{column}' contains only missing values")
                continue
                
            try:
                min_value = float(col_data.min())
                max_value = float(col_data.max())
                
                # Handle if min and max are the same
                if min_value == max_value:
                    st.write(f"All values in {column} are {min_value}")
                    continue
                    
                # Handle if min or max are invalid
                if pd.isna(min_value) or pd.isna(max_value) or np.isinf(min_value) or np.isinf(max_value):
                    st.warning(f"Cannot create slider for {column} due to invalid values")
                    continue
                
                # Create a slider for filtering with proper error handling
                # Add a small buffer to max_value to ensure it's included in the range
                step = (max_value - min_value) / 100 if max_value > min_value else 0.1
                
                try:
                    filter_range = st.slider(
                        f"Filter range for {column}",
                        min_value=min_value,
                        max_value=max_value,
                        value=(min_value, max_value),
                        step=step
                    )
                    
                    # Apply the filter
                    filtered_df = filtered_df[(filtered_df[column].fillna(min_value - 1) >= filter_range[0]) & 
                                             (filtered_df[column].fillna(max_value + 1) <= filter_range[1])]
                except Exception as e:
                    st.error(f"Error creating slider for {column}: {str(e)}")
                    continue
            except Exception as e:
                st.warning(f"Could not process column '{column}': {str(e)}")
                continue
    
    # Categorical column filters
    if categorical_columns:
        st.write("#### Categorical Filters")
        
        # Allow user to select which categorical columns to filter
        selected_categorical = st.multiselect(
            "Select categorical columns to filter",
            options=categorical_columns
        )
        
        for column in selected_categorical:
            unique_values = filtered_df[column].dropna().unique().tolist()
            
            # Create a multiselect for filtering
            selected_values = st.multiselect(
                f"Select values for {column}",
                options=unique_values,
                default=unique_values
            )
            
            # Apply the filter
            if selected_values:
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
    
    # Show filtering results summary
    st.write(f"Original data: {df.shape[0]} rows")
    st.write(f"Filtered data: {filtered_df.shape[0]} rows")
    
    return filtered_df

def sort_dataframe(df):
    """Create UI for sorting the dataframe."""
    sorted_df = df.copy()
    
    st.markdown("### Sort Data")
    
    # Get all columns
    all_columns = df.columns.tolist()
    
    # Allow user to select column for sorting
    sort_column = st.selectbox(
        "Select column to sort by",
        options=all_columns
    )
    
    # Sort direction
    sort_direction = st.radio(
        "Sort direction",
        options=["Ascending", "Descending"]
    )
    
    # Apply the sorting
    ascending = sort_direction == "Ascending"
    
    try:
        sorted_df = sorted_df.sort_values(by=sort_column, ascending=ascending)
        st.success(f"Data sorted by {sort_column} ({sort_direction.lower()})")
    except Exception as e:
        st.error(f"Error sorting data: {str(e)}")
    
    return sorted_df
