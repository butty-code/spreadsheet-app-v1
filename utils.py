import pandas as pd
import numpy as np
import streamlit as st
import re

def get_numeric_columns(df):
    """Return a list of numeric columns in the dataframe."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return numeric_cols

def get_object_columns(df):
    """Return a list of categorical (object) columns in the dataframe."""
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Also check for datetime columns since they can be useful as categories
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    return object_cols + datetime_cols

def check_date_columns(df):
    """Check for potential date columns that are stored as objects."""
    date_patterns = [
        # ISO format
        r'^\d{4}-\d{2}-\d{2}$',
        # US format
        r'^\d{2}/\d{2}/\d{4}$',
        # European format
        r'^\d{2}-\d{2}-\d{4}$',
        r'^\d{2}\.\d{2}\.\d{4}$',
    ]
    
    potential_date_cols = []
    
    for col in df.select_dtypes(include=['object']).columns:
        # Try a small sample to check if it matches date patterns
        sample = df[col].dropna().head(10)
        if len(sample) > 0 and all(sample.astype(str).str.match('|'.join(date_patterns))):
            potential_date_cols.append(col)
    
    return potential_date_cols

def check_numeric_columns(df):
    """Check for potential numeric columns that are stored as objects."""
    numeric_pattern = r'^-?\d+(\.\d+)?$'
    
    potential_numeric_cols = []
    
    for col in df.select_dtypes(include=['object']).columns:
        # Try a small sample to check if it matches numeric patterns
        sample = df[col].dropna().head(10)
        if len(sample) > 0 and all(sample.astype(str).str.match(numeric_pattern)):
            potential_numeric_cols.append(col)
    
    return potential_numeric_cols

def convert_data_types(df):
    """Try to convert object columns to more appropriate types."""
    # Make a copy to avoid modifying the original
    df_converted = df.copy()
    
    # Try to convert strings to numeric
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Try converting to numeric
            numeric_conversion = pd.to_numeric(df[col])
            df_converted[col] = numeric_conversion
        except:
            # Try converting to datetime
            try:
                date_conversion = pd.to_datetime(df[col])
                df_converted[col] = date_conversion
            except:
                # Keep as object if conversion fails
                pass
    
    return df_converted

def clean_data(df):
    """Provides interactive UI for cleaning data."""
    st.write("### Data Cleaning")
    
    if df.empty:
        st.warning("No data to clean.")
        return df
    
    cleaned_df = df.copy()
    
    # Create tabs for different cleaning operations
    clean_tab1, clean_tab2, clean_tab3, clean_tab4, clean_tab5 = st.tabs([
        "Data Types", "Missing Values", "Duplicate Rows", "Text Cleaning", "Outliers"
    ])
    
    # Tab 1: Data Types
    with clean_tab1:
        st.write("#### Convert Data Types")
        st.write("Fix columns that have the wrong data type:")
        
        # Auto-detect potential date columns
        date_columns = check_date_columns(cleaned_df)
        if date_columns:
            st.write("**Potential date columns detected:**")
            selected_date_cols = st.multiselect(
                "Select columns to convert to dates:",
                options=date_columns,
                default=date_columns
            )
            
            if selected_date_cols and st.button("Convert to Dates"):
                for col in selected_date_cols:
                    try:
                        cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                        st.success(f"Converted {col} to date format")
                    except Exception as e:
                        st.error(f"Error converting {col}: {str(e)}")
        
        # Auto-detect potential numeric columns
        numeric_columns = check_numeric_columns(cleaned_df)
        if numeric_columns:
            st.write("**Potential numeric columns detected:**")
            selected_numeric_cols = st.multiselect(
                "Select columns to convert to numbers:",
                options=numeric_columns,
                default=numeric_columns
            )
            
            if selected_numeric_cols and st.button("Convert to Numbers"):
                for col in selected_numeric_cols:
                    try:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col])
                        st.success(f"Converted {col} to numeric format")
                    except Exception as e:
                        st.error(f"Error converting {col}: {str(e)}")
        
        # Manual data type conversion
        st.write("**Manual type conversion:**")
        all_columns = cleaned_df.columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            selected_column = st.selectbox("Select column to convert:", ["None"] + all_columns)
        
        if selected_column != "None":
            with col2:
                target_type = st.selectbox(
                    "Convert to:",
                    ["Number", "Text", "Date", "Category"]
                )
            
            if st.button("Apply Conversion"):
                try:
                    if target_type == "Number":
                        cleaned_df[selected_column] = pd.to_numeric(cleaned_df[selected_column])
                        st.success(f"Converted {selected_column} to numeric")
                    elif target_type == "Text":
                        cleaned_df[selected_column] = cleaned_df[selected_column].astype(str)
                        st.success(f"Converted {selected_column} to text")
                    elif target_type == "Date":
                        cleaned_df[selected_column] = pd.to_datetime(cleaned_df[selected_column])
                        st.success(f"Converted {selected_column} to date")
                    elif target_type == "Category":
                        cleaned_df[selected_column] = cleaned_df[selected_column].astype('category')
                        st.success(f"Converted {selected_column} to category")
                except Exception as e:
                    st.error(f"Error converting {selected_column}: {str(e)}")
    
    # Tab 2: Missing Values
    with clean_tab2:
        st.write("#### Handle Missing Values")
        
        # First, show missing value counts
        missing_counts = cleaned_df.isna().sum()
        missing_columns = missing_counts[missing_counts > 0]
        
        if len(missing_columns) == 0:
            st.success("No missing values found in the data!")
        else:
            st.write("Columns with missing values:")
            missing_df = pd.DataFrame({
                'Column': missing_columns.index,
                'Missing Values': missing_columns.values,
                'Percentage': (missing_columns.values / len(cleaned_df) * 100).round(2)
            })
            st.dataframe(missing_df)
            
            st.write("**Select how to handle missing values:**")
            
            selected_cols_missing = st.multiselect(
                "Select columns to fix:",
                options=missing_columns.index.tolist(),
                default=missing_columns.index.tolist()
            )
            
            if selected_cols_missing:
                fill_method = st.selectbox(
                    "Choose fill method:",
                    ["Drop rows with missing values", 
                     "Fill with mean", 
                     "Fill with median",
                     "Fill with mode (most common value)",
                     "Fill with zero",
                     "Fill with custom value"]
                )
                
                if st.button("Apply Missing Value Fix"):
                    if fill_method == "Drop rows with missing values":
                        original_len = len(cleaned_df)
                        cleaned_df = cleaned_df.dropna(subset=selected_cols_missing)
                        st.success(f"Dropped {original_len - len(cleaned_df)} rows with missing values")
                    
                    elif fill_method == "Fill with mean":
                        for col in selected_cols_missing:
                            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                                st.success(f"Filled missing values in {col} with mean")
                            else:
                                st.warning(f"Cannot fill with mean for non-numeric column: {col}")
                    
                    elif fill_method == "Fill with median":
                        for col in selected_cols_missing:
                            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                                st.success(f"Filled missing values in {col} with median")
                            else:
                                st.warning(f"Cannot fill with median for non-numeric column: {col}")
                    
                    elif fill_method == "Fill with mode (most common value)":
                        for col in selected_cols_missing:
                            mode_value = cleaned_df[col].mode()[0]
                            cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                            st.success(f"Filled missing values in {col} with mode: {mode_value}")
                    
                    elif fill_method == "Fill with zero":
                        for col in selected_cols_missing:
                            cleaned_df[col] = cleaned_df[col].fillna(0)
                            st.success(f"Filled missing values in {col} with zero")
                    
                    elif fill_method == "Fill with custom value":
                        custom_value = st.text_input("Enter custom value to fill with:")
                        if custom_value:
                            for col in selected_cols_missing:
                                cleaned_df[col] = cleaned_df[col].fillna(custom_value)
                            st.success(f"Filled missing values with '{custom_value}'")
    
    # Tab 3: Duplicate Rows
    with clean_tab3:
        st.write("#### Find and Remove Duplicates")
        
        # Check for duplicate rows
        duplicate_rows = cleaned_df.duplicated().sum()
        
        if duplicate_rows == 0:
            st.success("No duplicate rows found in the data!")
        else:
            st.warning(f"Found {duplicate_rows} duplicate rows in the data")
            
            # Provide options for handling duplicates
            dup_action = st.radio(
                "How to handle duplicates:",
                ["Show duplicate rows", "Remove duplicate rows (keep first)", "Remove duplicate rows (keep last)"]
            )
            
            if dup_action == "Show duplicate rows":
                st.write("Duplicate rows:")
                st.dataframe(cleaned_df[cleaned_df.duplicated(keep=False)])
            
            elif dup_action == "Remove duplicate rows (keep first)":
                if st.button("Remove Duplicates (Keep First)"):
                    original_len = len(cleaned_df)
                    cleaned_df = cleaned_df.drop_duplicates(keep='first')
                    st.success(f"Removed {original_len - len(cleaned_df)} duplicate rows, keeping first occurrence")
            
            elif dup_action == "Remove duplicate rows (keep last)":
                if st.button("Remove Duplicates (Keep Last)"):
                    original_len = len(cleaned_df)
                    cleaned_df = cleaned_df.drop_duplicates(keep='last')
                    st.success(f"Removed {original_len - len(cleaned_df)} duplicate rows, keeping last occurrence")
            
            # Option to find duplicates based on specific columns
            st.write("**Find duplicates based on specific columns:**")
            dup_columns = st.multiselect(
                "Select columns to check for duplicates:",
                options=cleaned_df.columns.tolist()
            )
            
            if dup_columns and st.button("Check Selected Columns for Duplicates"):
                column_duplicates = cleaned_df.duplicated(subset=dup_columns).sum()
                if column_duplicates == 0:
                    st.success(f"No duplicates found based on selected columns")
                else:
                    st.warning(f"Found {column_duplicates} duplicate rows based on selected columns")
                    st.dataframe(cleaned_df[cleaned_df.duplicated(subset=dup_columns, keep=False)])
                    
                    if st.button("Remove These Duplicates"):
                        original_len = len(cleaned_df)
                        cleaned_df = cleaned_df.drop_duplicates(subset=dup_columns, keep='first')
                        st.success(f"Removed {original_len - len(cleaned_df)} duplicate rows based on selected columns")
    
    # Tab 4: Text Cleaning
    with clean_tab4:
        st.write("#### Clean Text Data")
        
        # Get text columns
        text_columns = cleaned_df.select_dtypes(include=['object']).columns.tolist()
        
        if not text_columns:
            st.info("No text columns found in the data")
        else:
            st.write("Select text cleaning operations:")
            
            selected_text_cols = st.multiselect(
                "Select text columns to clean:",
                options=text_columns
            )
            
            if selected_text_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    remove_extra_spaces = st.checkbox("Remove extra spaces")
                    convert_case = st.selectbox(
                        "Convert case:",
                        ["No change", "UPPERCASE", "lowercase", "Title Case"]
                    )
                
                with col2:
                    remove_special_chars = st.checkbox("Remove special characters")
                    strip_spaces = st.checkbox("Strip leading/trailing spaces", value=True)
                
                if st.button("Apply Text Cleaning"):
                    for col in selected_text_cols:
                        # Make sure we're working with strings
                        cleaned_df[col] = cleaned_df[col].astype(str)
                        
                        # Apply selected transformations
                        if strip_spaces:
                            cleaned_df[col] = cleaned_df[col].str.strip()
                            st.write(f"Stripped spaces from {col}")
                        
                        if remove_extra_spaces:
                            cleaned_df[col] = cleaned_df[col].str.replace(r'\s+', ' ', regex=True)
                            st.write(f"Removed extra spaces from {col}")
                        
                        if remove_special_chars:
                            cleaned_df[col] = cleaned_df[col].str.replace(r'[^\w\s]', '', regex=True)
                            st.write(f"Removed special characters from {col}")
                        
                        if convert_case != "No change":
                            if convert_case == "UPPERCASE":
                                cleaned_df[col] = cleaned_df[col].str.upper()
                            elif convert_case == "lowercase":
                                cleaned_df[col] = cleaned_df[col].str.lower()
                            elif convert_case == "Title Case":
                                cleaned_df[col] = cleaned_df[col].str.title()
                            st.write(f"Converted {col} to {convert_case}")
                    
                    st.success("Text cleaning applied!")
                    
                # Custom text replacement
                st.write("**Custom find and replace:**")
                col1, col2 = st.columns(2)
                with col1:
                    find_text = st.text_input("Find:")
                with col2:
                    replace_text = st.text_input("Replace with:")
                
                if find_text and st.button("Apply Custom Replacement"):
                    for col in selected_text_cols:
                        cleaned_df[col] = cleaned_df[col].astype(str).str.replace(find_text, replace_text)
                    st.success(f"Replaced '{find_text}' with '{replace_text}'")
    
    # Tab 5: Outliers
    with clean_tab5:
        st.write("#### Detect and Handle Outliers")
        
        # Get numeric columns for outlier detection
        numeric_cols = get_numeric_columns(cleaned_df)
        
        if not numeric_cols:
            st.info("No numeric columns found for outlier detection")
        else:
            st.write("Select columns to check for outliers:")
            
            selected_outlier_cols = st.multiselect(
                "Select numeric columns:",
                options=numeric_cols
            )
            
            if selected_outlier_cols:
                outlier_method = st.selectbox(
                    "Outlier detection method:",
                    ["Z-Score (standard deviations)", "IQR (interquartile range)"]
                )
                
                if outlier_method == "Z-Score (standard deviations)":
                    threshold = st.slider("Z-Score threshold", 1.0, 5.0, 3.0, 0.1)
                    
                    if st.button("Detect Outliers (Z-Score)"):
                        outlier_counts = {}
                        
                        for col in selected_outlier_cols:
                            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                            outliers = (z_scores > threshold).sum()
                            outlier_counts[col] = outliers
                            
                            if outliers > 0:
                                st.write(f"Found {outliers} outliers in {col} (Z-Score > {threshold})")
                                
                                # Show outlier values
                                st.write("Sample of outlier values:")
                                outlier_indices = np.where(z_scores > threshold)[0]
                                st.dataframe(cleaned_df.loc[outlier_indices, [col]].head(10))
                        
                        if sum(outlier_counts.values()) == 0:
                            st.success("No outliers found with the specified threshold")
                        else:
                            # Options for handling outliers
                            outlier_action = st.radio(
                                "How to handle outliers:",
                                ["Remove outlier rows", "Cap outliers", "Replace with mean/median"]
                            )
                            
                            if outlier_action == "Remove outlier rows" and st.button("Remove Outlier Rows"):
                                original_len = len(cleaned_df)
                                for col in selected_outlier_cols:
                                    z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                                    cleaned_df = cleaned_df[z_scores <= threshold]
                                st.success(f"Removed {original_len - len(cleaned_df)} rows with outliers")
                            
                            elif outlier_action == "Cap outliers" and st.button("Cap Outliers"):
                                for col in selected_outlier_cols:
                                    mean = cleaned_df[col].mean()
                                    std = cleaned_df[col].std()
                                    lower_bound = mean - threshold * std
                                    upper_bound = mean + threshold * std
                                    
                                    # Cap the outliers
                                    cleaned_df[col] = np.clip(cleaned_df[col], lower_bound, upper_bound)
                                st.success(f"Capped outliers to within {threshold} standard deviations")
                            
                            elif outlier_action == "Replace with mean/median" and st.button("Replace Outliers"):
                                replacement = st.radio("Replace with:", ["Mean", "Median"])
                                
                                for col in selected_outlier_cols:
                                    z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                                    outlier_mask = z_scores > threshold
                                    
                                    if replacement == "Mean":
                                        replacement_value = cleaned_df[col].mean()
                                    else:  # Median
                                        replacement_value = cleaned_df[col].median()
                                    
                                    cleaned_df.loc[outlier_mask, col] = replacement_value
                                
                                st.success(f"Replaced outliers with {replacement.lower()}")
                
                elif outlier_method == "IQR (interquartile range)":
                    iqr_multiplier = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
                    
                    if st.button("Detect Outliers (IQR)"):
                        outlier_counts = {}
                        
                        for col in selected_outlier_cols:
                            Q1 = cleaned_df[col].quantile(0.25)
                            Q3 = cleaned_df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            
                            lower_bound = Q1 - iqr_multiplier * IQR
                            upper_bound = Q3 + iqr_multiplier * IQR
                            
                            outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
                            outlier_counts[col] = outliers
                            
                            if outliers > 0:
                                st.write(f"Found {outliers} outliers in {col} (outside {iqr_multiplier} × IQR)")
                                
                                # Show some outlier values
                                st.write("Sample of outlier values:")
                                outlier_mask = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
                                st.dataframe(cleaned_df.loc[outlier_mask, [col]].head(10))
                        
                        if sum(outlier_counts.values()) == 0:
                            st.success("No outliers found with the specified IQR range")
                        else:
                            # Options for handling outliers
                            outlier_action = st.radio(
                                "How to handle outliers:",
                                ["Remove outlier rows", "Cap outliers", "Replace with median"]
                            )
                            
                            if outlier_action == "Remove outlier rows" and st.button("Remove Outlier Rows"):
                                original_len = len(cleaned_df)
                                for col in selected_outlier_cols:
                                    Q1 = cleaned_df[col].quantile(0.25)
                                    Q3 = cleaned_df[col].quantile(0.75)
                                    IQR = Q3 - Q1
                                    
                                    lower_bound = Q1 - iqr_multiplier * IQR
                                    upper_bound = Q3 + iqr_multiplier * IQR
                                    
                                    cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
                                st.success(f"Removed {original_len - len(cleaned_df)} rows with outliers")
                            
                            elif outlier_action == "Cap outliers" and st.button("Cap Outliers"):
                                for col in selected_outlier_cols:
                                    Q1 = cleaned_df[col].quantile(0.25)
                                    Q3 = cleaned_df[col].quantile(0.75)
                                    IQR = Q3 - Q1
                                    
                                    lower_bound = Q1 - iqr_multiplier * IQR
                                    upper_bound = Q3 + iqr_multiplier * IQR
                                    
                                    # Cap the outliers
                                    cleaned_df[col] = np.clip(cleaned_df[col], lower_bound, upper_bound)
                                st.success(f"Capped outliers to within {iqr_multiplier} × IQR")
                            
                            elif outlier_action == "Replace with median" and st.button("Replace Outliers"):
                                for col in selected_outlier_cols:
                                    Q1 = cleaned_df[col].quantile(0.25)
                                    Q3 = cleaned_df[col].quantile(0.75)
                                    IQR = Q3 - Q1
                                    
                                    lower_bound = Q1 - iqr_multiplier * IQR
                                    upper_bound = Q3 + iqr_multiplier * IQR
                                    
                                    outlier_mask = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
                                    
                                    cleaned_df.loc[outlier_mask, col] = cleaned_df[col].median()
                                
                                st.success(f"Replaced outliers with median values")
    
    # Summary of changes
    st.write("### Data Cleaning Summary")
    original_shape = df.shape
    new_shape = cleaned_df.shape
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Rows", original_shape[0])
    with col2:
        st.metric("Cleaned Rows", new_shape[0], new_shape[0] - original_shape[0])
    with col3:
        st.metric("Columns", new_shape[1])
    
    return cleaned_df
