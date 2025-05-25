import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import zipfile
import tempfile
import time
from data_operations import get_sheet_names, excel_to_csv, process_excel_file

def run_batch_processing():
    """Create UI for batch processing multiple Excel files."""
    st.markdown("## Batch Processing")
    st.write("Upload and process multiple Excel files at once.")
    
    # Initialize sample_files at the beginning
    sample_files = []
    
    # Add option to use sample files
    use_samples = st.checkbox(
        "Use optimized sample files", 
        value=False,
        help="Use smaller, optimized sample files (100 rows each) for faster processing"
    )
    
    if use_samples:
        # Check which sample files exist
        available_samples = []
        if os.path.exists("raceform_2023_2025_sample.xlsx"):
            available_samples.append("raceform_2023_2025_sample.xlsx")
        if os.path.exists("realistic_racing_data_sample.xlsx"):
            available_samples.append("realistic_racing_data_sample.xlsx")
            
        if available_samples:
            selected_samples = st.multiselect(
                "Select sample files to process:",
                options=available_samples,
                default=available_samples,
                help="These files contain 100 rows each for faster processing"
            )
            
            # Read the selected sample files
            for sample_file in selected_samples:
                try:
                    # Open file and create BytesIO object for compatibility with other code
                    with open(sample_file, "rb") as f:
                        file_data = io.BytesIO(f.read())
                        file_data.name = sample_file
                        sample_files.append(file_data)
                except Exception as e:
                    st.error(f"Error loading sample file {sample_file}: {str(e)}")
            
            if sample_files:
                st.success(f"Loaded {len(sample_files)} sample files for processing")
            else:
                st.warning("No sample files selected")
        else:
            st.warning("No sample files found in the current directory")
    
    # Initialize sample_files
    sample_files = []
    
    # File uploader for multiple files - only show if not using samples
    if not use_samples or not sample_files:
        uploaded_files = st.file_uploader(
            "Upload Excel files",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            help="Select multiple Excel files to process them in batch."
        )
    else:
        uploaded_files = sample_files
    
    if not uploaded_files:
        st.info("Please upload one or more Excel files or select sample files to begin batch processing.")
        return
    
    # Display the list of uploaded files
    st.write(f"**{len(uploaded_files)} files uploaded:**")
    for i, file in enumerate(uploaded_files):
        st.write(f"{i+1}. {file.name} ({file.size/1024:.1f} KB)")
    
    # Processing options
    st.subheader("Processing Options")
    
    process_option = st.radio(
        "Select processing mode:",
        options=["Preview Only", "Convert to CSV", "Full Analysis"],
        index=1,
        help="Preview Only: Just show file contents. Convert to CSV: Create optimized CSV files. Full Analysis: Generate statistics and visualizations."
    )
    
    # Create a form to collect additional options
    with st.form("batch_processing_form"):
        # Option to include file name in output
        include_filename = st.checkbox(
            "Add filename column",
            value=True,
            help="Add a column with the source filename to each processed file."
        )
        
        # Option to merge all sheets into one file
        merge_sheets = st.checkbox(
            "Merge all sheets from each file",
            value=False,
            help="Combine all sheets from each file into a single dataset."
        )
        
        # Option to merge all files into one dataset
        merge_all = st.checkbox(
            "Merge all files into one dataset",
            value=False,
            help="Combine all data from all files into a single dataset. Only works if the files have compatible structures."
        )
        
        # Process button
        submitted = st.form_submit_button("Process Files", type="primary")
    
    # Process the files when the button is clicked
    if submitted:
        process_batch_files(
            uploaded_files,
            process_option,
            include_filename,
            merge_sheets,
            merge_all
        )

def process_batch_files(files, process_option, include_filename, merge_sheets, merge_all):
    """
    Process multiple Excel files based on selected options.
    
    Args:
        files: List of uploaded files
        process_option: Selected processing mode
        include_filename: Whether to add filename column
        merge_sheets: Whether to merge sheets in each file
        merge_all: Whether to merge all files
    """
    # Initialize progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Initialize containers for processed data
    all_dfs = {}
    merged_df = None
    
    # Create a temporary directory to store files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process each file
        for i, file in enumerate(files):
            progress = (i / len(files))
            progress_bar.progress(progress)
            progress_text.text(f"Processing file {i+1} of {len(files)}: {file.name}")
            
            try:
                # Get all sheet names
                sheet_names = get_sheet_names(file)
                file_dfs = {}
                
                # Process each sheet
                for sheet in sheet_names:
                    # Convert Excel to CSV
                    df = excel_to_csv(file, sheet)
                    
                    # Add filename column if requested
                    if include_filename:
                        df['source_file'] = file.name
                        
                    # Store the dataframe
                    file_dfs[sheet] = df
                
                # Merge sheets if requested
                if merge_sheets and len(file_dfs) > 1:
                    merged_file_df = pd.concat(
                        [df.assign(sheet_name=sheet) for sheet, df in file_dfs.items()],
                        ignore_index=True
                    )
                    file_dfs = {f"{file.name}_merged": merged_file_df}
                
                # Store in the main dictionary
                all_dfs.update(file_dfs)
                
                # Merge all if requested
                if merge_all:
                    for df in file_dfs.values():
                        if merged_df is None:
                            merged_df = df.copy()
                        else:
                            # Try to merge, assuming compatible structures
                            try:
                                merged_df = pd.concat([merged_df, df], ignore_index=True)
                            except Exception as e:
                                st.warning(f"Could not merge {file.name}: {str(e)}")
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        progress_text.text("Processing complete!")
        
        # Show results based on the selected option
        if process_option == "Preview Only":
            display_preview_results(all_dfs)
        elif process_option == "Convert to CSV":
            download_csv_results(all_dfs, merged_df if merge_all else None)
        else:  # Full Analysis
            analyze_batch_results(all_dfs, merged_df if merge_all else None)

def display_preview_results(all_dfs):
    """
    Display preview of processed files.
    
    Args:
        all_dfs: Dictionary of dataframes keyed by sheet/file name
    """
    st.subheader("Preview Results")
    
    # Create tabs for each result
    if len(all_dfs) > 0:
        tabs = st.tabs(list(all_dfs.keys()))
        
        for i, (name, df) in enumerate(all_dfs.items()):
            with tabs[i]:
                st.write(f"**{name}** | Rows: {df.shape[0]} | Columns: {df.shape[1]}")
                st.dataframe(df.head(10))
                
                # Display column info
                st.write("**Column Types:**")
                col_types = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isna().sum()
                })
                st.dataframe(col_types)
    else:
        st.warning("No data was processed successfully.")

def download_csv_results(all_dfs, merged_df=None):
    """
    Provide download links for CSV files.
    
    Args:
        all_dfs: Dictionary of dataframes keyed by sheet/file name
        merged_df: Merged dataframe containing all data (optional)
    """
    st.subheader("Download CSV Files")
    
    # Function to create download link
    def get_csv_download_link(df, filename):
        csv = df.to_csv(index=False)
        b64 = io.StringIO(csv)
        return b64
    
    # Create a zip file containing all CSVs
    if len(all_dfs) > 0:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zipf:
                for name, df in all_dfs.items():
                    # Create a CSV in memory
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    
                    # Write to zip with a sanitized filename
                    safe_name = "".join([c if c.isalnum() or c in "._- " else "_" for c in name])
                    zipf.writestr(f"{safe_name}.csv", csv_buffer.getvalue())
            
            # Provide download link for the zip file
            with open(tmp.name, "rb") as f:
                st.download_button(
                    label="Download All CSV Files (ZIP)",
                    data=f,
                    file_name="batch_processed_files.zip",
                    mime="application/zip"
                )
        
        # Clean up temp file
        os.unlink(tmp.name)
    
    # Provide individual download links
    st.write("**Download Individual Files:**")
    for name, df in all_dfs.items():
        safe_name = "".join([c if c.isalnum() or c in "._- " else "_" for c in name])
        csv = df.to_csv(index=False)
        st.download_button(
            label=f"Download {name}",
            data=csv,
            file_name=f"{safe_name}.csv",
            mime="text/csv",
            key=f"download_{name}"
        )
    
    # Provide download for merged file if available
    if merged_df is not None:
        st.write("**Download Merged Dataset:**")
        merged_csv = merged_df.to_csv(index=False)
        st.download_button(
            label="Download Merged Dataset",
            data=merged_csv,
            file_name="merged_dataset.csv",
            mime="text/csv",
            key="download_merged"
        )

def analyze_batch_results(all_dfs, merged_df=None):
    """
    Analyze batch processing results.
    
    Args:
        all_dfs: Dictionary of dataframes keyed by sheet/file name
        merged_df: Merged dataframe containing all data (optional)
    """
    st.subheader("Batch Analysis Results")
    
    # Provide download links
    download_csv_results(all_dfs, merged_df)
    
    # Analysis options
    analysis_tabs = st.tabs(["Summary Stats", "Comparison Analysis", "Data Quality", "AI Insights"])
    
    with analysis_tabs[0]:  # Summary Stats
        # Display summary statistics
        st.subheader("Summary Statistics")
        
        # Create summary table
        summary_data = []
        for name, df in all_dfs.items():
            summary_data.append({
                'Name': name,
                'Rows': df.shape[0],
                'Columns': df.shape[1],
                'Missing Values': df.isna().sum().sum(),
                'Missing %': (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100).round(2),
                'Numeric Columns': len(df.select_dtypes(include=['number']).columns),
                'Categorical Columns': len(df.select_dtypes(include=['object', 'category']).columns),
                'Memory Usage (KB)': df.memory_usage(deep=True).sum() / 1024
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
        
        # Show overall stats if merged
        if merged_df is not None:
            st.subheader("Merged Dataset Statistics")
            st.write(f"Total Rows: {merged_df.shape[0]}")
            st.write(f"Total Columns: {merged_df.shape[1]}")
            
            # Display head of merged data
            st.write("**Preview of Merged Data:**")
            st.dataframe(merged_df.head(10))
            
            # Basic statistics for numeric columns
            numeric_cols = merged_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.write("**Numeric Column Statistics:**")
                st.dataframe(merged_df[numeric_cols].describe())
        
        # Individual file analysis
        st.subheader("Individual File Analysis")
        
        # Create tabs for each file
        if len(all_dfs) > 0:
            file_tabs = st.tabs(list(all_dfs.keys()))
            
            for i, (name, df) in enumerate(all_dfs.items()):
                with file_tabs[i]:
                    # Basic file info
                    st.write(f"**{name}** | Rows: {df.shape[0]} | Columns: {df.shape[1]}")
                    
                    # Display head
                    st.write("**Data Preview:**")
                    st.dataframe(df.head(5))
                    
                    # Column information
                    st.write("**Column Information:**")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null Count': df.count(),
                        'Null %': (df.isna().sum() / len(df) * 100).round(2),
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_info)
                    
                    # Basic statistics for numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        st.write("**Numeric Column Statistics:**")
                        st.dataframe(df[numeric_cols].describe())
    
    with analysis_tabs[1]:  # Comparison Analysis
        st.subheader("Compare Files")
        
        if len(all_dfs) < 2:
            st.warning("Need at least two datasets to perform comparison analysis.")
        else:
            # Allow user to select files to compare
            file_names = list(all_dfs.keys())
            col1, col2 = st.columns(2)
            
            with col1:
                file1 = st.selectbox("Select first dataset:", file_names, key="file1_compare")
            
            with col2:
                file2 = st.selectbox("Select second dataset:", 
                                    [f for f in file_names if f != file1], 
                                    key="file2_compare")
            
            if file1 and file2:
                df1 = all_dfs[file1]
                df2 = all_dfs[file2]
                
                # Basic comparison
                st.subheader("Basic Comparison")
                comparison_data = {
                    "Metric": ["Rows", "Columns", "Missing Values", "Memory Usage (KB)"],
                    file1: [df1.shape[0], df1.shape[1], df1.isna().sum().sum(), df1.memory_usage(deep=True).sum()/1024],
                    file2: [df2.shape[0], df2.shape[1], df2.isna().sum().sum(), df2.memory_usage(deep=True).sum()/1024],
                    "Difference": [
                        df1.shape[0] - df2.shape[0],
                        df1.shape[1] - df2.shape[1],
                        df1.isna().sum().sum() - df2.isna().sum().sum(),
                        (df1.memory_usage(deep=True).sum() - df2.memory_usage(deep=True).sum())/1024
                    ]
                }
                st.dataframe(pd.DataFrame(comparison_data))
                
                # Column comparison
                st.subheader("Column Comparison")
                common_cols = set(df1.columns).intersection(set(df2.columns))
                only_in_df1 = set(df1.columns) - set(df2.columns)
                only_in_df2 = set(df2.columns) - set(df1.columns)
                
                st.write(f"**Common columns:** {len(common_cols)}")
                st.write(f"**Columns only in {file1}:** {len(only_in_df1)}")
                st.write(f"**Columns only in {file2}:** {len(only_in_df2)}")
                
                # If there are common numeric columns, show correlation comparison
                common_numeric_cols = [col for col in common_cols 
                                      if col in df1.select_dtypes(include=['number']).columns
                                      and col in df2.select_dtypes(include=['number']).columns]
                
                if common_numeric_cols:
                    st.subheader("Statistical Comparison of Common Numeric Columns")
                    
                    compare_stats = []
                    for col in common_numeric_cols:
                        compare_stats.append({
                            'Column': col,
                            f'{file1} Mean': df1[col].mean(),
                            f'{file2} Mean': df2[col].mean(),
                            'Mean Diff %': ((df1[col].mean() - df2[col].mean()) / df1[col].mean() * 100 
                                            if df1[col].mean() != 0 else 0),
                            f'{file1} Std': df1[col].std(),
                            f'{file2} Std': df2[col].std(),
                            'Min Diff': df1[col].min() - df2[col].min(),
                            'Max Diff': df1[col].max() - df2[col].max()
                        })
                    
                    st.dataframe(pd.DataFrame(compare_stats))
    
    with analysis_tabs[2]:  # Data Quality
        st.subheader("Data Quality Assessment")
        
        # Select file to analyze
        selected_file = st.selectbox("Select dataset to analyze:", 
                                    list(all_dfs.keys()),
                                    key="file_quality")
        
        if selected_file:
            df = all_dfs[selected_file]
            
            # Missing values analysis
            st.write("### Missing Values Analysis")
            
            # Calculate missing values per column
            missing_vals = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isna().sum(),
                'Missing %': (df.isna().sum() / len(df) * 100).round(2)
            }).sort_values('Missing %', ascending=False)
            
            # Filter to show only columns with missing values
            missing_vals = missing_vals[missing_vals['Missing Count'] > 0]
            
            if len(missing_vals) > 0:
                st.dataframe(missing_vals)
                
                # Create a bar chart of missing values
                if not missing_vals.empty:
                    st.write("### Missing Values Chart")
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    fig, ax = plt.subplots(figsize=(10, min(10, max(4, len(missing_vals)/2))))
                    # Use matplotlib directly to avoid type issues
                    x_values = missing_vals['Missing %'].tolist()
                    y_positions = list(range(len(missing_vals)))
                    y_labels = missing_vals['Column'].tolist()
                    ax.barh(y_positions, x_values)
                    ax.set_yticks(y_positions)
                    ax.set_yticklabels(y_labels)
                    ax.set_title('Missing Values by Column')
                    ax.set_xlabel('Missing %')
                    ax.set_ylabel('Column')
                    st.pyplot(fig)
            else:
                st.success("No missing values found in this dataset!")
            
            # Duplicate rows analysis
            st.write("### Duplicate Rows Analysis")
            dup_count = df.duplicated().sum()
            st.write(f"Number of duplicate rows: {dup_count} ({dup_count/len(df)*100:.2f}% of data)")
            
            # Outlier detection for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.write("### Outlier Detection")
                st.write("Showing potential outliers using the IQR method (values outside 1.5 × IQR).")
                
                outlier_counts = {}
                for col in numeric_cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
                    outlier_counts[col] = {'count': outlier_count, 'percentage': outlier_count/len(df)*100}
                
                outlier_df = pd.DataFrame({
                    'Column': list(outlier_counts.keys()),
                    'Outlier Count': [v['count'] for v in outlier_counts.values()],
                    'Outlier %': [v['percentage'] for v in outlier_counts.values()]
                }).sort_values('Outlier %', ascending=False)
                
                st.dataframe(outlier_df)
    
    with analysis_tabs[3]:  # AI Insights
        st.subheader("AI-Powered Batch Insights")
        st.write("Use AI to analyze and compare your datasets for deeper insights.")
        
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("OpenAI API key is not configured. AI insights are not available.")
        else:
            # Option to select files to analyze
            if merged_df is not None:
                st.write("### Analyze Merged Dataset")
                if st.button("Generate AI Insights for Merged Data"):
                    try:
                        from ai_analysis import analyze_dataframe
                        with st.spinner("Analyzing merged dataset with AI..."):
                            insights = analyze_dataframe(merged_df, "This is a merged dataset from multiple Excel files.")
                            
                            # Display the results
                            st.markdown("#### Merged Data Summary")
                            for item in insights.get("summary", []):
                                st.write(f"• {item}")
                            
                            st.markdown("#### Key Insights")
                            for item in insights.get("insights", []):
                                st.write(f"• {item}")
                            
                            st.markdown("#### Recommendations")
                            for item in insights.get("recommendations", []):
                                st.write(f"• {item}")
                    except Exception as e:
                        st.error(f"Error analyzing data with AI: {str(e)}")
            
            st.write("### Compare Datasets with AI")
            if len(all_dfs) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    ai_file1 = st.selectbox("Select first dataset:", 
                                          list(all_dfs.keys()), 
                                          key="ai_file1")
                
                with col2:
                    ai_file2 = st.selectbox("Select second dataset:", 
                                          [f for f in all_dfs.keys() if f != ai_file1], 
                                          key="ai_file2")
                
                if st.button("Generate Comparison Insights"):
                    try:
                        from ai_analysis import client
                        import json
                        
                        df1 = all_dfs[ai_file1]
                        df2 = all_dfs[ai_file2]
                        
                        with st.spinner("Comparing datasets with AI..."):
                            # Get basic statistics
                            common_cols = set(df1.columns).intersection(set(df2.columns))
                            only_in_df1 = set(df1.columns) - set(df2.columns)
                            only_in_df2 = set(df2.columns) - set(df1.columns)
                            
                            # Create a sample of the dataframes to avoid token limits
                            sample_df1 = df1.sample(min(50, len(df1)))
                            sample_df2 = df2.sample(min(50, len(df2)))
                            
                            # Create a prompt for the AI
                            prompt = f"""
                            I have two datasets that I want to compare:
                            
                            Dataset 1: {ai_file1}
                            - Rows: {df1.shape[0]}
                            - Columns: {df1.shape[1]}
                            - First 5 columns: {', '.join(df1.columns[:5].tolist())}
                            
                            Dataset 2: {ai_file2}
                            - Rows: {df2.shape[0]}
                            - Columns: {df2.shape[1]}
                            - First 5 columns: {', '.join(df2.columns[:5].tolist())}
                            
                            Common columns: {len(common_cols)} columns
                            Columns only in Dataset 1: {len(only_in_df1)} columns
                            Columns only in Dataset 2: {len(only_in_df2)} columns
                            
                            Sample of Dataset 1:
                            {sample_df1.head(5).to_string()}
                            
                            Sample of Dataset 2:
                            {sample_df2.head(5).to_string()}
                            
                            Please analyze these datasets and provide:
                            1. A comparison of their structures and content
                            2. Key differences between the two datasets
                            3. Recommendations for working with these datasets together
                            
                            Format your response as a JSON object with the keys: 'comparison', 'differences', 'recommendations'.
                            Each value should be an array of strings.
                            """
                            
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "You are a data analysis expert helping to compare Excel datasets."},
                                    {"role": "user", "content": prompt}
                                ],
                                response_format={"type": "json_object"}
                            )
                            
                            content = response.choices[0].message.content
                            if content:
                                comparison = json.loads(content)
                                
                                st.markdown("#### Structural Comparison")
                                for item in comparison.get("comparison", []):
                                    st.write(f"• {item}")
                                
                                st.markdown("#### Key Differences")
                                for item in comparison.get("differences", []):
                                    st.write(f"• {item}")
                                
                                st.markdown("#### Recommendations")
                                for item in comparison.get("recommendations", []):
                                    st.write(f"• {item}")
                            else:
                                st.error("Failed to generate comparison. Please try again.")
                    except Exception as e:
                        st.error(f"Error comparing datasets with AI: {str(e)}")
            else:
                st.info("Need at least two datasets to generate comparison insights.")