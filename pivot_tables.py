import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def run_pivot_table(df):
    """Create UI for interactive pivot table creation and analysis."""
    st.markdown("## Pivot Table Analysis")
    st.write("Create interactive pivot tables to summarize and analyze your data.")
    
    if df.empty:
        st.warning("Please upload an Excel file to create pivot tables.")
        return
    
    # Get column types for better selection options
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Basic pivot table controls
    st.markdown("### Pivot Table Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Row selections (categorical variables work best as rows)
        row_variables = st.multiselect(
            "Select Row Variables:",
            options=categorical_cols + numerical_cols,
            help="Variables to use as rows in the pivot table"
        )
        
        # Value selections (numerical variables work best as values)
        value_variables = st.multiselect(
            "Select Value Variables (metrics to calculate):",
            options=numerical_cols,
            help="Numerical variables to aggregate in the pivot table"
        )
    
    with col2:
        # Column selections (categorical variables work best as columns)
        column_variables = st.multiselect(
            "Select Column Variables:",
            options=categorical_cols + numerical_cols,
            help="Variables to use as columns in the pivot table"
        )
        
        # Aggregation function
        agg_function = st.selectbox(
            "Select Aggregation Function:",
            options=["mean", "sum", "count", "min", "max", "median", "std"],
            help="Function to aggregate the values"
        )
    
    # Advanced options
    with st.expander("Advanced Options"):
        fill_value = st.number_input(
            "Fill value for missing data:",
            value=0,
            help="Value to use for missing entries in the pivot table"
        )
        
        margins = st.checkbox(
            "Show Totals (Grand Total)",
            value=False,
            help="Add row and column totals"
        )
        
        normalize = st.selectbox(
            "Normalize Values:",
            options=["None", "all", "index", "columns"],
            help="Normalize values to show percentages instead of raw values"
        )
        
        normalize_map = {
            "None": None,
            "all": "all",
            "index": "index", 
            "columns": "columns"
        }
        
        # Display options
        precision = st.slider(
            "Decimal Precision:", 
            min_value=0, 
            max_value=6, 
            value=2,
            help="Number of decimal places to display"
        )
    
    # Generate pivot table when user has selected both rows and values
    if row_variables and value_variables:
        st.markdown("### Pivot Table Results")
        
        try:
            with st.spinner("Generating pivot table..."):
                # Map string aggregation function to actual function
                import numpy as np
                agg_functions = {
                    "mean": np.mean,
                    "sum": np.sum,
                    "count": len,
                    "min": np.min,
                    "max": np.max,
                    "median": np.median,
                    "std": np.std
                }
                
                # Pre-process data to handle potential issues
                processed_df = df.copy()
                
                # Check for numeric columns that might be stored as strings
                for col in value_variables:
                    if processed_df[col].dtype == 'object':
                        try:
                            # Try to convert to numeric, coercing errors to NaN
                            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                            st.info(f"Converted '{col}' to numeric format for analysis.")
                        except Exception:
                            st.warning(f"Column '{col}' could not be converted to numeric format.")
                
                # Check for complex data in row/column variables that might cause grouping issues
                for col in row_variables + (column_variables if column_variables else []):
                    try:
                        # If the column has nested lists/dicts or other complex types, convert to string
                        if any(isinstance(val, (dict, list)) for val in processed_df[col].dropna().head(100)):
                            processed_df[col] = processed_df[col].astype(str)
                            st.info(f"Converted complex values in '{col}' to string format.")
                    except Exception:
                        # If error in checking, convert to string anyway
                        try:
                            processed_df[col] = processed_df[col].astype(str)
                        except:
                            st.warning(f"Column '{col}' contains complex data and might cause issues.")
                
                # Create a dictionary to map value columns to aggregation functions
                aggfunc_dict = {col: agg_functions[agg_function] for col in value_variables}
                
                # Generate the pivot table with more error handling
                try:
                    pivot_df = pd.pivot_table(
                        processed_df,
                        values=value_variables,
                        index=row_variables,
                        columns=column_variables if column_variables else None,
                        aggfunc=aggfunc_dict,
                        fill_value=fill_value,
                        margins=margins,
                        margins_name="Grand Total"
                    )
                except ValueError as e:
                    if "not 1-dimensional" in str(e):
                        # Special handling for dimensionality errors
                        st.error(f"Error: {str(e)}. Try these solutions:")
                        st.info("1. Select different row/column variables - the current selection contains nested or complex data")
                        st.info("2. If using numerical columns as index/columns, try using categorical columns instead")
                        st.info("3. For advanced pivot tables, try pre-processing your data in Excel before uploading")
                        return
                    else:
                        # Re-raise other value errors
                        raise e
                
                # Apply normalization if selected
                if normalize_map[normalize]:
                    pivot_df = pivot_df.div(pivot_df.sum(axis=0 if normalize_map[normalize] == "index" else 1), 
                                           axis=1 if normalize_map[normalize] == "index" else 0)
                    # Format as percentages
                    pivot_df = pivot_df.applymap(lambda x: f"{x:.{precision}%}" if isinstance(x, (int, float)) else x)
                    st.dataframe(pivot_df)
                else:
                    # Round to specified precision
                    numeric_cols = pivot_df.select_dtypes(include=['float']).columns
                    pivot_df[numeric_cols] = pivot_df[numeric_cols].round(precision)
                    st.dataframe(pivot_df)
                
                # Add download button for pivot table
                csv = pivot_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Pivot Table as CSV",
                    data=csv,
                    file_name="pivot_table.csv",
                    mime="text/csv",
                )
                
                # Create pivot table visualizations if we have numerical results
                if (not normalize_map[normalize]) and value_variables:
                    st.markdown("### Pivot Table Visualization")
                    
                    # Prepare data for visualization
                    # Reset index to get row variables as columns for plotting
                    plot_df = pivot_df.reset_index()
                    
                    # Visualization type
                    viz_type = st.selectbox(
                        "Select Visualization Type:",
                        options=["Bar Chart", "Heatmap", "Line Chart", "Scatter Plot"],
                        help="Type of visualization to create from pivot table data"
                    )
                    
                    if viz_type == "Bar Chart":
                        # Select which value to visualize if multiple exist
                        if len(value_variables) > 1:
                            value_to_plot = st.selectbox(
                                "Select Value to Visualize:",
                                options=value_variables
                            )
                            metric_col = value_to_plot
                        else:
                            metric_col = value_variables[0]
                        
                        # Determine if we have a multi-level column index
                        if isinstance(pivot_df.columns, pd.MultiIndex):
                            # For multi-level columns, we need to flatten
                            pivot_df_flat = pivot_df.copy()
                            pivot_df_flat.columns = [' - '.join(col).strip() for col in 
                                                   pivot_df.columns.values]
                            plot_df = pivot_df_flat.reset_index()
                            
                            # Get the specific value column names after flattening
                            value_cols = [col for col in plot_df.columns 
                                         if metric_col in col and "Grand Total" not in col]
                            
                            # For each value column, create a bar chart
                            for i, val_col in enumerate(value_cols):
                                fig = px.bar(
                                    plot_df,
                                    x=row_variables[0],  # Use first row variable as x-axis
                                    y=val_col,
                                    title=f"Pivot Chart: {val_col}"
                                )
                                # Add value labels on bars
                                fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                                
                        else:
                            # Single level columns are simpler
                            fig = px.bar(
                                plot_df,
                                x=row_variables[0],  # Use first row variable as x-axis
                                y=metric_col,
                                title=f"Pivot Chart: {metric_col} by {row_variables[0]}"
                            )
                            # Add value labels on bars
                            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Heatmap":
                        # Heatmaps work best with pivot tables
                        # If multi-level index, choose specific levels to plot
                        if len(row_variables) > 1 or len(column_variables) > 1:
                            st.info("Heatmap showing first level of row and column variables")
                        
                        # For heatmap we need to ensure we have a 2D matrix
                        # Get data in right format based on pivot structure
                        if isinstance(pivot_df.columns, pd.MultiIndex):
                            # If multiple metrics, let user choose one
                            if len(value_variables) > 1:
                                value_to_plot = st.selectbox(
                                    "Select Value to Visualize:",
                                    options=value_variables
                                )
                                # Extract the sub-dataframe for this metric
                                heatmap_df = pivot_df.xs(value_to_plot, axis=1, level=0, drop_level=False)
                                
                                # Remove any remaining multi-level complexity if possible
                                if isinstance(heatmap_df.columns, pd.MultiIndex):
                                    heatmap_df.columns = heatmap_df.columns.droplevel(0)
                            else:
                                heatmap_df = pivot_df
                                if isinstance(heatmap_df.columns, pd.MultiIndex):
                                    heatmap_df.columns = heatmap_df.columns.droplevel(0)
                        else:
                            heatmap_df = pivot_df
                        
                        # Create heatmap
                        fig = px.imshow(
                            heatmap_df,
                            aspect="auto",     # Adjust aspect ratio 
                            color_continuous_scale='viridis'
                        )
                        # Add text annotations with values
                        for i in range(len(heatmap_df.index)):
                            for j in range(len(heatmap_df.columns)):
                                try:
                                    value = heatmap_df.iloc[i, j]
                                    if pd.notnull(value):
                                        fig.add_annotation(
                                            x=j, 
                                            y=i,
                                            text=f"{value:.2f}",
                                            showarrow=False,
                                            font=dict(color="white")
                                        )
                                except Exception:
                                    pass
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Line Chart":
                        # Line charts work well for trends over categories
                        # Choose a value to plot if multiple exist
                        if len(value_variables) > 1:
                            value_to_plot = st.selectbox(
                                "Select Value to Visualize:",
                                options=value_variables
                            )
                            metric_col = value_to_plot
                        else:
                            metric_col = value_variables[0]
                            
                        # Determine if we have a multi-level column index
                        if isinstance(pivot_df.columns, pd.MultiIndex):
                            # For multi-level columns, we need to flatten
                            pivot_df_flat = pivot_df.copy()
                            pivot_df_flat.columns = [' - '.join(col).strip() for col in 
                                                   pivot_df.columns.values]
                            plot_df = pivot_df_flat.reset_index()
                            
                            # Get the specific value column names after flattening
                            value_cols = [col for col in plot_df.columns 
                                         if metric_col in col and "Grand Total" not in col]
                            
                            # Create a line chart with multiple lines
                            fig = px.line(
                                plot_df, 
                                x=row_variables[0],
                                y=value_cols,
                                title=f"Pivot Chart: {metric_col} Trends",
                                markers=True  # Add markers at each data point
                            )
                        else:
                            # Single level columns
                            fig = px.line(
                                plot_df,
                                x=row_variables[0],  # Use first row variable as x-axis
                                y=metric_col,
                                title=f"Pivot Chart: {metric_col} by {row_variables[0]}",
                                markers=True  # Add markers at each data point
                            )
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Scatter Plot":
                        # For scatter plots, we need at least two numerical columns
                        if len(value_variables) < 2:
                            st.warning("Scatter plot requires at least two metrics. Please select more value variables.")
                        else:
                            # Let user choose which metrics to plot
                            scatter_options = st.columns(2)
                            with scatter_options[0]:
                                x_metric = st.selectbox(
                                    "X-axis Metric:",
                                    options=value_variables
                                )
                            with scatter_options[1]:
                                y_metric = st.selectbox(
                                    "Y-axis Metric:",
                                    options=[v for v in value_variables if v != x_metric],
                                    index=0
                                )
                            
                            # Prepare data for scatter plot
                            if isinstance(pivot_df.columns, pd.MultiIndex):
                                # Need to extract specific metrics from multi-level columns
                                x_data = pivot_df.xs(x_metric, axis=1, level=0, drop_level=True)
                                y_data = pivot_df.xs(y_metric, axis=1, level=0, drop_level=True)
                                
                                # Combine into a new DataFrame for plotting
                                scatter_data = pd.DataFrame({
                                    'group': x_data.index,
                                    'x_metric': x_data.iloc[:, 0] if len(x_data.columns) > 0 else x_data,
                                    'y_metric': y_data.iloc[:, 0] if len(y_data.columns) > 0 else y_data
                                })
                                
                                fig = px.scatter(
                                    scatter_data,
                                    x='x_metric',
                                    y='y_metric',
                                    hover_name='group',
                                    text='group',
                                    title=f"Scatter Plot: {y_metric} vs {x_metric}",
                                    labels={
                                        'x_metric': x_metric,
                                        'y_metric': y_metric
                                    }
                                )
                            else:
                                # Simpler case for single-level columns
                                scatter_data = pivot_df.reset_index()
                                fig = px.scatter(
                                    scatter_data,
                                    x=x_metric,
                                    y=y_metric,
                                    hover_name=row_variables[0],
                                    title=f"Scatter Plot: {y_metric} vs {x_metric}"
                                )
                            
                            fig.update_traces(marker=dict(size=12))
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generating pivot table: {str(e)}")
            st.info("Try different row, column, or value selections.")
    else:
        st.info("Select at least one row variable and one value variable to generate a pivot table.")

def save_pivot_configuration(config):
    """Save pivot table configuration for future use."""
    # Check if configurations exist in session state
    if 'saved_pivot_configs' not in st.session_state:
        st.session_state.saved_pivot_configs = []
    
    # Add new configuration with timestamp
    from datetime import datetime
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    st.session_state.saved_pivot_configs.append(config)
    return len(st.session_state.saved_pivot_configs) - 1  # Return index of saved config

def load_pivot_configuration(index):
    """Load a previously saved pivot table configuration."""
    if 'saved_pivot_configs' in st.session_state and index < len(st.session_state.saved_pivot_configs):
        return st.session_state.saved_pivot_configs[index]
    return None