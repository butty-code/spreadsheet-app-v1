import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import json
import os
from utils import get_numeric_columns, get_object_columns

# Create directory for saved charts if it doesn't exist
if not os.path.exists("saved_charts"):
    os.makedirs("saved_charts")

def create_chart(df):
    """Create visualization options for the dataframe."""
    # Initialize session state for saved charts if not exists
    if 'saved_charts' not in st.session_state:
        # Load saved charts if file exists
        if os.path.exists("saved_charts/saved_configs.json"):
            try:
                with open("saved_charts/saved_configs.json", "r") as f:
                    st.session_state.saved_charts = json.load(f)
            except:
                st.session_state.saved_charts = {}
        else:
            st.session_state.saved_charts = {}
    
    # Check if dashboard mode is enabled
    if 'dashboard_mode' not in st.session_state:
        st.session_state.dashboard_mode = False
    
    # Get column lists by type
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_object_columns(df)
    
    if not numeric_columns:
        st.warning("No numeric columns found for visualization.")
        return
    
    # Dashboard mode toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("## Data Visualization")
        st.write("Create professional visualizations from your data:")
    with col2:
        st.session_state.dashboard_mode = st.toggle("Dashboard Mode", st.session_state.dashboard_mode)
    
    if st.session_state.dashboard_mode:
        create_dashboard(df, numeric_columns, categorical_columns)
    else:
        create_single_chart(df, numeric_columns, categorical_columns)

def create_single_chart(df, numeric_columns, categorical_columns):
    """Create a single chart with advanced options."""
    # Chart type selection with explanations
    chart_options = {
        "Bar Chart": "Compare values across categories",
        "Line Chart": "Show trends over time or sequence",
        "Scatter Plot": "Show relationship between two values",
        "Histogram": "Show distribution of values",
        "Box Plot": "Show statistical distribution with outliers",
        "Pie Chart": "Show parts of a whole (percentages)",
        "Heatmap": "Show patterns and correlations across many variables",
        "Area Chart": "Show cumulative values across a sequence",
        "Violin Plot": "Show distribution density and statistics",
        "Bubble Chart": "Compare three dimensions of data",
        "Sunburst Chart": "Show hierarchical data as nested rings"
    }
    
    # Initialize variables
    chart_type = "Bar Chart"  # Default chart type
    color_theme = "Default"
    chart_title = "Chart"
    
    # Allow loading saved chart configuration
    st.write("### Create New Chart or Load Saved Configuration")
    load_saved = st.checkbox("Load a saved chart configuration")
    
    if load_saved and st.session_state.saved_charts:
        saved_chart_name = st.selectbox(
            "Select a saved chart:",
            options=list(st.session_state.saved_charts.keys())
        )
        # Load the configuration
        if saved_chart_name:
            chart_config = st.session_state.saved_charts[saved_chart_name]
            chart_type = chart_config.get("chart_type", "Bar Chart")
            st.success(f"Loaded configuration: {saved_chart_name}")
    else:
        # Add descriptions to chart types in the selectbox
        chart_descriptions = [f"{chart}: {desc}" for chart, desc in chart_options.items()]
        selected_chart_with_desc = st.selectbox(
            "Select chart type:",
            options=chart_descriptions
        )
        chart_type = selected_chart_with_desc.split(":", 1)[0]
        
    # Add advanced appearance options
    st.write("### Chart Appearance")
    col1, col2 = st.columns(2)
    with col1:
        color_theme = st.selectbox(
            "Color theme:",
            ["Default", "Viridis", "Plasma", "Inferno", "Magma", "Cividis", 
             "Blues", "Greens", "Reds", "Purples", "Greys"]
        )
    with col2:
        chart_title = st.text_input("Chart title:", value=f"{chart_type} of Data")
    
    # Chart settings based on chart type
    if chart_type == "Bar Chart":
        if not categorical_columns:
            st.warning("Bar charts need at least one categorical column. No categorical columns found.")
            return
            
        # Detect if we're working with racing data and provide appropriate defaults
        racing_columns = {
            'categorical': ['racecourse', 'jockey', 'trainer', 'horse_name', 'race_type', 'race_class', 'going', 'country'],
            'numeric': ['prize_money', 'sp_odds', 'official_rating', 'age', 'win_percentage', 'place_percentage']
        }
        
        # Get default x-axis based on data (prefer racecourse if available)
        default_x = next((col for col in ['racecourse', 'country', 'race_type'] if col in categorical_columns), categorical_columns[0])
        
        # Get default y-axis based on data (prefer prize_money or win_percentage if available)
        numeric_preferences = ['prize_money', 'win_percentage', 'official_rating', 'sp_odds']
        default_y = next((col for col in numeric_preferences if col in numeric_columns), numeric_columns[0])
        
        x_axis = st.selectbox("Select X-axis (categorical):", categorical_columns, index=categorical_columns.index(default_x) if default_x in categorical_columns else 0)
        
        # Provide helpful examples for racing data
        if 'racecourse' in categorical_columns or 'horse_name' in categorical_columns:
            st.info("📊 Racing data tip: Try comparing racecourses by average prize money, or jockeys by win percentages!")
        
        y_axis = st.selectbox("Select Y-axis (numeric):", 
                             [col for col in numeric_columns if col not in ['race_id']], 
                             index=numeric_columns.index(default_y) if default_y in numeric_columns else 0)
        
        # Aggregation method - set more logical defaults for racing data
        default_agg = "Count"
        if y_axis == 'prize_money':
            default_agg = "Mean"
        elif y_axis == 'official_rating':
            default_agg = "Mean"
        elif y_axis == 'sp_odds':
            default_agg = "Mean"
        elif y_axis == 'win_percentage' or y_axis == 'place_percentage':
            default_agg = "Mean"
            
        agg_methods = ["Mean", "Count", "Sum", "Median", "Min", "Max"]
        agg_method = st.selectbox(
            "Select aggregation method:",
            agg_methods,
            index=agg_methods.index(default_agg)
        )
        
        # Create chart
        try:
            # Group and aggregate data
            agg_func = agg_method.lower()
            if agg_func == "mean":
                grouped_data = df.groupby(x_axis)[y_axis].mean().reset_index()
            elif agg_func == "sum":
                grouped_data = df.groupby(x_axis)[y_axis].sum().reset_index()
            elif agg_func == "count":
                grouped_data = df.groupby(x_axis)[y_axis].count().reset_index()
            elif agg_func == "median":
                grouped_data = df.groupby(x_axis)[y_axis].median().reset_index()
            elif agg_func == "min":
                grouped_data = df.groupby(x_axis)[y_axis].min().reset_index()
            elif agg_func == "max":
                grouped_data = df.groupby(x_axis)[y_axis].max().reset_index()
            
            # Sort by value for better visualization (optional)
            sort_by = st.checkbox("Sort bars by value", value=True)
            if sort_by:
                grouped_data = grouped_data.sort_values(by=y_axis, ascending=False)
            
            # Limit number of bars if too many categories
            if len(grouped_data) > 20:
                show_top_n = st.slider("Show top N categories:", min_value=5, max_value=50, value=15)
                grouped_data = grouped_data.head(show_top_n)
            
            # Use selected color theme
            if color_theme != "Default":
                color_scale = color_theme.lower()
            else:
                color_scale = None
                
            # Generate chart with Plotly
            fig = px.bar(
                grouped_data, 
                x=x_axis, 
                y=y_axis, 
                title=chart_title,
                labels={x_axis: x_axis, y_axis: f"{agg_method} of {y_axis}"},
                color_continuous_scale=color_scale
            )
            
            # Add styling
            fig.update_layout(
                plot_bgcolor='rgba(240,240,240,0.2)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                margin=dict(l=40, r=40, t=60, b=40),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating bar chart: {str(e)}")
    
    elif chart_type == "Line Chart":
        x_axis = st.selectbox("Select X-axis:", df.columns.tolist())
        y_axis = st.selectbox("Select Y-axis:", numeric_columns)
        
        # Create chart
        try:
            # Try to sort by x-axis (especially for time series)
            try:
                chart_data = df[[x_axis, y_axis]].sort_values(by=x_axis)
            except:
                chart_data = df[[x_axis, y_axis]]
            
            fig = px.line(
                chart_data, 
                x=x_axis, 
                y=y_axis,
                title=f"{y_axis} over {x_axis}",
                labels={x_axis: x_axis, y_axis: y_axis}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating line chart: {str(e)}")
    
    elif chart_type == "Scatter Plot":
        x_axis = st.selectbox("Select X-axis (numeric):", numeric_columns)
        y_axis = st.selectbox("Select Y-axis (numeric):", 
                             [col for col in numeric_columns if col != x_axis] if len(numeric_columns) > 1 else numeric_columns)
        
        # Optional color grouping
        color_by = st.selectbox(
            "Color points by (optional):",
            ["None"] + categorical_columns
        )
        
        # Create chart
        try:
            if color_by == "None":
                fig = px.scatter(
                    df, 
                    x=x_axis, 
                    y=y_axis,
                    title=f"{y_axis} vs {x_axis}",
                    labels={x_axis: x_axis, y_axis: y_axis}
                )
            else:
                fig = px.scatter(
                    df, 
                    x=x_axis, 
                    y=y_axis,
                    color=color_by,
                    title=f"{y_axis} vs {x_axis} (grouped by {color_by})",
                    labels={x_axis: x_axis, y_axis: y_axis, color_by: color_by}
                )
            
            # Add trend line if requested
            if st.checkbox("Add trend line"):
                fig.update_layout(showlegend=True)
                fig.update_traces(marker=dict(size=8))
                fig = px.scatter(
                    df, 
                    x=x_axis, 
                    y=y_axis,
                    color=color_by if color_by != "None" else None,
                    trendline="ols",
                    title=f"{y_axis} vs {x_axis}" + (f" (grouped by {color_by})" if color_by != "None" else ""),
                    labels={x_axis: x_axis, y_axis: y_axis}
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating scatter plot: {str(e)}")
    
    elif chart_type == "Histogram":
        column = st.selectbox("Select column for histogram:", numeric_columns)
        bins = st.slider("Number of bins:", min_value=5, max_value=100, value=20)
        
        # Create chart
        try:
            fig = px.histogram(
                df, 
                x=column,
                nbins=bins,
                title=f"Histogram of {column}",
                labels={column: column}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating histogram: {str(e)}")
    
    elif chart_type == "Box Plot":
        y_axis = st.selectbox("Select numeric column for box plot:", numeric_columns)
        
        # Optional grouping
        group_by = st.selectbox(
            "Group by (optional):",
            ["None"] + categorical_columns
        )
        
        # Create chart
        try:
            if group_by == "None":
                fig = px.box(
                    df, 
                    y=y_axis,
                    title=f"Box Plot of {y_axis}",
                    labels={y_axis: y_axis}
                )
            else:
                fig = px.box(
                    df, 
                    x=group_by,
                    y=y_axis,
                    title=f"Box Plot of {y_axis} grouped by {group_by}",
                    labels={group_by: group_by, y_axis: y_axis}
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating box plot: {str(e)}")
    
    elif chart_type == "Pie Chart":
        if not categorical_columns:
            st.warning("Pie charts need at least one categorical column. No categorical columns found.")
            return
            
        names = st.selectbox("Select category column:", categorical_columns)
        values = st.selectbox("Select values column:", numeric_columns)
        
        # Create chart
        try:
            # Group and sum data
            pie_data = df.groupby(names)[values].sum().reset_index()
            
            fig = px.pie(
                pie_data, 
                names=names, 
                values=values,
                title=f"Distribution of {values} by {names}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating pie chart: {str(e)}")
    
    elif chart_type == "Heatmap":
        if len(numeric_columns) < 2:
            st.warning("Heatmaps need at least two numeric columns. Not enough numeric columns found.")
            return
            
        # Select columns for correlation
        selected_columns = st.multiselect(
            "Select columns for correlation heatmap:",
            numeric_columns,
            default=numeric_columns[:min(5, len(numeric_columns))]
        )
        
        if not selected_columns or len(selected_columns) < 2:
            st.warning("Please select at least two columns for the heatmap.")
            return
        
        # Create chart
        try:
            # Calculate correlation matrix
            corr_matrix = df[selected_columns].corr()
            
            # Create heatmap with Plotly
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show the correlation matrix as a table
            if st.checkbox("Show correlation values"):
                st.write("Correlation Matrix:")
                st.write(corr_matrix)
            
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")

    # Chart configuration saving
    st.write("### Save This Chart Configuration")
    
    try:
        # Initialize variables for config
        used_columns = []
        # Safely check for variables
        x_axis_val = locals().get('x_axis', None)
        y_axis_val = locals().get('y_axis', None)
        
        if x_axis_val is not None:
            used_columns.append(x_axis_val)
        if y_axis_val is not None:
            used_columns.append(y_axis_val)
            
        # Create chart config with all relevant parameters
        current_config = {
            "chart_type": chart_type,
            "columns_used": used_columns,
            "color_theme": color_theme,
            "chart_title": chart_title,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        st.error(f"Error preparing chart configuration: {str(e)}")
        current_config = {
            "chart_type": chart_type,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    col1, col2 = st.columns([3, 1])
    with col1:
        chart_name = st.text_input("Enter a name for this chart configuration:", 
                                    value=f"{chart_type} - {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    
    with col2:
        if st.button("Save Configuration"):
            st.session_state.saved_charts[chart_name] = current_config
            # Save to disk
            with open("saved_charts/saved_configs.json", "w") as f:
                json.dump(st.session_state.saved_charts, f)
            st.success(f"Saved chart configuration: {chart_name}")
    
    # Option to download the chart
    st.write("Note: You can download the chart by clicking the camera icon in the top-right corner of the chart.")
    
def create_dashboard(df, numeric_columns, categorical_columns):
    """Create a dashboard with multiple charts."""
    st.write("### Interactive Dashboard")
    st.write("Create a professional dashboard with multiple charts")
    
    # Dashboard layout options
    layout = st.radio("Select dashboard layout:", 
                     ["2×2 Grid (4 charts)", "1×3 Row (3 charts)", "3×1 Column (3 charts)"])
    
    # Select charts to include
    st.write("### Select Charts for Dashboard")
    
    # Load saved chart configurations
    saved_configs = st.session_state.saved_charts
    if not saved_configs:
        st.info("No saved chart configurations found. Create and save some charts first.")
        return
    
    # Let user select from saved charts
    selected_charts = []
    if layout == "2×2 Grid (4 charts)":
        max_charts = 4
    else:
        max_charts = 3
    
    # Create columns for chart selection
    cols = st.columns(max_charts)
    for i in range(max_charts):
        with cols[i]:
            st.write(f"Chart {i+1}")
            selected = st.selectbox(
                f"Select saved chart {i+1}:",
                options=["None"] + list(saved_configs.keys()),
                key=f"chart_select_{i}"
            )
            if selected != "None":
                selected_charts.append(selected)
    
    # Create dashboard if charts are selected
    if selected_charts:
        st.write("### Your Dashboard")
        
        if layout == "2×2 Grid (4 charts)":
            # Create a 2x2 grid of charts
            fig = make_subplots(rows=2, cols=2, subplot_titles=selected_charts[:4])
            
            # Generate charts
            for i, chart_name in enumerate(selected_charts[:4]):
                row, col = (i // 2) + 1, (i % 2) + 1
                chart_config = saved_configs[chart_name]
                
                # Create a modified figure based on the saved config
                chart_fig = create_chart_from_config(df, chart_config, numeric_columns, categorical_columns)
                
                # Add the traces from chart_fig to the dashboard
                if chart_fig:
                    for trace in chart_fig.data:
                        fig.add_trace(trace, row=row, col=col)
            
            # Update layout
            fig.update_layout(height=800, title_text="Excel Analyzer Dashboard")
            st.plotly_chart(fig, use_container_width=True)
        
        elif layout == "1×3 Row (3 charts)":
            # Create a row of 3 charts
            fig = make_subplots(rows=1, cols=3, subplot_titles=selected_charts[:3])
            
            # Generate charts
            for i, chart_name in enumerate(selected_charts[:3]):
                chart_config = saved_configs[chart_name]
                
                # Create a figure based on the saved config
                chart_fig = create_chart_from_config(df, chart_config, numeric_columns, categorical_columns)
                
                # Add traces to the dashboard
                if chart_fig:
                    for trace in chart_fig.data:
                        fig.add_trace(trace, row=1, col=i+1)
            
            # Update layout
            fig.update_layout(height=500, title_text="Excel Analyzer Dashboard")
            st.plotly_chart(fig, use_container_width=True)
        
        elif layout == "3×1 Column (3 charts)":
            # Create a column of 3 charts
            fig = make_subplots(rows=3, cols=1, subplot_titles=selected_charts[:3])
            
            # Generate charts
            for i, chart_name in enumerate(selected_charts[:3]):
                chart_config = saved_configs[chart_name]
                
                # Create a figure based on the saved config
                chart_fig = create_chart_from_config(df, chart_config, numeric_columns, categorical_columns)
                
                # Add traces to the dashboard
                if chart_fig:
                    for trace in chart_fig.data:
                        fig.add_trace(trace, row=i+1, col=1)
            
            # Update layout
            fig.update_layout(height=900, title_text="Excel Analyzer Dashboard")
            st.plotly_chart(fig, use_container_width=True)
        
        # Option to download the dashboard
        st.write("Note: You can download the dashboard by clicking the camera icon in the top-right corner.")

def create_chart_from_config(df, config, numeric_columns, categorical_columns):
    """Create a chart based on a saved configuration."""
    try:
        chart_type = config.get("chart_type", "Bar Chart")
        color_theme = config.get("color_theme", "Default")
        chart_title = config.get("chart_title", f"{chart_type} Chart")
        
        # Apply color theme if specified
        if color_theme != "Default":
            color_scale = color_theme.lower()
        else:
            color_scale = None
        
        # Use the first available columns if none specified
        if len(df.columns) > 0:
            x_col = df.columns[0]
            y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        else:
            return None
        
        # Create appropriate chart based on type
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, title=chart_title, color_continuous_scale=color_scale)
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, title=chart_title, color_discrete_sequence=[color_scale])
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, title=chart_title, color_continuous_scale=color_scale)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, title=chart_title, color_discrete_sequence=[color_scale])
        elif chart_type == "Box Plot":
            fig = px.box(df, y=x_col, title=chart_title, color_discrete_sequence=[color_scale])
        elif chart_type == "Pie Chart" and len(df.columns) > 1:
            fig = px.pie(df, names=x_col, values=y_col, title=chart_title, color_discrete_sequence=px.colors.sequential.get(color_scale, None))
        elif chart_type == "Heatmap" and len(numeric_columns) > 1:
            # Use correlation matrix for heatmap
            corr_matrix = df[numeric_columns].corr()
            fig = px.imshow(corr_matrix, text_auto=True, title=chart_title, color_continuous_scale=color_scale)
        elif chart_type == "Area Chart":
            fig = px.area(df, x=x_col, y=y_col, title=chart_title, color_discrete_sequence=[color_scale])
        elif chart_type == "Violin Plot" and len(numeric_columns) > 0:
            fig = px.violin(df, y=numeric_columns[0], title=chart_title, color_discrete_sequence=[color_scale])
        elif chart_type == "Bubble Chart" and len(numeric_columns) > 2:
            size_col = numeric_columns[2] if len(numeric_columns) > 2 else None
            fig = px.scatter(df, x=numeric_columns[0], y=numeric_columns[1], 
                          size=size_col, title=chart_title, color_continuous_scale=color_scale)
        else:
            # Default to a simple bar chart if the specified type can't be created
            fig = px.bar(df, x=x_col, y=y_col, title=f"Default {chart_title}")
        
        # Add styling to all charts
        fig.update_layout(
            plot_bgcolor='rgba(240,240,240,0.2)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart from configuration: {str(e)}")
        return None
