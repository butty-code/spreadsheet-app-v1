import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
import base64
from datetime import datetime
import zipfile

def run_powerbi_export(df):
    """
    Create UI for Power BI export options and template generation.
    """
    st.markdown("## Power BI Integration")
    st.info("Prepare your data for analysis in Power BI with optimized exports and templates.")
    
    # Create tabs for different Power BI integration features
    tab1, tab2, tab3, tab4 = st.tabs(["Data Export", "Template Generation", "DAX Functions", "Best Practices"])
    
    with tab1:
        st.markdown("### Optimize Data for Power BI")
        st.write("Prepare your data for optimal performance in Power BI.")
        
        # Data transformation options
        st.markdown("#### Data Transformation Options")
        
        col1, col2 = st.columns(2)
        with col1:
            optimize_datatypes = st.checkbox("Optimize data types", value=True,
                                           help="Convert columns to most efficient data types")
            create_date_table = st.checkbox("Create date table", value=True,
                                          help="Generate a separate date dimension table for better date filtering")
        
        with col2:
            remove_duplicates = st.checkbox("Remove duplicate rows", value=False,
                                          help="Identify and remove duplicate records")
            add_row_id = st.checkbox("Add row ID column", value=True,
                                    help="Add a unique identifier column")
        
        # Advanced options
        with st.expander("Advanced Options"):
            date_columns = st.multiselect(
                "Select date columns for calendar relationships:",
                options=[col for col in df.columns if 'date' in col.lower() or 
                        df[col].dtype == 'datetime64[ns]' or
                        (df[col].dtype == 'object' and is_potential_date_column(df[col]))],
                help="Date columns will be related to the generated date table"
            )
            
            # Table naming
            fact_table_name = st.text_input("Fact Table Name", "FactTable", 
                                           help="Name for the main data table in Power BI")
            
            # Key metrics
            metric_columns = st.multiselect(
                "Select key metric columns:",
                options=df.select_dtypes(include=['number']).columns.tolist(),
                help="These columns will be highlighted as key measures in Power BI"
            )
            
            # Hierarchies
            hierarchies = create_hierarchy_ui(df)
        
        if st.button("Generate Power BI Data Export", type="primary"):
            with st.spinner("Preparing data for Power BI..."):
                # Process the data based on options selected
                processed_df = prepare_data_for_powerbi(
                    df, 
                    optimize_datatypes, 
                    remove_duplicates, 
                    add_row_id, 
                    fact_table_name
                )
                
                # Generate date table if option selected
                date_tables = {}
                date_relationships = []
                if create_date_table and date_columns:
                    for date_col in date_columns:
                        try:
                            date_table, relationship = generate_date_table(
                                processed_df, date_col, fact_table_name
                            )
                            table_name = f"Date_{date_col.replace(' ', '_')}"
                            date_tables[table_name] = date_table
                            date_relationships.append(relationship)
                        except:
                            st.warning(f"Could not generate date table for {date_col}")
                
                # Create model.json configuration
                model_json = generate_model_json(
                    processed_df, 
                    fact_table_name,
                    date_tables,
                    date_relationships,
                    metric_columns,
                    hierarchies
                )
                
                # Provide download links
                col1, col2 = st.columns(2)
                with col1:
                    # Main data CSV
                    csv_data = processed_df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {fact_table_name}.csv",
                        data=csv_data,
                        file_name=f"{fact_table_name}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Model configuration JSON
                    st.download_button(
                        label="Download Power BI Model Config",
                        data=json.dumps(model_json, indent=2),
                        file_name="powerbi_model_config.json",
                        mime="application/json"
                    )
                
                # If date tables were created, provide download links
                if date_tables:
                    st.markdown("#### Date Tables")
                    cols = st.columns(min(len(date_tables), 3))
                    for i, (table_name, date_table) in enumerate(date_tables.items()):
                        with cols[i % len(cols)]:
                            csv_data = date_table.to_csv(index=False)
                            st.download_button(
                                label=f"Download {table_name}.csv",
                                data=csv_data,
                                file_name=f"{table_name}.csv",
                                mime="text/csv"
                            )
                
                # Create a zip file with all components
                with io.BytesIO() as buffer:
                    with zipfile.ZipFile(buffer, "w") as zip_file:
                        # Add main data file
                        zip_file.writestr(
                            f"{fact_table_name}.csv", 
                            processed_df.to_csv(index=False)
                        )
                        
                        # Add model configuration
                        zip_file.writestr(
                            "powerbi_model_config.json",
                            json.dumps(model_json, indent=2)
                        )
                        
                        # Add date tables
                        for table_name, date_table in date_tables.items():
                            zip_file.writestr(
                                f"{table_name}.csv",
                                date_table.to_csv(index=False)
                            )
                        
                        # Add readme with instructions
                        zip_file.writestr(
                            "README.txt",
                            generate_readme(fact_table_name, list(date_tables.keys()))
                        )
                    
                    buffer.seek(0)
                    st.download_button(
                        label="Download All Files (ZIP)",
                        data=buffer,
                        file_name="powerbi_data_files.zip",
                        mime="application/zip"
                    )
                
                st.success("Data prepared successfully for Power BI!")
                
                # Display preview of the processed data
                st.markdown("#### Preview of Processed Data")
                st.dataframe(processed_df.head(10))
    
    with tab2:
        st.markdown("### Power BI Template Generation")
        st.write("Generate a Power BI template file (.pbit) with pre-configured visuals and relationships.")
        
        # Template configuration options
        st.markdown("#### Template Configuration")
        
        template_name = st.text_input("Template Name", "Data Analysis Template")
        
        col1, col2 = st.columns(2)
        with col1:
            include_visuals = st.checkbox("Include sample visuals", value=True,
                                        help="Add pre-configured dashboards and reports")
            include_measures = st.checkbox("Generate common measures", value=True,
                                         help="Create DAX measures for common calculations")
        
        with col2:
            include_relationships = st.checkbox("Configure relationships", value=True,
                                              help="Set up data relationships automatically")
            include_theme = st.checkbox("Apply custom theme", value=True,
                                       help="Use a professional color scheme")
        
        # Select columns for different visual types
        st.markdown("#### Dashboard Configuration")
        
        with st.expander("Visualization Settings"):
            # For bar charts
            bar_chart_cols = st.multiselect(
                "Columns for bar charts:",
                options=df.columns.tolist(),
                default=[col for col in df.columns if df[col].dtype in ['int64', 'float64']][:2],
                help="Select numeric columns to visualize in bar charts"
            )
            
            # For line charts
            line_chart_cols = st.multiselect(
                "Columns for line charts:",
                options=df.columns.tolist(),
                default=[col for col in df.columns if 'date' in col.lower()][:1] + 
                         [col for col in df.columns if df[col].dtype in ['int64', 'float64']][:1],
                help="Select date and numeric columns to visualize in line charts"
            )
            
            # For tables
            table_cols = st.multiselect(
                "Columns for tables:",
                options=df.columns.tolist(),
                default=df.columns.tolist()[:5],
                help="Select columns to include in table visuals"
            )
        
        # Generate template when button is clicked
        if st.button("Generate Power BI Template", type="primary"):
            with st.spinner("Generating Power BI template..."):
                # Since we can't actually create a .pbit file directly,
                # we'll create a comprehensive setup document instead
                template_instructions = generate_template_instructions(
                    df,
                    template_name,
                    include_visuals,
                    include_measures,
                    include_relationships,
                    include_theme,
                    bar_chart_cols,
                    line_chart_cols,
                    table_cols
                )
                
                # Provide download for instructions
                st.download_button(
                    label="Download Template Instructions",
                    data=template_instructions,
                    file_name="powerbi_template_instructions.md",
                    mime="text/markdown"
                )
                
                st.success("Template instructions generated successfully!")
                
                # Display instructions preview
                st.markdown("#### Template Instructions Preview")
                st.markdown(template_instructions[:1000] + "...", unsafe_allow_html=True)
                
                # Show disclaimer
                st.info("Note: While we can't generate a .pbit file directly, these instructions will help you set up an equivalent template in Power BI Desktop.")
    
    with tab3:
        st.markdown("### DAX Functions Reference")
        st.write("Learn essential DAX functions for your Power BI reports")
        
        with st.expander("📌 How to use DAX functions", expanded=True):
            st.markdown("""
            **DAX (Data Analysis Expressions)** is the formula language used in Power BI.
            
            ### When to use DAX:
            - Creating calculated columns
            - Creating measures for aggregations or complex calculations
            - Filtering and manipulating data within visuals
            - Building time intelligence calculations
            
            ### How to use this reference:
            1. Browse the function categories below
            2. Click on any function to see its syntax and examples
            3. Copy and adapt the examples to your own data
            4. Test in Power BI by creating a new measure
            """)
        
        # Create tabs for different DAX function categories
        dax_tab1, dax_tab2, dax_tab3, dax_tab4, dax_tab5 = st.tabs([
            "Aggregation Functions", "Time Intelligence", "Filter Functions", "Text & Logical", "Racing-Specific Examples"
        ])
        
        with dax_tab1:
            st.markdown("### Aggregation Functions")
            st.markdown("""
            #### SUM
            ```
            Sales = SUM(Sales[Amount])
            ```
            *Adds up all values in the Amount column*
            
            #### AVERAGE
            ```
            AvgSales = AVERAGE(Sales[Amount])
            ```
            *Calculates the average of values in the Amount column*
            
            #### MIN / MAX
            ```
            FastestTime = MIN(Races[FinishTime])
            SlowestTime = MAX(Races[FinishTime])
            ```
            *Finds the minimum or maximum value in a column*
            
            #### COUNT / COUNTA / COUNTROWS
            ```
            TotalRaces = COUNTROWS(Races)
            RidersWithResults = COUNTA(Races[RiderName])
            ```
            *Counts rows or non-blank values*
            
            #### DISTINCTCOUNT
            ```
            UniqueRiders = DISTINCTCOUNT(Races[RiderName])
            ```
            *Counts unique values in a column*
            """)
            
        with dax_tab2:
            st.markdown("### Time Intelligence Functions")
            st.markdown("""
            #### TOTALYTD (Year to Date)
            ```
            YTDSales = TOTALYTD(SUM(Sales[Amount]), 'Date'[Date])
            ```
            *Calculates year-to-date total of the Amount column*
            
            #### SAMEPERIODLASTYEAR
            ```
            SalesPreviousYear = CALCULATE(
                SUM(Sales[Amount]),
                SAMEPERIODLASTYEAR('Date'[Date])
            )
            ```
            *Calculates amount for the same period in the previous year*
            
            #### DATEADD
            ```
            SalesLastMonth = CALCULATE(
                SUM(Sales[Amount]),
                DATEADD('Date'[Date], -1, MONTH)
            )
            ```
            *Shifts date values by specified interval (e.g., month, quarter, year)*
            
            #### PARALLELPERIOD
            ```
            SalesParallelQuarter = CALCULATE(
                SUM(Sales[Amount]),
                PARALLELPERIOD('Date'[Date], -1, QUARTER)
            )
            ```
            *Returns date values for a period parallel to specified dates*
            
            #### DATESYTD / DATESQTD / DATESMTD
            ```
            YearToDateFilter = DATESYTD('Date'[Date])
            YTDSales = CALCULATE(SUM(Sales[Amount]), YearToDateFilter)
            ```
            *Returns a table of dates from start of year/quarter/month to specified date*
            """)
            
        with dax_tab3:
            st.markdown("### Filter Functions")
            st.markdown("""
            #### CALCULATE
            ```
            HighValueSales = CALCULATE(
                SUM(Sales[Amount]),
                Sales[Amount] > 1000
            )
            ```
            *Evaluates an expression in a modified filter context*
            
            #### FILTER
            ```
            Top10Sales = CALCULATE(
                SUM(Sales[Amount]),
                FILTER(Sales, Sales[Amount] >= 1000)
            )
            ```
            *Returns a filtered table based on specified conditions*
            
            #### ALL / ALLEXCEPT
            ```
            TotalSales = CALCULATE(
                SUM(Sales[Amount]),
                ALL(Sales)
            )
            ```
            *Removes filters from specified tables or columns*
            
            #### TOPN
            ```
            Top5Races = CALCULATE(
                COUNTROWS(Races),
                TOPN(5, Races, Races[Prize], DESC)
            )
            ```
            *Returns the top N rows from specified table by sorting on an expression*
            
            #### RANKX
            ```
            RacerRank = RANKX(
                ALL(Racers),
                CALCULATE(SUM(Results[Points])),
                ,
                DESC
            )
            ```
            *Ranks values in a column across the specified table*
            """)
            
        with dax_tab4:
            st.markdown("### Text & Logical Functions")
            st.markdown("""
            #### CONCATENATE / CONCATENATEX
            ```
            FullName = CONCATENATE(People[FirstName], " " & People[LastName])
            
            AllRacers = CONCATENATEX(
                Racers,
                Racers[Name],
                ", "
            )
            ```
            *Joins text strings together*
            
            #### IF / SWITCH
            ```
            PriceCategory = IF(
                Products[Price] > 100,
                "Premium",
                "Standard"
            )
            
            RatingText = SWITCH(
                Products[Rating],
                5, "Excellent",
                4, "Good",
                3, "Average",
                "Poor"
            )
            ```
            *Checks conditions and returns different values based on results*
            
            #### FORMAT
            ```
            FormattedDate = FORMAT('Date'[Date], "MMM YYYY")
            ```
            *Converts values to formatted text*
            
            #### RELATED / RELATEDTABLE
            ```
            CategoryName = RELATED(Categories[Name])
            ```
            *Retrieves a related value from another table*
            
            #### ISBLANK / ISEMPTY
            ```
            HasResults = IF(
                ISBLANK(Races[Result]),
                "No",
                "Yes"
            )
            ```
            *Checks if a value is blank or if a table is empty*
            """)
            
        with dax_tab5:
            st.markdown("### Racing-Specific DAX Examples")
            st.markdown("""
            #### Win Percentage
            ```
            WinPercentage = 
            DIVIDE(
                CALCULATE(
                    COUNTROWS(Results),
                    Results[Position] = 1
                ),
                COUNTROWS(Results),
                0
            ) * 100
            ```
            *Calculates the percentage of races won by each racer*
            
            #### Average Position
            ```
            AvgPosition = 
            AVERAGE(Results[Position])
            ```
            *Calculates the average finishing position*
            
            #### Points Per Race
            ```
            PointsPerRace = 
            DIVIDE(
                SUM(Results[Points]),
                COUNTROWS(Results),
                0
            )
            ```
            *Calculates average points earned per race*
            
            #### Position Improvement
            ```
            PositionImprovement = 
            AVERAGEX(
                Results,
                Results[StartPosition] - Results[FinishPosition]
            )
            ```
            *Positive values indicate improvement from starting position*
            
            #### Season Comparison
            ```
            PointsYOY = 
            VAR CurrentSeasonPoints = CALCULATE(
                SUM(Results[Points]),
                FILTER(
                    ALL('Date'),
                    'Date'[Season] = MAX('Date'[Season])
                )
            )
            VAR PriorSeasonPoints = CALCULATE(
                SUM(Results[Points]),
                FILTER(
                    ALL('Date'),
                    'Date'[Season] = MAX('Date'[Season]) - 1
                )
            )
            RETURN
            IF(
                ISBLANK(PriorSeasonPoints),
                BLANK(),
                CurrentSeasonPoints - PriorSeasonPoints
            )
            ```
            *Compares points between current and previous season*
            """)
    
    with tab4:
        st.markdown("### Power BI Best Practices")
        st.write("Learn how to optimize your Power BI reports with these best practices.")
        
        # Best practices accordion
        with st.expander("Data Modeling Tips", expanded=True):
            st.markdown("""
            * **Use star schema**: Organize your data into fact and dimension tables
            * **Avoid relationships on calculated columns**: They're not as efficient
            * **Minimize row count**: Pre-aggregate data when possible
            * **Use integers for relationships**: They're more efficient than text
            * **Keep column names simple**: Avoid spaces and special characters
            """)
        
        with st.expander("Performance Optimization"):
            st.markdown("""
            * **Use appropriate data types**: Make sure each column uses the optimal data type
            * **Use calculated columns sparingly**: Prefer measures when possible
            * **Import vs. DirectQuery**: Import mode is typically faster
            * **Disable unnecessary interactions**: Limit cross-filtering when not needed
            * **Avoid complex visuals**: Simple visuals perform better than complex ones
            """)
        
        with st.expander("Visualization Best Practices"):
            st.markdown("""
            * **Start with an overview**: Provide high-level KPIs first
            * **Use consistent formatting**: Maintain visual hierarchy with consistent colors and fonts
            * **Limit visuals per page**: Aim for 4-8 visuals per dashboard page
            * **Use slicers wisely**: Place common filters in a filter pane
            * **Consider accessibility**: Use alt text and color-blind friendly palettes
            """)
        
        with st.expander("DAX Formulas"):
            st.markdown("""
            * **Keep measures simple**: Break complex calculations into smaller steps
            * **Use variables**: Improve readability and performance
            * **Avoid CALCULATE with FILTER**: Use ALL with conditions instead
            * **Understand evaluation context**: Context transition can impact performance
            * **Use SUMMARIZE instead of GROUPBY**: It's often more efficient
            """)

# Helper functions

def is_potential_date_column(series):
    """Check if a column might contain dates stored as strings."""
    if series.dtype != 'object':
        return False
    
    # Sample non-null values
    sample = series.dropna().sample(min(10, len(series.dropna())))
    
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY or MM/DD/YYYY
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY/MM/DD
        r'\d{1,2}-[A-Za-z]{3}-\d{2,4}',     # DD-MMM-YYYY
    ]
    
    import re
    pattern_matches = 0
    
    for val in sample:
        if not isinstance(val, str):
            continue
        for pattern in date_patterns:
            if re.search(pattern, val):
                pattern_matches += 1
                break
    
    # If more than half the samples match a date pattern
    return pattern_matches >= len(sample) / 2

def create_hierarchy_ui(df):
    """Create UI for defining hierarchies."""
    hierarchies = []
    
    st.write("Define Hierarchies (optional)")
    
    # Predefined hierarchies based on common patterns
    predefined = {
        "Date Hierarchy": [col for col in df.columns if 'year' in col.lower() or 
                         'month' in col.lower() or 
                         'day' in col.lower() or 
                         'quarter' in col.lower()],
        "Geography Hierarchy": [col for col in df.columns if 'country' in col.lower() or 
                              'region' in col.lower() or 
                              'city' in col.lower() or 
                              'state' in col.lower()],
        "Product Hierarchy": [col for col in df.columns if 'category' in col.lower() or 
                            'subcategory' in col.lower() or 
                            'product' in col.lower() or 
                            'item' in col.lower()],
        "Racing Hierarchy": [col for col in df.columns if 'race_type' in col.lower() or 
                           'racecourse' in col.lower() or 
                           'track_condition' in col.lower()]
    }
    
    # Only offer predefined hierarchies that have at least 2 columns
    valid_predefined = {k: v for k, v in predefined.items() if len(v) >= 2}
    
    if valid_predefined:
        for name, columns in valid_predefined.items():
            if st.checkbox(f"Use {name}", value=True if len(columns) >= 2 else False):
                hierarchy = {
                    "name": name,
                    "columns": columns
                }
                hierarchies.append(hierarchy)
    
    # Custom hierarchy - moved out of expander to avoid nesting
    st.markdown("### Add Custom Hierarchy")
    hierarchy_name = st.text_input("Hierarchy Name", "Custom Hierarchy")
    hierarchy_columns = st.multiselect(
        "Select columns (in hierarchical order):",
        options=df.columns.tolist()
    )
    
    if st.button("Add Hierarchy") and hierarchy_columns:
        hierarchy = {
            "name": hierarchy_name,
            "columns": hierarchy_columns
        }
        hierarchies.append(hierarchy)
        st.success(f"Added {hierarchy_name} hierarchy")
    
    return hierarchies

def prepare_data_for_powerbi(df, optimize_datatypes, remove_duplicates, add_row_id, table_name):
    """Prepare data according to Power BI best practices."""
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Add row ID if requested
    if add_row_id:
        processed_df.insert(0, f"{table_name}ID", range(1, len(processed_df) + 1))
    
    # Remove duplicates if requested
    if remove_duplicates:
        original_len = len(processed_df)
        processed_df = processed_df.drop_duplicates()
        if len(processed_df) < original_len:
            st.info(f"Removed {original_len - len(processed_df)} duplicate rows")
    
    # Optimize data types if requested
    if optimize_datatypes:
        # Try to convert object columns to more efficient types
        for col in processed_df.select_dtypes(include=['object']):
            # Try to convert to datetime
            try:
                if is_potential_date_column(processed_df[col]):
                    processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
                    continue
            except:
                pass
            
            # Try to convert to numeric
            try:
                numeric_col = pd.to_numeric(processed_df[col], errors='coerce')
                if numeric_col.notna().sum() > 0.8 * len(numeric_col):
                    processed_df[col] = numeric_col
                    continue
            except:
                pass
            
            # Try to convert to categorical if many repeats
            if processed_df[col].nunique() < len(processed_df) * 0.2:
                processed_df[col] = processed_df[col].astype('category')
    
    return processed_df

def generate_date_table(df, date_column, fact_table_name):
    """Generate a date table for the selected date column."""
    # Extract unique dates
    try:
        dates = pd.to_datetime(df[date_column], errors='coerce').dropna().unique()
        
        # Create a continuous range from min to max date
        min_date = min(dates)
        max_date = max(dates)
        
        # Create date table
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        date_df = pd.DataFrame({'Date': date_range})
        
        # Add useful date columns
        date_df['Year'] = date_df['Date'].dt.year
        date_df['Quarter'] = date_df['Date'].dt.quarter
        date_df['Month'] = date_df['Date'].dt.month
        date_df['MonthName'] = date_df['Date'].dt.strftime('%B')
        date_df['Week'] = date_df['Date'].dt.isocalendar().week
        date_df['Day'] = date_df['Date'].dt.day
        date_df['DayOfWeek'] = date_df['Date'].dt.dayofweek
        date_df['DayName'] = date_df['Date'].dt.strftime('%A')
        date_df['IsWeekend'] = date_df['DayOfWeek'].isin([5, 6])
        
        # Add fiscal year (assuming fiscal year starts in July)
        date_df['FiscalYear'] = date_df['Year'] + date_df['Month'].apply(lambda x: 1 if x >= 7 else 0)
        
        # Add DateKey for relationships (YYYYMMDD format)
        date_df['DateKey'] = date_df['Date'].dt.strftime('%Y%m%d').astype(int)
        
        # Create relationship info
        relationship = {
            "from": {
                "table": f"Date_{date_column.replace(' ', '_')}",
                "column": "Date"
            },
            "to": {
                "table": fact_table_name,
                "column": date_column
            }
        }
        
        return date_df, relationship
    except Exception as e:
        st.error(f"Error generating date table: {str(e)}")
        # Return empty dataframe and relationship
        return pd.DataFrame(), {}

def generate_model_json(df, table_name, date_tables, date_relationships, metric_columns, hierarchies):
    """Generate a model.json file with table and relationship definitions."""
    model = {
        "tables": [],
        "relationships": [],
        "measures": [],
        "hierarchies": []
    }
    
    # Add main fact table
    fact_table = {
        "name": table_name,
        "columns": []
    }
    
    for col in df.columns:
        column_info = {
            "name": col,
            "dataType": map_dtype_to_powerbi(df[col].dtype)
        }
        
        # Mark metrics
        if col in metric_columns:
            column_info["isMetric"] = True
        
        fact_table["columns"].append(column_info)
    
    model["tables"].append(fact_table)
    
    # Add date tables
    for table_name, date_df in date_tables.items():
        date_table = {
            "name": table_name,
            "isDateTable": True,
            "columns": []
        }
        
        for col in date_df.columns:
            column_info = {
                "name": col,
                "dataType": map_dtype_to_powerbi(date_df[col].dtype)
            }
            date_table["columns"].append(column_info)
        
        model["tables"].append(date_table)
    
    # Add relationships
    for relationship in date_relationships:
        model["relationships"].append(relationship)
    
    # Add hierarchies
    for hierarchy in hierarchies:
        hierarchy_def = {
            "name": hierarchy["name"],
            "table": table_name,
            "levels": []
        }
        
        for i, column in enumerate(hierarchy["columns"]):
            level = {
                "name": column,
                "column": column,
                "ordinal": i
            }
            hierarchy_def["levels"].append(level)
        
        model["hierarchies"].append(hierarchy_def)
    
    # Add some basic measures
    if metric_columns:
        for column in metric_columns:
            measure = {
                "name": f"Avg {column}",
                "table": table_name,
                "expression": f"AVERAGE({table_name}[{column}])"
            }
            model["measures"].append(measure)
            
            measure = {
                "name": f"Sum {column}",
                "table": table_name,
                "expression": f"SUM({table_name}[{column}])"
            }
            model["measures"].append(measure)
    
    return model

def map_dtype_to_powerbi(dtype):
    """Map pandas data types to Power BI data types."""
    dtype_str = str(dtype)
    
    if 'int' in dtype_str:
        return "int64"
    elif 'float' in dtype_str:
        return "double"
    elif 'datetime' in dtype_str:
        return "dateTime"
    elif 'bool' in dtype_str:
        return "boolean"
    elif 'category' in dtype_str:
        return "string"
    else:
        return "string"

def generate_readme(fact_table_name, date_table_names):
    """Generate a README file with instructions."""
    return f"""# Power BI Data Files

This folder contains data files optimized for use in Microsoft Power BI.

## Contents

1. `{fact_table_name}.csv` - Main fact table containing your data
2. `powerbi_model_config.json` - Power BI model configuration
{chr(10).join([f"3. `{name}.csv` - Date dimension table" for name in date_table_names])}

## How to Use

1. Open Power BI Desktop
2. Import the CSV files using Get Data > CSV
3. Apply the relationships defined in the model configuration
4. Build your reports and dashboards

## Best Practices

- Use the date tables for time intelligence functions
- Create calculated columns only when necessary
- Use measures for aggregations
- Consider using incremental refresh for large datasets

For more information, visit the Microsoft Power BI documentation:
https://docs.microsoft.com/en-us/power-bi/

"""

def generate_template_instructions(df, template_name, include_visuals, include_measures, 
                                 include_relationships, include_theme, bar_chart_cols, 
                                 line_chart_cols, table_cols):
    """Generate detailed instructions for setting up a Power BI template."""
    instructions = f"""# Power BI Template: {template_name}

This document provides instructions for setting up a Power BI template based on your data.
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Overview

Your dataset has {len(df)} rows and {len(df.columns)} columns.

### Column Summary

```
{df.dtypes.to_string()}
```

## Template Setup Instructions

Follow these steps to recreate this template in Power BI Desktop:

1. **Create a new Power BI Desktop file**
2. **Import your data**
   - Use Get Data > CSV to import your main data file
"""

    if include_relationships:
        instructions += """   
3. **Set up relationships**
   - Go to Model view and create the following relationships:
"""
        # Add any relationship instructions based on the data
        date_columns = [col for col in df.columns if 'date' in col.lower() or 
                      df[col].dtype == 'datetime64[ns]']
        for date_col in date_columns:
            instructions += f"     - Connect `Date[Date]` to `Data[{date_col}]` (Many to One)\n"

    if include_measures:
        instructions += """
4. **Create calculated measures**
   - Create the following DAX measures:
"""
        # Add measure examples for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        for i, col in enumerate(numeric_cols[:5]):  # Limit to first 5
            instructions += f"""
     ```
     Avg {col} = AVERAGE(Data[{col}])
     Total {col} = SUM(Data[{col}])
     {col} YoY Change = 
         VAR CurrentValue = SUM(Data[{col}])
         VAR PriorValue = CALCULATE(SUM(Data[{col}]), SAMEPERIODLASTYEAR('Date'[Date]))
         RETURN
             IF(PriorValue = 0, BLANK(), (CurrentValue - PriorValue) / PriorValue)
     ```
"""

    if include_visuals:
        instructions += """
5. **Create dashboard pages**
   - Create the following pages in your report:
"""
        
        # Overview page
        instructions += """
   ### Overview Page
   
   - **Top section**: KPI cards showing:
"""
        for col in df.select_dtypes(include=['number']).columns.tolist()[:3]:
            instructions += f"     - Total {col}\n"
            
        instructions += """
   - **Middle section**: Two column layout with:
     - Left: Bar chart showing overall performance
     - Right: Line chart showing trends over time
   
   - **Bottom section**: Table with detailed data
"""

        # Detailed analysis page
        instructions += """
   ### Detailed Analysis Page
   
   - **Top section**: Slicers for filtering data
   
   - **Middle section**: Three column layout with:
     - Left: Bar chart comparing categories
     - Middle: Line chart showing performance over time
     - Right: Pie chart showing distribution
   
   - **Bottom section**: Matrix visual with drilldown capability
"""

        # Comparison page
        instructions += """
   ### Comparison Page
   
   - **Top section**: Time period slicer (this month, last month, YTD, etc.)
   
   - **Middle section**: Two column layout with:
     - Left: Current period metrics
     - Right: Prior period metrics
   
   - **Bottom section**: Small multiples visual showing trends across categories
"""

    if include_theme:
        instructions += """
6. **Apply custom theme**
   - Go to View > Themes and select "Professional" or "Executive"
   - Alternatively, create a custom theme with these colors:
     - Primary: #0078D4
     - Secondary: #50E6FF
     - Accent 1: #C239B3
     - Accent 2: #Bad80A
     - Accent 3: #FF8C00
     - Accent 4: #E81123
"""

    instructions += """
## Specific Visual Setup Examples

### Bar Chart
```
- Visual: Clustered Bar Chart
- Axis: [Category Column]
- Values: [Numeric Column]
- Legend: [Optional Category]
- Title: "Performance by Category"
```

### Line Chart
```
- Visual: Line Chart
- Axis: [Date Column]
- Values: [Numeric Column]
- Legend: [Optional Category]
- Title: "Trend Analysis"
```

### KPI Card
```
- Visual: Card
- Fields: [Measure]
- Title: "[Measure Name]"
```

## Power BI Best Practices

1. **Use bookmarks** for different view states
2. **Set up drill-through pages** for detailed analysis
3. **Use tooltips** to show additional context
4. **Apply conditional formatting** to highlight important values
5. **Create hierarchies** for better drill-down experience
6. **Use field parameters** for dynamic measure selection
7. **Organize fields** into display folders
8. **Use consistent formatting** across all visuals
9. **Add text boxes with instructions** for business users
10. **Create a mobile layout** for on-the-go access

## Questions?

For more information about Power BI best practices, visit:
https://docs.microsoft.com/en-us/power-bi/guidance/

"""
    return instructions