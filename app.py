import streamlit as st
st.set_page_config(
    page_title="SpreadsheetSage",
    page_icon="📊",
    layout="wide"
)
import pandas as pd
import numpy as np
import io
import os
from data_operations import filter_dataframe, sort_dataframe, get_sheet_names, process_excel_file
from visualization import create_chart
from prediction import run_prediction
from time_series import run_time_series_analysis
from ai_analysis import run_ai_analysis
from batch_processing import run_batch_processing
from pivot_tables import run_pivot_table
from powerbi_export import run_powerbi_export
from donation import add_donation_button, feature_suggestions
from utils import get_numeric_columns, get_object_columns, clean_data



def main():
    st.title("📊 Excel Analyzer")
    st.write("Upload, analyze, and visualize your Excel data easily!")

    # Add a super simple guide at the top
    st.info("""
    ### 👋 Welcome! Here's how to use this app:

    1. **Upload** your Excel file using the sidebar on the left
    2. Or **select** a sample dataset to explore
    3. **Click** on any tab to do different things with your data

    Each button and option has a simple explanation - just look for the 📌 help icons!
    """)

    # Check if user guide exists and show link to it
    if os.path.exists("user_guide.md"):
        with open("user_guide.md", "r") as f:
            user_guide = f.read()
        with st.expander("📘 Help: How to use this app"):
            st.markdown(user_guide)

    # Session state initialization
    if 'excel_file' not in st.session_state:
        st.session_state.excel_file = None
    if 'current_sheet' not in st.session_state:
        st.session_state.current_sheet = None
    if 'all_dfs' not in st.session_state:
        st.session_state.all_dfs = {}
    if 'sheet_names' not in st.session_state:
        st.session_state.sheet_names = []
    if 'ai_analysis_result' not in st.session_state:
        st.session_state.ai_analysis_result = None

    # Create tabs for main app sections
    main_tab1, main_tab2 = st.tabs(["Single File Analysis", "Batch Processing"])

    with main_tab1:
        # Add option to use example file
        #use_example = st.checkbox("", value=False)
        use_example = st.session_state["use_example"] = False

        if use_example:
            # Use the smaller sample file for faster loading
            example_file_path = "realistic_racing_data_sample.xlsx"
            # Fall back to the original file if sample doesn't exist
            if not os.path.exists(example_file_path):
                example_file_path = "realistic_racing_data.xlsx"

            if os.path.exists(example_file_path):
                if 'example_loaded' not in st.session_state or not st.session_state.example_loaded:
                    st.info(f"Loading example file: {example_file_path}")
                    # Load the example file
                    try:
                        with st.spinner("Processing example file..."):
                            # We need to use different approach for files on disk
                            example_dfs = {}
                            import openpyxl
                            workbook = openpyxl.load_workbook(example_file_path)
                            sheet_names = workbook.sheetnames

                            for sheet in sheet_names:
                                df = pd.read_excel(example_file_path, sheet_name=sheet)
                                example_dfs[sheet] = df

                            st.session_state.all_dfs = example_dfs
                            st.session_state.sheet_names = sheet_names
                            st.session_state.current_sheet = sheet_names[0] if sheet_names else None
                            st.session_state.excel_file = "example_file"
                            st.session_state.example_loaded = True

                            st.success(f"Successfully loaded example file with {len(sheet_names)} sheets!")
                    except Exception as e:
                        st.error(f"Error loading example file: {str(e)}")
                        st.session_state.example_loaded = False
            else:
                st.error(f"Example file not found: {example_file_path}")
        else:
            # Reset example loaded flag if checkbox is unchecked
            if 'example_loaded' in st.session_state and st.session_state.example_loaded:
                st.session_state.example_loaded = False
                st.session_state.excel_file = None
                st.session_state.all_dfs = {}
                st.session_state.sheet_names = []
                st.session_state.current_sheet = None

            # Regular file uploader
            uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"], key="single_file_uploader")

            if uploaded_file is not None:
                # Store the file in session state to prevent reloading
                if st.session_state.excel_file != uploaded_file:
                    st.session_state.excel_file = uploaded_file

                    # Convert Excel to optimized CSV format
                    try:
                        with st.spinner("Processing Excel file..."):
                            # Process Excel file - convert all sheets to CSV for better performance
                            st.session_state.all_dfs, st.session_state.sheet_names = process_excel_file(uploaded_file)

                            # Set default current sheet
                            if st.session_state.sheet_names:
                                st.session_state.current_sheet = st.session_state.sheet_names[0]

                            st.success(f"Successfully loaded and optimized {len(st.session_state.sheet_names)} sheets!")
                    except Exception as e:
                        st.error(f"Error loading Excel file: {str(e)}")
                        return

        # If we have loaded sheets, display them
        if st.session_state.excel_file is not None and st.session_state.current_sheet is not None:
            display_data_analysis()

    with main_tab2:
        # Run batch processing interface
        run_batch_processing()

def display_data_analysis():
    # Back to main button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔙 Back to Main", key="back_to_main_btn"):
            # Reset session state to return to main menu
            st.session_state.excel_file = None
            st.session_state.current_sheet = None
            st.session_state.all_dfs = {}
            st.session_state.sheet_names = []
            st.rerun()

    # Sheet selection
    sheet_name = st.selectbox(
        "Select a sheet to analyze:",
        st.session_state.sheet_names,
        index=st.session_state.sheet_names.index(st.session_state.current_sheet)
    )

    # Update current sheet if changed
    if sheet_name != st.session_state.current_sheet:
        st.session_state.current_sheet = sheet_name

    # Get the current dataframe
    df = st.session_state.all_dfs[sheet_name]

    # Add quick guide about navigation
    with st.expander("📌 How to use this interface"):
        st.markdown("""
        ### Quick Navigation Guide

        - **Click on any tab** above to access different analysis tools
        - **Use 'Back to Main'** button to return to the file upload screen
        - **Select different sheets** using the dropdown above
        - **Each tab has specific tools** for different types of analysis

        Need more help? Click on the built-in help expanders in each section!
        """)


    # Create tabs for different operations with clear descriptions
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "Data View", "Data Cleaning", "Data Operations", "Visualization", "Pivot Tables",
        "Prediction", "Time Series", "AI Analysis", "Power BI", "Export", "Support"
    ])

    # Add sidebar section with simplified explanation of each tab
    st.sidebar.markdown("## 🔍 What Each Tab Does")
    st.sidebar.info("""
    **Data View**: See your data and simple numbers about it

    **Data Cleaning**: Fix mistakes in your data like missing numbers

    **Data Operations**: Sort and filter your data to find what you want

    **Visualization**: Make charts and graphs to see patterns

    **Pivot Tables**: Group similar data together to see totals

    **Prediction**: Guess missing values based on patterns

    **Time Series**: See how things change over time and predict future values

    **AI Analysis**: Let the computer tell you interesting things about your data

    **Power BI**: Get your data ready for Microsoft Power BI

    **Export**: Save your work in different file types

    **Support**: Get help using the app or make a donation
    """)

    with tab1:
        st.subheader("Data Preview")

        # Simple explanation for non-technical users
        st.info("""
        📌 **What is this?**
        This shows your data table and tells you how big it is.
        Think of it like looking at the first few rows of your Excel sheet.
        """)

        st.write(f"**Sheet:** {sheet_name} | **Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

        # Display head with slider
        rows_to_show = st.slider("Number of rows to display", 5, min(100, df.shape[0]), 10,
                              help="Move this slider to see more or fewer rows of your data")
        st.dataframe(df.head(rows_to_show))

        # Basic statistics
        if st.checkbox("Show basic statistics", help="Click here to see averages, highest and lowest values in your data"):
            st.subheader("Basic Statistics")

            with st.expander("📌 What are these numbers?"):
                st.markdown("""
                These numbers tell you important facts about your data:

                * **count**: How many values you have (not counting missing ones)
                * **mean**: The average value
                * **std**: How spread out the values are
                * **min**: The smallest value
                * **25%**: 25% of values are smaller than this
                * **50%**: The middle value (half are smaller, half are larger)
                * **75%**: 75% of values are smaller than this
                * **max**: The largest value
                """)

            numeric_cols = get_numeric_columns(df)
            if numeric_cols:
                st.write(df[numeric_cols].describe())
            else:
                st.info("No numeric columns found for statistics.")

    with tab2:
        st.subheader("Data Cleaning")

        # Simple explanation for non-technical users
        st.info("""
        📌 **What is this?**
        This helps you fix problems in your data like missing values or incorrect formats.
        Think of it like cleaning up mistakes in your Excel sheet.
        """)

        with st.expander("📌 Data Cleaning Guide", expanded=True):
            st.markdown("""
            ### What You Can Do Here:

            1. **Find and fix missing values** - empty cells that need filling
            2. **Convert data types** - make sure numbers are treated as numbers
            3. **Remove duplicates** - get rid of repeated information
            4. **Fix text formatting** - make text consistent (uppercase/lowercase)

            After making changes, click the button below to save your cleaned data.
            """)

        # Use the data cleaning interface
        cleaned_df = clean_data(df)

        # Option to replace the current dataframe with cleaned data
        if st.button("Save cleaned data", help="Click this to save all your data cleaning changes"):
            st.session_state.all_dfs[sheet_name] = cleaned_df
            st.success("Your data has been cleaned and saved successfully!")
            st.rerun()

    with tab3:
        st.subheader("Data Operations")

        # Simple explanation for non-technical users
        st.info("""
        📌 **What is this?**
        This lets you find and organize your data in different ways.
        Think of it like using Excel's sort and filter buttons, but with more options.
        """)

        operation = st.radio(
            "Select operation:",
            ["Filter", "Sort"],
            help="Filter means only show certain rows, Sort means put rows in order"
        )

        if operation == "Filter":
            # Simple explanation for filtering
            with st.expander("📌 What is filtering?", expanded=True):
                st.markdown("""
                **Filtering** helps you find specific information in your data.

                For example, you might want to:
                * Only see horses that finished in the top 3
                * Only look at races from a certain year
                * Only include records with specific values

                The app will show you only the rows that match your filters.
                """)

            # Filter operation
            filtered_df = filter_dataframe(df)

            if st.checkbox("Show filtered data", help="Check this to see your filtered results"):
                st.write(f"Filtered data: {filtered_df.shape[0]} rows")
                st.dataframe(filtered_df)

                # Option to replace the current dataframe with filtered one
                if st.button("Save filtered data", help="Click to keep only the data that matches your filters"):
                    st.session_state.all_dfs[sheet_name] = filtered_df
                    st.success("Your filtered data has been saved! You're now only seeing the matching rows.")
                    st.rerun()

        elif operation == "Sort":
            # Simple explanation for sorting
            with st.expander("📌 What is sorting?", expanded=True):
                st.markdown("""
                **Sorting** arranges your data in order based on values in a column.

                For example, you might want to:
                * Order races from fastest to slowest time
                * Sort horses alphabetically by name
                * Arrange data from highest to lowest value

                The app will rearrange the rows in the order you choose.
                """)

            # Sort operation
            sorted_df = sort_dataframe(df)

            if st.checkbox("Show sorted data", help="Check this to see your data in the new order"):
                st.write("Sorted data:")
                st.dataframe(sorted_df)

                # Option to replace the current dataframe with sorted one
                if st.button("Save sorted data", help="Click to keep your data in the sorted order"):
                    st.session_state.all_dfs[sheet_name] = sorted_df
                    st.success("Your data has been sorted and saved in the new order!")
                    st.rerun()

    with tab4:
        st.subheader("Data Visualization")

        # Simple explanation for non-technical users
        st.info("""
        📌 **What is this?**
        This turns your numbers into pictures like charts and graphs.
        Think of it like making colorful diagrams to show patterns in your data.
        """)

        with st.expander("📌 Chart Types Guide", expanded=True):
            st.markdown("""
            ### Chart Types Explained:

            **Bar Chart** - Shows comparison between categories with bars (taller bars = bigger numbers)

            **Line Chart** - Shows how numbers change over time (good for trends)

            **Scatter Plot** - Shows relationships between two things (do they move together?)

            **Pie Chart** - Shows parts of a whole (like slices of a pie)

            **Heatmap** - Uses colors to show high and low values (dark colors = bigger numbers)

            Choose the chart that best shows what you want to see in your data!
            """)

        create_chart(df)

    with tab5:
        st.subheader("Pivot Table Analysis")

        # Simple explanation for non-technical users
        st.info("""
        📌 **What is this?**
        This groups your data to show totals, averages, or counts.
        Think of it like organizing things into categories to see patterns.
        """)

        with st.expander("📌 Understanding Pivot Tables", expanded=True):
            st.markdown("""
            ### What is a Pivot Table?

            A pivot table helps you organize and summarize your data. It's like sorting your data into neat boxes.

            **Example:** If you have race results data with 100 races:

            - You could group by **Jockey** to see how many races each jockey won
            - You could group by **Year** to count races in each year
            - You could group by **Track** to find the average race time at each location

            ### How to use this tool:

            1. Pick the column you want to group by (like "Jockey" or "Year")
            2. Pick what values you want to see (like "Race Time" or "Position")
            3. Choose how to calculate (Sum, Average, Count, etc.)
            4. View your results!
            """)

        run_pivot_table(df)

    with tab6:
        st.subheader("Predictive Analysis")

        # Simple explanation for non-technical users
        st.info("""
        📌 **What is this?**
        This tries to predict missing values based on patterns in your data.
        Think of it like a smart calculator that can guess answers based on your existing data.
        """)

        with st.expander("📌 How Prediction Works", expanded=True):
            st.markdown("""
            ### What Does Prediction Do?

            Prediction looks at patterns in your data to make educated guesses about missing values.

            **Example:** If you have horse racing data:

            - You could predict a horse's **finish time** based on its past performance
            - You could guess the **position** a horse might finish based on other factors
            - You could predict if a horse will **win or lose** based on track conditions

            ### Two Main Types of Prediction:

            **Regression** - Predicts a number (like finish time, speed, or points)

            **Classification** - Predicts a category (like win/lose, 1st/2nd/3rd place)

            The computer looks at all the patterns in your data, then makes its best guess!
            """)

        run_prediction(df)

    with tab7:
        st.subheader("Time Series Forecasting")

        # Simple explanation for non-technical users
        st.info("""
        📌 **What is this?**
        This shows how your data changes over time and predicts future values.
        Think of it like a weather forecast, but for your numbers!
        """)

        with st.expander("📌 Understanding Time Series", expanded=True):
            st.markdown("""
            ### What is Time Series Forecasting?

            Time Series analysis looks at data that changes over time (like daily race times or monthly sales)
            and tries to predict what will happen in the future.

            **For example:**
            - If you have **race times** from 2023-2025, you could predict race times for 2026
            - If you have **monthly results** for several years, you can see seasonal patterns
            - If you have **daily performance data**, you can spot trends over time

            ### How it Works:

            1. You select a date/time column from your data
            2. You choose what value you want to predict
            3. The app finds patterns in your past data
            4. The app makes predictions about future values

            It's perfect for racing data where you want to see if performance is improving or declining over time!
            """)

        run_time_series_analysis(df)

    with tab8:
        st.subheader("AI-Powered Analysis")

        # Simple explanation for non-technical users
        st.info("""
        📌 **What is this?**
        This uses artificial intelligence to understand and explain your data.
        Think of it like having a super-smart assistant look at your numbers and tell you what's interesting!
        """)

        with st.expander("📌 How AI Analysis Works", expanded=True):
            st.markdown("""
            ### What AI Analysis Does

            The AI (artificial intelligence) looks at your data and:

            1. **Finds patterns** you might not notice on your own
            2. **Explains trends** in simple language
            3. **Suggests insights** that could be helpful
            4. **Answers questions** about your data

            ### Example Use Cases:

            - **Racing data analysis**: Find which factors most affect race outcomes
            - **Seasonal patterns**: Discover if certain months have better results
            - **Performance drivers**: Identify what makes the biggest difference in results
            - **Story creation**: Generate an easy-to-understand story about your data

            ### How to Use It:

            1. Select what you want the AI to analyze
            2. Ask specific questions if you have them
            3. Review the insights it discovers

            This feature requires an internet connection to access the AI service.
            """)

        run_ai_analysis(df)

    with tab9:
        st.subheader("Power BI Integration")

        # Simple explanation for non-technical users
        st.info("""
        📌 **What is this?**
        This prepares your data to work with Power BI (Microsoft's data visualization tool).
        Think of it like packaging your data so it looks great in another program!
        """)

        with st.expander("📌 About Power BI Export", expanded=True):
            st.markdown("""
            ### What is Power BI?

            Power BI is Microsoft's tool for creating interactive dashboards and reports.
            This feature helps you prepare your data to work perfectly in Power BI.

            ### What This Feature Does:

            1. **Optimizes your data** for Power BI's format
            2. **Creates relationships** between different parts of your data
            3. **Sets up templates** so your reports look great instantly
            4. **Suggests visualizations** that work well with your specific data

            ### How to Use It:

            1. Choose optimization options for your data
            2. Select columns to include in your reports
            3. Download the prepared files
            4. Open them in Power BI Desktop

            The tool creates everything you need to get started with beautiful dashboards!
            """)

        run_powerbi_export(df)

    with tab10:
        st.subheader("Export Data")

        # Simple explanation for non-technical users
        st.info("""
        📌 **What is this?**
        This lets you save your processed data to use in other programs.
        Think of it like saving your work to share with friends or use later!
        """)

        with st.expander("📌 About Data Export", expanded=True):
            st.markdown("""
            ### Why Export Your Data?

            After you've analyzed, filtered, or transformed your data, you might want to:

            1. **Save your work** to continue later
            2. **Share with others** who don't have this app
            3. **Use in other programs** like Excel or Google Sheets

            ### Available Export Formats:

            - **CSV file**: Works with almost any program
            - **Excel file**: Perfect for Microsoft Excel users
            - **JSON file**: Good for web developers or technical users

            ### How to Use:

            1. Select the format you want
            2. Choose what parts of the data to include
            3. Click download
            4. Save the file to your computer

            It's that simple!
            """)

        export_format = st.selectbox(
            "Select export format:",
            ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"]
        )

    with tab11:
        st.subheader("Support & Suggestions")

        # Simple explanation for non-technical users
        st.info("""
        📌 **What is this section?**
        Here you can support the app by making a donation or suggesting new features.
        Think of it like helping to make the tool even better!
        """)

        support_tab1, support_tab2 = st.tabs(["Donate", "Suggest Features"])

        with support_tab1:
            with st.expander("📌 Why Support This Project?", expanded=True):
                st.markdown("""
                ### Why Donations Help

                The app uses AI technology (called GPT-4o) to analyze your data,
                which costs money each time it's used. Your donation helps:

                1. **Keep the AI features working** - every analysis costs a small amount
                2. **Add new features** - your support helps improve the app
                3. **Maintain the service** - keep the app available for everyone

                ### What You Get

                - **Continued access** to all the powerful features
                - **Future improvements** as they're developed
                - **Good karma** for supporting independent software!

                Every little bit helps - even a small donation makes a difference!
                """)

            add_donation_button()

        with support_tab2:
            st.write("We value your input! Have ideas for new features or improvements?")
            st.write("Let us know what would make Excel Analyzer even more useful for you.")

            feature_suggestions()

if __name__ == "__main__":
    main()
