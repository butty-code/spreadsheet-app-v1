import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def analyze_dataframe(df, context=""):
    """
    Use OpenAI's GPT-4o to analyze a dataframe and provide insights.
    
    Args:
        df: Pandas DataFrame to analyze
        context: Additional context about the data (optional)
        
    Returns:
        Dictionary containing analysis results
    """
    # Create a better sample - using more rows and stratified by key columns if they exist
    try:
        # Get a larger sample, stratified by key columns if possible
        if len(df) > 300:
            # Try to identify key categorical columns for stratification
            if 'horse_name' in df.columns and 'jockey' in df.columns:
                sample_df = pd.concat([
                    df.groupby('jockey', group_keys=False).apply(lambda x: x.sample(min(2, len(x)))),
                    df.sample(min(200, len(df)))
                ]).drop_duplicates().head(300)
            else:
                sample_df = df.sample(min(300, len(df)))
        else:
            sample_df = df
    except:
        # Fallback to simple random sampling
        sample_df = df.sample(min(300, len(df)))
    
    # Get dataframe info
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # Get column percentages for categorical columns
    column_stats = ""
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if len(df[col].unique()) < 20:  # Only for columns with reasonable number of categories
            value_counts = df[col].value_counts(normalize=True).mul(100).round(1).head(10)
            column_stats += f"\nDistribution of {col}:\n"
            column_stats += value_counts.to_string() + "\n"
    
    # Get basic statistics for numeric columns
    stats = df.describe().to_string()
    
    # Get column correlations for numeric data
    correlation_str = ""
    try:
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr().round(2)
            # Only include strong correlations
            strong_corr = corr_matrix[((corr_matrix > 0.5) | (corr_matrix < -0.5)) & (corr_matrix != 1.0)]
            if not strong_corr.empty:
                correlation_str = "\nStrong correlations between numeric columns:\n" + strong_corr.to_string()
    except:
        correlation_str = ""
    
    # Add specific context based on column names
    dataset_context = ""
    if any(col in df.columns for col in ['horse_name', 'jockey', 'trainer', 'racecourse']):
        dataset_context = """
        This appears to be a horse racing dataset containing information about races, horses, jockeys, and trainers.
        Focus your analysis on insights relevant to horse racing such as:
        - Performance patterns of jockeys and trainers
        - Race statistics by racecourse or race type
        - Correlations between factors like odds and race outcomes
        - Factors that might predict race performance
        Do not assume any data issues or problems unless clearly evident in the statistics provided.
        """
    
    # Create a prompt for the AI
    prompt = f"""
    I have a horse racing dataset with the following information:
    
    Summary:
    - Rows: {df.shape[0]}
    - Columns: {df.shape[1]}
    
    Column Information:
    {info_str}
    
    Column Value Distributions:
    {column_stats}
    
    Basic Numeric Statistics:
    {stats}
    
    {correlation_str}
    
    Sample of the data (first few rows):
    {sample_df.head(15).to_string()}
    
    Additional context provided: {context}
    {dataset_context}
    
    Please analyze this horse racing data and provide:
    1. A concise summary of what the data represents (focus on horse racing insights)
    2. Key insights or patterns you notice about jockeys, horses, racecourses, etc.
    3. Only mention data anomalies if there are clear issues (like missing values)
    4. Recommendations for further analysis that would be valuable for racing insights
    
    Format your response as a JSON object with the keys: 'summary', 'insights', 'anomalies', 'recommendations'.
    Each value should be an array of strings for easy display in a web interface.
    Make sure your analysis is specifically tailored to horse racing data, using proper terminology.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use the latest GPT-4o model
            messages=[
                {"role": "system", "content": "You are a horse racing data analysis expert helping to analyze Excel racing data. Provide clear, concise insights that would be helpful for racing fans and punters. Focus on race performance patterns, jockey and trainer success rates, and factors that influence race outcomes. Only mention data quality issues if they are significant and clearly evident in the data. Don't make assumptions about missing columns or features - analyze only what is present."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response as JSON
        content = response.choices[0].message.content
        if content:
            analysis = json.loads(content)
            return analysis
        else:
            return {
                "summary": ["Unable to complete AI analysis due to empty response."],
                "insights": [],
                "anomalies": [],
                "recommendations": []
            }
    
    except Exception as e:
        st.error(f"Error analyzing data with AI: {str(e)}")
        return {
            "summary": ["Unable to complete AI analysis. Please try again later."],
            "insights": [],
            "anomalies": [],
            "recommendations": []
        }

def run_ai_analysis(df):
    """
    Create UI for AI-powered data analysis.
    """
    st.markdown("## AI-Powered Analysis")
    st.write("Use OpenAI's GPT-4o to analyze your data and provide insights.")
    
    # Check if API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OpenAI API key is not configured. Please contact the administrator.")
        return
    
    # Create tabs for different AI analysis features
    tab1, tab2, tab3 = st.tabs(["Data Analysis", "Visualization Suggestions", "Data Story"])
    
    with tab1:
        st.info("This feature uses OpenAI's GPT-4o model to provide racing-specific insights from your data. The analysis will identify patterns in jockey, trainer and horse performance, and suggest further investigations.")
        
        # Additional context input for racing analysis
        context = st.text_area(
            "Additional racing context (optional):",
            help="Add any racing-specific information that might help with analysis. For example: 'Focus on flat races' or 'Analyze how track conditions affect performance'."
        )
        
        # Column selection for targeted analysis with racing-focused defaults
        racing_columns = [col for col in df.columns if col.lower() in ['horse_name', 'jockey', 'trainer', 'owner', 'race_distance', 'race_type', 'position', 'odds', 'racecourse']]
        
        if racing_columns:
            default_columns = racing_columns
        else:
            default_columns = []
            
        selected_columns = st.multiselect(
            "Select columns for analysis (optional):",
            options=df.columns.tolist(),
            default=default_columns,
            help="If left empty, all columns will be analyzed. For best results with racing data, include columns related to horses, jockeys, trainers and race conditions."
        )
        
        # Use the selected columns or all columns
        if selected_columns:
            analysis_df = df[selected_columns].copy()
        else:
            analysis_df = df.copy()
        
        # Button to trigger analysis
        if st.button("Run AI Analysis", type="primary"):
            with st.spinner("Analyzing horse racing data with OpenAI GPT-4o..."):
                result = analyze_dataframe(analysis_df, context)
                
                # Store the results in session state for later reference
                st.session_state.ai_analysis_result = result
                
                # Display the results in a visually appealing way with cards
                st.markdown("## AI Racing Data Analysis Results")
                
                # Summary in a success message box
                with st.expander("🏇 Data Summary", expanded=True):
                    for item in result.get("summary", []):
                        st.write(f"• {item}")
                
                # Insights in a info message box
                with st.expander("🔍 Key Racing Insights", expanded=True):
                    for item in result.get("insights", []):
                        st.write(f"• {item}")
                
                # Only show anomalies section if it contains items
                if result.get("anomalies", []):
                    with st.expander("⚠️ Data Quality Issues", expanded=False):
                        for item in result.get("anomalies", []):
                            st.write(f"• {item}")
                
                # Recommendations in an info message box
                with st.expander("💡 Recommended Analysis", expanded=True):
                    for item in result.get("recommendations", []):
                        st.write(f"• {item}")
                
                # Add download options for the analysis
                st.markdown("### Save Analysis Results")
                col1, col2 = st.columns(2)
                with col1:
                    json_result = json.dumps(result, indent=2)
                    st.download_button(
                        label="Download Analysis as JSON",
                        data=json_result,
                        file_name="racing_analysis_result.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Create a formatted markdown report
                    summary_md = ''.join(['* ' + item + '\n' for item in result.get("summary", [])])
                    insights_md = ''.join(['* ' + item + '\n' for item in result.get("insights", [])])
                    anomalies_md = ''.join(['* ' + item + '\n' for item in result.get("anomalies", [])])
                    recommendations_md = ''.join(['* ' + item + '\n' for item in result.get("recommendations", [])])
                    
                    markdown_report = f"# AI Analysis Report\n\n## Data Summary\n{summary_md}\n## Key Insights\n{insights_md}\n## Potential Issues & Anomalies\n{anomalies_md}\n## Recommendations\n{recommendations_md}"
                    st.download_button(
                        label="Download Report as Markdown",
                        data=markdown_report,
                        file_name="ai_analysis_report.md",
                        mime="text/markdown"
                    )
    
    with tab2:
        st.subheader("AI Visualization Recommendations")
        st.write("Get smart visualization suggestions based on your data patterns.")
        
        if st.button("Generate Visualization Suggestions"):
            with st.spinner("Analyzing data to recommend visualizations..."):
                viz_suggestions = suggest_visualizations(df)
                
                if viz_suggestions:
                    for i, suggestion in enumerate(viz_suggestions):
                        with st.expander(f"Suggestion {i+1}: {suggestion.get('type', 'Chart')}"):
                            st.markdown(f"**Chart Type:** {suggestion.get('type', 'Unknown')}")
                            st.markdown(f"**Recommended Columns:** {', '.join(suggestion.get('columns', []))}")
                            st.markdown(f"**Expected Insights:** {suggestion.get('insights', 'No insights provided')}")
                            
                            # Create a "Use This Suggestion" button to open visualization tab with pre-filled settings
                            if st.button(f"Apply This Suggestion", key=f"apply_viz_{i}"):
                                st.session_state.viz_suggestion = {
                                    'type': suggestion.get('type', ''),
                                    'columns': suggestion.get('columns', [])
                                }
                                st.info("Suggestion applied! Go to the Visualization tab to see the result.")
                                
                else:
                    st.warning("No visualization suggestions could be generated. Try with a different dataset.")
    
    with tab3:
        st.subheader("Generate Data Story")
        st.write("Create a narrative about your data for easy sharing with stakeholders.")
        
        if st.button("Generate Data Story"):
            with st.spinner("Creating data story with AI..."):
                # Use existing analysis result if available, otherwise generate new one
                if 'ai_analysis_result' in st.session_state:
                    story = generate_data_story(df, st.session_state.ai_analysis_result)
                else:
                    story = generate_data_story(df)
                
                st.markdown("### Data Story")
                st.markdown(story)
                
                # Add a download option for the story
                if story:
                    # Ensure story is a string and encode it
                    story_text = str(story)
                    st.download_button(
                        label="Download Story as Text",
                        data=story_text.encode('utf-8'),
                        file_name="data_story.txt",
                        mime="text/plain"
                    )

def generate_data_story(df, analysis_result=None):
    """
    Generate a narrative about the data based on AI analysis.
    
    Args:
        df: DataFrame containing the data
        analysis_result: Previous analysis result (optional)
        
    Returns:
        String containing the data story
    """
    # If no previous analysis, run a quick one
    if analysis_result is None:
        with st.spinner("Generating data story..."):
            analysis_result = analyze_dataframe(df)
    
    # Create a summarized view of the data
    summary_stats = {
        "row_count": df.shape[0],
        "column_count": df.shape[1],
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "missing_values": df.isna().sum().sum()
    }
    
    # Create a prompt for generating the data story
    prompt = f"""
    Based on the following dataset summary and analysis:
    
    Dataset Statistics:
    - Total rows: {summary_stats['row_count']}
    - Total columns: {summary_stats['column_count']}
    - Numeric columns: {', '.join(summary_stats['numeric_columns'][:5])}{"..." if len(summary_stats['numeric_columns']) > 5 else ""}
    - Categorical columns: {', '.join(summary_stats['categorical_columns'][:5])}{"..." if len(summary_stats['categorical_columns']) > 5 else ""}
    - Missing values: {summary_stats['missing_values']}
    
    Analysis Summary:
    {analysis_result.get('summary', ['No summary available'])[0]}
    
    Key Insights:
    {' '.join([f"- {insight}" for insight in analysis_result.get('insights', ['No insights available'])[:3]])}
    
    Please generate a coherent, engaging narrative that tells the story of this data. 
    Write it as if explaining the data to a business stakeholder who doesn't have technical knowledge.
    Focus on making the insights actionable and relevant.
    
    The narrative should be 2-3 paragraphs long and include:
    1. An introduction to what the data represents
    2. The key findings and patterns
    3. What these findings might mean for business decisions
    
    Be concise but informative.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data storytelling expert who translates complex data analysis into engaging narratives for business users."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error generating data story: {str(e)}")
        return "Unable to generate data story. Please try again later."

def suggest_visualizations(df):
    """
    Suggest appropriate visualizations based on the data.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of visualization suggestions with details
    """
    # Analyze column types
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Create a summary for the AI
    col_summary = {
        "numeric": numeric_cols[:5],
        "categorical": categorical_cols[:5],
        "datetime": date_cols[:5]
    }
    
    # Sample correlation data if there are numeric columns
    corr_data = ""
    if len(numeric_cols) >= 2:
        try:
            correlation = df[numeric_cols].corr().round(2)
            # Get top 3 highest correlations (absolute value)
            corr_pairs = []
            for i in range(len(correlation.columns)):
                for j in range(i+1, len(correlation.columns)):
                    if not np.isnan(correlation.iloc[i, j]):
                        corr_pairs.append((correlation.columns[i], correlation.columns[j], abs(correlation.iloc[i, j])))
            
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            top_corrs = corr_pairs[:3]
            
            corr_data = "Top correlations:\n"
            for col1, col2, val in top_corrs:
                corr_data += f"- {col1} and {col2}: {val}\n"
        except:
            corr_data = "Unable to calculate correlations."
    
    # Create prompt for the AI
    prompt = f"""
    I have a dataset with the following column types:
    
    Numeric columns: {', '.join(col_summary['numeric'])}{"..." if len(numeric_cols) > 5 else ""}
    Categorical columns: {', '.join(col_summary['categorical'])}{"..." if len(categorical_cols) > 5 else ""}
    Date/time columns: {', '.join(col_summary['datetime'])}{"..." if len(date_cols) > 5 else ""}
    
    {corr_data}
    
    Based on this information, suggest 3-5 effective visualizations that would help understand the data better.
    
    For each visualization, please provide:
    1. The type of chart/plot
    2. Which columns to use
    3. A brief explanation of what insights this visualization might reveal
    
    Format your response as a JSON object with a key "visualizations" that contains an array of objects.
    Each object should have the keys "type", "columns", and "insights".
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data visualization expert who knows how to choose the most effective visualizations for different types of data."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response as JSON
        content = response.choices[0].message.content
        if content:
            suggestions = json.loads(content)
            return suggestions.get("visualizations", [])
        else:
            return []
    
    except Exception as e:
        st.error(f"Error generating visualization suggestions: {str(e)}")
        return []