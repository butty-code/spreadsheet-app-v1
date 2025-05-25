import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from datetime import datetime, timedelta
from prophet import Prophet
from utils import get_numeric_columns, get_object_columns

def run_time_series_analysis(df):
    """Create UI for time series analysis and forecasting."""
    st.markdown("## Time Series Forecasting")
    st.write("Analyze trends and predict future values from your time series data.")
    
    # First, check if the dataframe has any date/time columns
    date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # If no datetime columns found, try to convert string columns that might contain dates
    if not date_columns:
        st.info("No date columns detected. Trying to identify date columns...")
        
        # Get object columns that might contain dates
        object_columns = get_object_columns(df)
        
        # Try to convert object columns to datetime
        potential_date_cols = []
        for col in object_columns:
            try:
                # Check a sample of values
                sample = df[col].dropna().head(10)
                # Use format='mixed' instead of deprecated infer_datetime_format
                pd.to_datetime(sample, format='mixed')
                potential_date_cols.append(col)
            except:
                continue
        
        if potential_date_cols:
            date_column = st.selectbox(
                "Select column containing dates:",
                options=potential_date_cols
            )
            
            try:
                # Convert to datetime
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                date_columns = [date_column]
                st.success(f"Successfully converted {date_column} to date format!")
            except Exception as e:
                st.error(f"Error converting column to date: {str(e)}")
                return
        else:
            st.warning("No columns with date-like values found. Time series analysis requires a date column.")
            return
    else:
        # If datetime columns are found, let user choose which one to use
        date_column = st.selectbox(
            "Select date column for time series analysis:",
            options=date_columns
        )
    
    # Get numeric columns for time series analysis
    numeric_columns = get_numeric_columns(df)
    
    if not numeric_columns:
        st.warning("No numeric columns found for time series analysis.")
        return
    
    # Let user select which numeric column to forecast
    target_column = st.selectbox(
        "Select the numeric column to forecast:",
        options=numeric_columns,
        help="This is the value you want to predict into the future"
    )
    
    # Provide some information about the date range
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    date_range = (max_date - min_date).days
    
    st.info(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({date_range} days)")
    
    # Create tabs for different steps
    tabs = st.tabs(["Data Preparation", "Exploration", "Forecasting", "Results"])
    
    with tabs[0]:
        st.markdown("### Data Preparation")
        
        # Prepare the data
        df_ts = df[[date_column, target_column]].copy()
        df_ts = df_ts.dropna()
        
        # Check for duplicates in the date column
        duplicates = df_ts[date_column].duplicated().sum()
        if duplicates > 0:
            st.warning(f"Found {duplicates} duplicate dates. Time series analysis works best with unique dates.")
            
            # Let user choose how to handle duplicates
            agg_method = st.selectbox(
                "How do you want to handle duplicate dates?",
                options=["Mean", "Sum", "Min", "Max", "First", "Last"],
                help="This determines how values with the same date will be combined"
            )
            
            # Apply aggregation
            agg_fn = agg_method.lower()
            df_ts = df_ts.groupby(date_column).agg({target_column: agg_fn}).reset_index()
            st.success(f"Aggregated duplicate dates using {agg_method} method.")
        
        # Sort by date
        df_ts = df_ts.sort_values(by=date_column)
        
        # Check for gaps in the time series
        date_diff = df_ts[date_column].diff().dt.days.dropna()
        
        if date_diff.max() > 1:
            st.warning(f"Found gaps in the time series. Largest gap is {date_diff.max()} days.")
            
            # Let user choose how to handle gaps
            gap_method = st.selectbox(
                "How do you want to handle gaps in the time series?",
                options=["Keep as is", "Resample to daily and fill NaN", "Resample to weekly and fill NaN", 
                         "Resample to monthly and fill NaN"],
                help="This determines how missing dates will be handled"
            )
            
            if gap_method != "Keep as is":
                # Set the date column as index
                df_ts = df_ts.set_index(date_column)
                
                # Resample and fill NaN
                if gap_method == "Resample to daily and fill NaN":
                    df_ts = df_ts.resample('D').asfreq()
                elif gap_method == "Resample to weekly and fill NaN":
                    df_ts = df_ts.resample('W').asfreq()
                elif gap_method == "Resample to monthly and fill NaN":
                    df_ts = df_ts.resample('M').asfreq()
                
                # Method to fill missing values
                fill_method = st.selectbox(
                    "Method to fill missing values in gaps:",
                    options=["Forward fill", "Backward fill", "Linear interpolation", "Leave as NaN"]
                )
                
                if fill_method == "Forward fill":
                    df_ts = df_ts.ffill()
                elif fill_method == "Backward fill":
                    df_ts = df_ts.bfill()
                elif fill_method == "Linear interpolation":
                    df_ts = df_ts.interpolate(method='linear')
                
                # Reset index to get date column back
                df_ts = df_ts.reset_index()
                
                st.success(f"Applied {gap_method} with {fill_method}.")
        
        # Show the prepared data
        st.subheader("Prepared Time Series Data")
        st.dataframe(df_ts.head(10))
        st.write(f"Total records: {len(df_ts)}")
        
        # Set up for advanced settings
        with st.expander("Advanced Settings"):
            # Option to log transform the data if it's skewed
            log_transform = st.checkbox("Apply log transformation", 
                                      help="Useful for data with exponential growth or high variance")
            
            # Option to remove outliers
            remove_outliers = st.checkbox("Remove outliers", 
                                        help="Detect and remove extreme values that might affect forecasting")
            
            if remove_outliers:
                outlier_threshold = st.slider("Outlier threshold (standard deviations)", 
                                           min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        
        # Apply transformations if selected
        if log_transform and df_ts[target_column].min() > 0:
            df_ts[f"{target_column}_log"] = np.log(df_ts[target_column])
            st.info(f"Created log-transformed column: {target_column}_log")
            # Update target for forecasting
            forecast_target = f"{target_column}_log"
        else:
            forecast_target = target_column
        
        if remove_outliers:
            # Calculate z-scores
            z_scores = np.abs((df_ts[forecast_target] - df_ts[forecast_target].mean()) / df_ts[forecast_target].std())
            df_ts_no_outliers = df_ts[z_scores < outlier_threshold].copy()
            
            st.info(f"Removed {len(df_ts) - len(df_ts_no_outliers)} outliers out of {len(df_ts)} records.")
            df_ts = df_ts_no_outliers
        
        # Store the prepared data in session state
        st.session_state.time_series_data = df_ts
        st.session_state.time_series_date_col = date_column
        st.session_state.time_series_target_col = target_column
        st.session_state.time_series_forecast_target = forecast_target
        
        if log_transform and forecast_target != target_column:
            st.session_state.time_series_log_transform = True
        else:
            st.session_state.time_series_log_transform = False
    
    with tabs[1]:
        st.markdown("### Exploratory Analysis")
        
        if 'time_series_data' not in st.session_state:
            st.warning("Please prepare your data in the Data Preparation tab first.")
            return
        
        df_ts = st.session_state.time_series_data
        date_column = st.session_state.time_series_date_col
        target_column = st.session_state.time_series_target_col
        forecast_target = st.session_state.time_series_forecast_target
        
        # Time series plot
        st.subheader("Time Series Plot")
        fig = px.line(df_ts, x=date_column, y=forecast_target, title=f"Time Series: {target_column}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal Decomposition
        if len(df_ts) >= 10:  # Need enough data points
            try:
                # Set date as index for statsmodels
                df_decompose = df_ts.set_index(date_column)
                
                # Determine frequency
                inferred_freq = pd.infer_freq(df_decompose.index)
                if inferred_freq is None:
                    # Try to determine a reasonable frequency
                    if len(df_decompose) < 50:
                        period = 7  # Weekly
                    else:
                        period = 30  # Monthly
                    st.warning(f"Could not infer frequency, using {period} as period for decomposition.")
                else:
                    # Convert frequency code to days
                    freq_map = {'D': 1, 'W': 7, 'M': 30, 'Q': 90, 'A': 365}
                    period = freq_map.get(inferred_freq[0], 7)
                    st.info(f"Inferred time series frequency: {inferred_freq}")
                
                # Seasonal decomposition
                decomposition = seasonal_decompose(df_decompose[forecast_target], model='additive', period=period)
                
                # Create plots
                trend = px.line(x=decomposition.trend.index, y=decomposition.trend.values, title="Trend Component")
                seasonal = px.line(x=decomposition.seasonal.index, y=decomposition.seasonal.values, title="Seasonal Component")
                residual = px.line(x=decomposition.resid.index, y=decomposition.resid.values, title="Residual Component")
                
                # Display plots in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(trend, use_container_width=True)
                    st.plotly_chart(residual, use_container_width=True)
                with col2:
                    st.plotly_chart(seasonal, use_container_width=True)
                    
                    # Show stats
                    st.subheader("Time Series Statistics")
                    st.write(f"Standard Deviation: {df_decompose[forecast_target].std():.2f}")
                    st.write(f"Seasonal Component Strength: {np.abs(decomposition.seasonal).mean():.2f}")
                    st.write(f"Trend Direction: {'Increasing' if decomposition.trend.iloc[-1] > decomposition.trend.iloc[0] else 'Decreasing'}")
                
                # Augmented Dickey-Fuller test for stationarity
                st.subheader("Stationarity Test (Augmented Dickey-Fuller)")
                result = adfuller(df_decompose[forecast_target].dropna())
                
                st.write(f"ADF Statistic: {result[0]:.3f}")
                st.write(f"p-value: {result[1]:.3f}")
                
                if result[1] < 0.05:
                    st.success("Time series is stationary (p < 0.05)")
                else:
                    st.warning("Time series is non-stationary (p > 0.05)")
                    st.info("Non-stationary series may need differencing for accurate forecasting.")
                
            except Exception as e:
                st.error(f"Error in seasonal decomposition: {str(e)}")
        else:
            st.warning("Need at least 10 data points for seasonal decomposition.")
            
    with tabs[2]:
        st.markdown("### Forecast Configuration")
        
        if 'time_series_data' not in st.session_state:
            st.warning("Please prepare your data in the Data Preparation tab first.")
            return
        
        df_ts = st.session_state.time_series_data
        date_column = st.session_state.time_series_date_col
        target_column = st.session_state.time_series_target_col
        forecast_target = st.session_state.time_series_forecast_target
        
        # Forecasting model selection
        st.subheader("Select Forecasting Model")
        
        model_options = {
            "Prophet": "Facebook's time series forecasting model (handles seasonality well)",
            "ARIMA": "Autoregressive Integrated Moving Average (classic statistical method)",
            "Exponential Smoothing": "Simple but effective for many time series"
        }
        
        # Create model selection with descriptions
        model_descriptions = [f"{model}: {desc}" for model, desc in model_options.items()]
        selected_model_with_desc = st.radio(
            "Select forecasting algorithm:",
            options=model_descriptions
        )
        model_type = selected_model_with_desc.split(":", 1)[0]
        
        # Forecast horizon (how far into the future)
        st.subheader("Forecast Horizon")
        
        # Determine data frequency and current year from the data
        date_diff = df_ts[date_column].diff().median()
        current_year = df_ts[date_column].max().year
        next_year = current_year + 1
        
        # Offer forecasting options based on data frequency
        forecast_option = st.radio(
            "Forecast by:",
            options=["Time periods", "Until specific date", "Future years"],
            index=2  # Default to Future years for year-over-year forecasting
        )
        
        if forecast_option == "Time periods":
            # Determine appropriate time units
            if date_diff.days < 2:  # Daily data
                default_horizon = 30
                horizon_unit = "days"
                max_periods = 365  # Allow up to a year
            elif date_diff.days < 8:  # Weekly data
                default_horizon = 12
                horizon_unit = "weeks"
                max_periods = 52  # Allow up to a year
            else:  # Monthly or longer
                default_horizon = 6
                horizon_unit = "months"
                max_periods = 24  # Allow up to two years
            
            forecast_periods = st.slider(
                f"Number of {horizon_unit} to forecast:",
                min_value=1,
                max_value=max_periods,
                value=default_horizon
            )
            
            # Convert forecast periods to days for internal calculations
            if horizon_unit == "days":
                forecast_days = forecast_periods
            elif horizon_unit == "weeks":
                forecast_days = forecast_periods * 7
            else:  # months
                forecast_days = forecast_periods * 30
        
        elif forecast_option == "Until specific date":
            # Allow user to specify an end date
            last_date = df_ts[date_column].max()
            max_future_date = last_date + timedelta(days=730)  # Allow up to 2 years ahead
            
            forecast_end_date = st.date_input(
                "Forecast until date:",
                value=last_date + timedelta(days=180),
                min_value=last_date + timedelta(days=1),
                max_value=max_future_date
            )
            
            # Calculate days between last date and forecast end date
            forecast_days = (forecast_end_date - last_date.date()).days
            
            # Show user how many days/months/years this represents
            if forecast_days < 30:
                st.info(f"Forecasting {forecast_days} days ahead")
            elif forecast_days < 365:
                st.info(f"Forecasting approximately {forecast_days // 30} months ahead")
            else:
                st.info(f"Forecasting approximately {forecast_days / 365:.1f} years ahead")
        
        else:  # Future years
            # Allow forecasting for specific future years (e.g., 2025)
            years_to_forecast = st.slider(
                "Number of years to forecast:",
                min_value=1,
                max_value=5,  # Allow up to 5 years into the future
                value=1
            )
            
            # Show which years will be forecasted
            year_list = [current_year + i for i in range(1, years_to_forecast + 1)]
            st.info(f"Forecasting for years: {', '.join(map(str, year_list))}")
            
            # Calculate days needed for the forecast (approximate)
            last_date = df_ts[date_column].max()
            forecast_end_date = datetime(current_year + years_to_forecast, 12, 31)
            forecast_days = (forecast_end_date - last_date).days
            
            # Ensure at least 30 days of forecast
            forecast_days = max(forecast_days, 30)
        
        # Model specific parameters
        if model_type == "ARIMA":
            st.subheader("ARIMA Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("p (AR order)", min_value=0, max_value=5, value=1)
            with col2:
                d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
            with col3:
                q = st.number_input("q (MA order)", min_value=0, max_value=5, value=1)
                
        elif model_type == "Prophet":
            st.subheader("Prophet Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                yearly_seasonality = st.selectbox("Yearly Seasonality", 
                                                 options=["auto", "True", "False"], 
                                                 index=0)
                if yearly_seasonality == "auto":
                    yearly_seasonality = 'auto'
                else:
                    yearly_seasonality = yearly_seasonality == "True"
                    
            with col2:
                weekly_seasonality = st.selectbox("Weekly Seasonality", 
                                                 options=["auto", "True", "False"], 
                                                 index=0)
                if weekly_seasonality == "auto":
                    weekly_seasonality = 'auto'
                else:
                    weekly_seasonality = weekly_seasonality == "True"
            
            daily_seasonality = st.selectbox("Daily Seasonality", 
                                          options=["auto", "True", "False"], 
                                          index=0)
            if daily_seasonality == "auto":
                daily_seasonality = 'auto'
            else:
                daily_seasonality = daily_seasonality == "True"
                
            include_holidays = st.checkbox("Include holiday effects", value=True)
            
        elif model_type == "Exponential Smoothing":
            st.subheader("Exponential Smoothing Parameters")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                trend = st.selectbox("Trend type", options=["None", "Add", "Mul"], index=1)
            with col2:
                seasonal = st.selectbox("Seasonal type", options=["None", "Add", "Mul"], index=1)
            with col3:
                seasonal_periods = st.number_input("Seasonal periods", min_value=0, max_value=52, value=7)
        
        # Button to run forecast
        forecast_button = st.button("Generate Forecast", type="primary")
        
        if forecast_button:
            with st.spinner(f"Generating forecast with {model_type}..."):
                try:
                    # Set up last date from the data
                    last_date = df_ts[date_column].max()
                    
                    # Create future dates for forecasting
                    future_dates = pd.date_range(
                        start=last_date + timedelta(days=1),
                        periods=forecast_days,
                        freq='D'
                    )
                    
                    # Prepare for results
                    forecast_results = pd.DataFrame({date_column: future_dates})
                    
                    # Forecasting with Prophet
                    if model_type == "Prophet":
                        # Prepare data for Prophet
                        prophet_data = df_ts[[date_column, forecast_target]].rename(
                            columns={date_column: 'ds', forecast_target: 'y'}
                        )
                        
                        # Create and fit model
                        model = Prophet(
                            yearly_seasonality=yearly_seasonality,
                            weekly_seasonality=weekly_seasonality,
                            daily_seasonality=daily_seasonality
                        )
                        
                        if include_holidays:
                            model.add_country_holidays(country_name='US')
                            
                        model.fit(prophet_data)
                        
                        # Create future dataframe
                        future = model.make_future_dataframe(periods=forecast_days)
                        
                        # Generate forecast
                        forecast = model.predict(future)
                        
                        # Extract predictions
                        combined_results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                        combined_results = combined_results.rename(
                            columns={'ds': date_column, 'yhat': f'{target_column}_Forecast',
                                    'yhat_lower': f'{target_column}_Lower', 'yhat_upper': f'{target_column}_Upper'}
                        )
                        
                    # Forecasting with ARIMA
                    elif model_type == "ARIMA":
                        # Prepare data for ARIMA
                        arima_data = df_ts[[date_column, forecast_target]].set_index(date_column)
                        
                        # Create and fit model
                        model = ARIMA(arima_data[forecast_target], order=(p, d, q))
                        model_fit = model.fit()
                        
                        # Generate forecast
                        forecast = model_fit.forecast(steps=forecast_days)
                        
                        # Create results dataframe
                        forecast_results[f'{target_column}_Forecast'] = forecast.values
                        
                        # Add historical data
                        historical = df_ts[[date_column, forecast_target]].rename(
                            columns={forecast_target: f'{target_column}_Forecast'}
                        )
                        
                        combined_results = pd.concat([historical, forecast_results])
                        
                    # Forecasting with Exponential Smoothing
                    elif model_type == "Exponential Smoothing":
                        # Prepare data
                        ets_data = df_ts[[date_column, forecast_target]].set_index(date_column)
                        
                        # Map trend and seasonal parameters
                        trend_map = {'None': None, 'Add': 'add', 'Mul': 'mul'}
                        trend_type = trend_map[trend]
                        seasonal_type = trend_map[seasonal]
                        
                        # Create and fit model
                        model = sm.tsa.ExponentialSmoothing(
                            ets_data[forecast_target],
                            trend=trend_type,
                            seasonal=seasonal_type,
                            seasonal_periods=seasonal_periods if seasonal_type else None
                        ).fit()
                        
                        # Generate forecast
                        forecast = model.forecast(forecast_days)
                        
                        # Create results dataframe
                        forecast_results[f'{target_column}_Forecast'] = forecast.values
                        
                        # Add historical data
                        historical = df_ts[[date_column, forecast_target]].rename(
                            columns={forecast_target: f'{target_column}_Forecast'}
                        )
                        
                        combined_results = pd.concat([historical, forecast_results])
                    
                    # Inverse transform if log was applied
                    if st.session_state.time_series_log_transform:
                        if f'{target_column}_Forecast' in combined_results.columns:
                            combined_results[f'{target_column}_Forecast'] = np.exp(combined_results[f'{target_column}_Forecast'])
                        if f'{target_column}_Lower' in combined_results.columns:
                            combined_results[f'{target_column}_Lower'] = np.exp(combined_results[f'{target_column}_Lower'])
                        if f'{target_column}_Upper' in combined_results.columns:
                            combined_results[f'{target_column}_Upper'] = np.exp(combined_results[f'{target_column}_Upper'])
                    
                    # Store results in session state
                    st.session_state.time_series_forecast_results = combined_results
                    st.session_state.time_series_forecast_model = model_type
                    st.session_state.time_series_forecast_days = forecast_days
                    
                    st.success(f"Forecast generated successfully! Go to the Results tab to view it.")
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
    
    with tabs[3]:
        st.markdown("### Forecast Results")
        
        if 'time_series_forecast_results' not in st.session_state:
            st.warning("Please generate a forecast in the Forecasting tab first.")
            return
        
        combined_results = st.session_state.time_series_forecast_results
        date_column = st.session_state.time_series_date_col
        target_column = st.session_state.time_series_target_col
        model_type = st.session_state.time_series_forecast_model
        forecast_days = st.session_state.time_series_forecast_days
        
        # Display forecast results
        st.subheader(f"Forecast Results ({model_type})")
        
        # Split data into historical and future
        last_historical_date = combined_results[date_column].iloc[len(combined_results) - forecast_days - 1]
        
        historical_data = combined_results[combined_results[date_column] <= last_historical_date].copy()
        future_data = combined_results[combined_results[date_column] > last_historical_date].copy()
        
        # Create a more informative visualization
        if model_type == "Prophet" and {'yhat_lower', 'yhat_upper'}.issubset(combined_results.columns):
            # If we have confidence intervals
            fig = go.Figure()
            
            # Historical data points
            fig.add_trace(go.Scatter(
                x=historical_data[date_column], 
                y=historical_data[f'{target_column}_Forecast'],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue')
            ))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=future_data[date_column], 
                y=future_data[f'{target_column}_Forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='red')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=future_data[date_column].tolist() + future_data[date_column].tolist()[::-1],
                y=future_data[f'{target_column}_Upper'].tolist() + future_data[f'{target_column}_Lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(231,107,243,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
            
        else:
            # Simple forecast line
            fig = go.Figure()
            
            # Historical data points
            fig.add_trace(go.Scatter(
                x=historical_data[date_column], 
                y=historical_data[f'{target_column}_Forecast'],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue')
            ))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=future_data[date_column], 
                y=future_data[f'{target_column}_Forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='red')
            ))
        
        # Get unique years in the future data
        future_years = pd.DatetimeIndex(future_data[date_column]).year.unique()
        future_years_str = ", ".join(map(str, future_years))
        
        fig.update_layout(
            title=f"Forecast: {target_column} for future years ({future_years_str})",
            xaxis_title="Date",
            yaxis_title=target_column,
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Add a vertical line to distinguish historical from future data
        fig.add_shape(
            type="line",
            x0=last_historical_date,
            y0=0,
            x1=last_historical_date,
            y1=1,
            yref="paper",
            line=dict(
                color="gray",
                width=2,
                dash="dash",
            )
        )
        
        # Add annotation
        fig.add_annotation(
            x=last_historical_date,
            y=0.95,
            yref="paper",
            text="Forecast Start",
            showarrow=True,
            arrowhead=1,
            ax=30,
            ay=-30
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast stats
        st.subheader("Forecast Statistics")
        future_min = future_data[f'{target_column}_Forecast'].min()
        future_max = future_data[f'{target_column}_Forecast'].max()
        future_mean = future_data[f'{target_column}_Forecast'].mean()
        future_total = future_data[f'{target_column}_Forecast'].sum()
        
        cols = st.columns(4)
        cols[0].metric("Minimum", f"{future_min:.2f}")
        cols[1].metric("Maximum", f"{future_max:.2f}")
        cols[2].metric("Average", f"{future_mean:.2f}")
        cols[3].metric("Total", f"{future_total:.2f}")
        
        # Display forecast data
        st.subheader("Forecast Data")
        
        # Add a year column for clarity
        future_display = future_data.copy()
        future_display['Year'] = pd.DatetimeIndex(future_display[date_column]).year
        future_display['Month'] = pd.DatetimeIndex(future_display[date_column]).month
        
        # Group by year and month to show yearly/monthly patterns
        if st.checkbox("Show yearly summary", value=True):
            yearly_summary = future_display.groupby('Year')[f'{target_column}_Forecast'].agg(['mean', 'min', 'max', 'sum']).reset_index()
            yearly_summary.columns = ['Year', 'Average', 'Minimum', 'Maximum', 'Total']
            
            # Format columns
            for col in ['Average', 'Minimum', 'Maximum', 'Total']:
                yearly_summary[col] = yearly_summary[col].round(2)
            
            st.subheader("Yearly Summary (2025+)")
            # Highlight 2025+ years
            def highlight_future_years(row):
                if row['Year'] >= 2025:
                    return ['background-color: #c2f0c2'] * len(row)
                return [''] * len(row)
            
            # Apply styling
            st.dataframe(yearly_summary.style.apply(highlight_future_years, axis=1))
        
        # Show raw forecast data
        st.subheader("Detailed Forecast Data")
        st.dataframe(future_display)
        
        # Export options
        st.subheader("Export Forecast")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export to CSV"):
                tmp_download_link = download_csv(future_data, f"{target_column}_forecast.csv")
                st.markdown(tmp_download_link, unsafe_allow_html=True)
        
        with col2:
            if st.button("Add Forecast to Original Data"):
                # Merge forecast back to the original dataframe
                # This is a simplistic approach; in a real app would need more sophistication
                st.session_state.forecast_added_to_df = future_data
                st.success("Forecast data has been stored in session. You can access it in the main app.")

def download_csv(df, filename):
    """Generate a download link for a dataframe."""
    import base64
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href