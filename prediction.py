import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
import statsmodels.api as sm
from utils import get_numeric_columns, get_object_columns

# Create directory for saved models if it doesn't exist
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

def run_prediction(df):
    """Create UI for advanced predictive modeling."""
    # Initialize session state for saved models if not exists
    if 'saved_models' not in st.session_state:
        # Load saved models if file exists
        if os.path.exists("saved_models/saved_models.json"):
            try:
                with open("saved_models/saved_models.json", "r") as f:
                    st.session_state.saved_models = json.load(f)
            except:
                st.session_state.saved_models = {}
        else:
            st.session_state.saved_models = {}
            
    # Professional UI header with styling
    st.markdown("""
    ## Professional Predictive Analytics
    Create sophisticated prediction models from your data with advanced analytics
    """)
    
    # Get column lists by type
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_object_columns(df)
    
    if len(numeric_columns) < 2:
        st.warning("⚠️ Prediction requires at least two numeric columns. Not enough numeric columns found.")
        return
    
    # Enhanced model options with explanations  
    col1, col2 = st.columns([3, 1])
    with col1:
        model_options = {
            "Regression (predict a number)": "Forecast continuous values like sales, prices, or ratings",
            "Classification (predict a category)": "Predict categories like Yes/No, High/Medium/Low, or customer segments"
        }
        
        # Create model type radio with descriptions
        model_descriptions = [f"{model}: {desc}" for model, desc in model_options.items()]
        selected_model_with_desc = st.radio(
            "Select prediction type:",
            options=model_descriptions
        )
        model_type = selected_model_with_desc.split(":", 1)[0]
    
    with col2:
        # Option to load saved model
        load_saved = st.checkbox("Load saved model")
        
        if load_saved and st.session_state.saved_models:
            saved_model_names = list(st.session_state.saved_models.keys())
            if len(saved_model_names) > 0:
                selected_model = st.selectbox("Select model:", saved_model_names)
                st.success(f"Loaded model: {selected_model}")
                # Model will be loaded in the respective function
    
    # Run the appropriate prediction workflow
    if model_type == "Regression (predict a number)":
        run_regression_prediction(df, numeric_columns, categorical_columns, load_saved)
    else:
        run_classification_prediction(df, numeric_columns, categorical_columns, load_saved)

def run_regression_prediction(df, numeric_columns, categorical_columns, load_saved=False):
    """Run enhanced regression prediction workflow with advanced models and visualizations."""
    st.markdown("### Professional Regression Analysis")
    
    # Create tabs for different steps of the workflow
    tabs = st.tabs(["Data Selection", "Model Configuration", "Results & Analysis", "Prediction"])
    
    with tabs[0]:
        st.markdown("#### Select Data for Your Model")
        # Target selection with better UI
        target_column = st.selectbox(
            "Select the column to predict (target):",
            numeric_columns,
            help="This is the value you want your model to predict"
        )
        
        # Feature selection with interactive UI
        available_features = [col for col in numeric_columns if col != target_column]
        
        if not available_features:
            st.warning("⚠️ No numeric features available for prediction after selecting target.")
            return
        
        # Option to include categorical features with one-hot encoding
        include_categorical = st.checkbox("Include categorical features", value=True, 
                                        help="Convert text/category columns to numbers automatically")
        
        # Feature selection with search and info
        st.markdown("**Select features for your prediction model:**")
        feature_col1, feature_col2 = st.columns([3, 1])
        
        with feature_col1:
            selected_features = st.multiselect(
                "Numeric features:",
                available_features,
                default=available_features[:min(3, len(available_features))]
            )
            
        with feature_col2:
            # Quick selection buttons
            if st.button("Select All"):
                selected_features = available_features
                st.rerun()
                
        if include_categorical and categorical_columns:
            selected_categorical = st.multiselect(
                "Categorical features:",
                categorical_columns,
                default=[]
            )
        else:
            selected_categorical = []
        
        if not selected_features and not selected_categorical:
            st.warning("⚠️ Please select at least one feature for prediction.")
            return
            
        # Data split preview with helpful explanation
        st.markdown("##### Data Splitting")
        st.info("Your data will be split into training data (to build the model) and testing data (to verify accuracy)")
        
        # Train-test split ratio with visual indicator
        col1, col2 = st.columns([3, 1])
        with col1:
            test_size = st.slider(
                "Test set size (%):",
                min_value=10,
                max_value=50,
                value=20,
                step=5
            ) / 100
        
        with col2:
            # Visual representation of the split
            st.markdown(f"Training: **{int((1-test_size)*100)}%**")
            st.markdown(f"Testing: **{int(test_size*100)}%**")
    
    with tabs[1]:
        st.markdown("#### Configure Your Prediction Model")
        
        # Model selection with descriptions
        model_options = {
            "Linear Regression": "Simple, fast model for clear relationships",
            "Ridge Regression": "Linear model with reduced overfitting",
            "Lasso Regression": "Linear model with feature selection",
            "Random Forest": "Robust ensemble model for complex data",
            "Gradient Boosting": "Advanced model for highest accuracy",
            "XGBoost": "State-of-the-art algorithm for tabular data",
            "Support Vector Machine": "Works well for smaller datasets"
        }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # Create model selection with descriptions
            model_descriptions = [f"{model}: {desc}" for model, desc in model_options.items()]
            selected_model_with_desc = st.radio(
                "Select algorithm:",
                options=model_descriptions
            )
            model_algorithm = selected_model_with_desc.split(":", 1)[0]
            
        with col2:
            # Model complexity/hyperparameter adjustment
            st.markdown("##### Model Settings")
            
            if model_algorithm == "Linear Regression":
                pass  # No hyperparameters for basic linear regression
                
            elif model_algorithm == "Ridge Regression":
                alpha = st.slider("Regularization strength:", 0.01, 10.0, 1.0, 0.1,
                                 help="Higher values reduce model complexity")
                
            elif model_algorithm == "Lasso Regression":
                alpha = st.slider("Regularization strength:", 0.01, 10.0, 1.0, 0.1,
                                 help="Higher values increase feature selection")
                
            elif model_algorithm == "Random Forest":
                n_estimators = st.slider("Number of trees:", 10, 200, 100, 10,
                                       help="More trees = better accuracy but slower")
                max_depth = st.slider("Maximum tree depth:", 2, 20, 10, 1,
                                    help="Deeper trees can overfit")
                
            elif model_algorithm == "Gradient Boosting":
                n_estimators = st.slider("Number of stages:", 10, 200, 100, 10)
                learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1, 0.01,
                                       help="Lower values need more trees but may generalize better")
                
            elif model_algorithm == "XGBoost":
                n_estimators = st.slider("Number of boosting rounds:", 10, 500, 100, 10)
                learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1, 0.01)
                max_depth = st.slider("Maximum tree depth:", 3, 10, 6, 1)
                
            elif model_algorithm == "Support Vector Machine":
                kernel = st.selectbox("Kernel type:", ["linear", "rbf", "poly"])
                C = st.slider("Regularization parameter:", 0.1, 10.0, 1.0, 0.1,
                            help="Lower values = stronger regularization")
        
        # Cross-validation for more reliable results
        cross_validation = st.checkbox("Use cross-validation", value=True,
                                     help="More reliable performance assessment but slower")
        
        if cross_validation:
            cv_folds = st.slider("Number of cross-validation folds:", 3, 10, 5, 1)
        
        # Create a name for saving the model
        save_model = st.checkbox("Save model for future use", value=False)
        if save_model:
            model_name = st.text_input("Model name:", value=f"{model_algorithm} for {target_column}")
    
    # Train button in a third tab
    with tabs[2]:
        train_model = st.button("Train Model", type="primary")
        
        if train_model:
            try:
                # Prepare data
                X = df[selected_features]
                y = df[target_column]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Alert for training
                samples_train = len(X_train)
                samples_test = len(X_test)
                st.info(f"Training model with {samples_train} samples, testing with {samples_test} samples.")
                
                # Initialize and train model based on selected algorithm
                if model_algorithm == "Linear Regression":
                    model = LinearRegression()
                    model_name = "Linear Regression"
                    has_feature_importance = False
                elif model_algorithm == "Ridge Regression":
                    model = Ridge(alpha=alpha)
                    model_name = "Ridge Regression"
                    has_feature_importance = False
                elif model_algorithm == "Lasso Regression":
                    model = Lasso(alpha=alpha)
                    model_name = "Lasso Regression"
                    has_feature_importance = False
                elif model_algorithm == "Random Forest":
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    model_name = "Random Forest Regression"
                    has_feature_importance = True
                elif model_algorithm == "Gradient Boosting":
                    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                    model_name = "Gradient Boosting Regression"
                    has_feature_importance = True
                elif model_algorithm == "XGBoost":
                    model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                    model_name = "XGBoost Regression"
                    has_feature_importance = True
                elif model_algorithm == "Support Vector Machine":
                    model = SVR(kernel=kernel, C=C)
                    model_name = "Support Vector Regression"
                    has_feature_importance = False
                
                # Fit the model with progress indication
                with st.spinner(f"Training {model_name} model..."):
                    model.fit(X_train, y_train)
                    
                    # If cross-validation is enabled, calculate CV scores
                    if cross_validation:
                        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                        mean_cv_score = np.mean(cv_scores)
                        std_cv_score = np.std(cv_scores)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # Display summary results
                st.success(f"Model trained successfully! {model_name} is ready for analysis.")
                
                # Professional metrics display
                st.markdown("#### Model Performance Metrics")
                
                # Show cross-validation results if enabled
                if cross_validation:
                    st.info(f"Cross-validation R² score: {mean_cv_score:.4f} ± {std_cv_score:.4f} (averaged over {cv_folds} folds)")
                
                # Create metrics dashboard
                metrics_cols = st.columns(4)
                with metrics_cols[0]:
                    st.metric("Training R²", f"{train_r2:.4f}", 
                            help="How well the model explains variation in training data (0-1, higher is better)")
                with metrics_cols[1]:
                    st.metric("Testing R²", f"{test_r2:.4f}", 
                            delta=f"{test_r2-train_r2:.4f}",
                            delta_color="inverse",
                            help="How well the model explains variation in unseen data (0-1, higher is better)")
                with metrics_cols[2]:
                    st.metric("Training RMSE", f"{train_rmse:.4f}", 
                            help="How far off predictions are on training data (lower is better)")
                with metrics_cols[3]:
                    st.metric("Testing RMSE", f"{test_rmse:.4f}", 
                            delta=f"{test_rmse-train_rmse:.4f}",
                            help="How far off predictions are on unseen data (lower is better)")
                
                # Explanation of what these metrics mean
                with st.expander("📘 What do these metrics mean? (Click to learn more)"):
                    st.markdown("""
                    **R² (R-squared)**: Shows how much of the variation your model explains (0-1)
                    - 0.9 or higher: Excellent! Your model explains 90%+ of what affects your target
                    - 0.7-0.9: Good model, captures most patterns
                    - 0.5-0.7: Moderate, captures about half the patterns
                    - Below 0.5: Weak, misses most patterns
                    
                    **RMSE (Root Mean Square Error)**: The typical error in your predictions
                    - Lower is better
                    - In the same units as your target (dollars, counts, etc.)
                    - If predicting dollars, RMSE of 100 means predictions are typically off by about $100
                    
                    **Training vs Testing**: 
                    - If testing metrics are much worse than training metrics, your model may be overfitting
                    - Good models have similar performance on both training and testing data
                    
                    **Cross-validation**:
                    - Tests your model on multiple different splits of your data
                    - More reliable estimate of how your model will perform on new data
                    - The "±" shows the variation in performance across different folds
                    """)
                
                # Visualize predictions vs actual
                st.markdown("#### Prediction Analysis")
                
                # Create tabs for different visualizations
                visual_tabs = st.tabs(["Predictions vs Actual", "Residuals Analysis", "Feature Importance"])
                
                with visual_tabs[0]:
                    # Create scatter plot of predicted vs actual values
                    pred_vs_actual = pd.DataFrame({
                        'Actual': y_test,
                        'Predicted': y_pred_test
                    })
                    
                    fig = px.scatter(pred_vs_actual, x='Actual', y='Predicted',
                                    title=f'Predicted vs Actual Values for {target_column}',
                                    labels={'Actual': f'Actual {target_column}', 
                                            'Predicted': f'Predicted {target_column}'},
                                    height=500)
                    
                    # Add perfect prediction line
                    min_val = min(pred_vs_actual['Actual'].min(), pred_vs_actual['Predicted'].min())
                    max_val = max(pred_vs_actual['Actual'].max(), pred_vs_actual['Predicted'].max())
                    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                            mode='lines', name='Perfect Prediction',
                                            line=dict(color='red', dash='dash')))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **How to interpret this chart:**
                    - Points close to the red line = accurate predictions
                    - Points above the line = model overestimated
                    - Points below the line = model underestimated
                    - Scattered points = less accurate model
                    """)
                
                with visual_tabs[1]:
                    # Calculate residuals
                    residuals = y_test - y_pred_test
                    residual_df = pd.DataFrame({
                        'Predicted': y_pred_test,
                        'Residuals': residuals
                    })
                    
                    # Create residual plot
                    fig = px.scatter(residual_df, x='Predicted', y='Residuals',
                                    title='Residuals Analysis',
                                    labels={'Predicted': f'Predicted {target_column}', 
                                            'Residuals': 'Residuals (Actual - Predicted)'},
                                    height=500)
                    
                    # Add zero line
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Histogram of residuals
                    fig2 = px.histogram(residual_df, x='Residuals', 
                                    title='Distribution of Residuals',
                                    height=400)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    st.markdown("""
                    **How to interpret these charts:**
                    - Residuals should be randomly scattered around zero
                    - Patterns in residuals suggest the model is missing something
                    - The histogram should be roughly bell-shaped and centered at zero
                    """)
                
                with visual_tabs[2]:
                    # Feature importance visualization
                    if has_feature_importance:
                        st.subheader("Feature Importance")
                        importances = model.feature_importances_
                        feature_importance = pd.DataFrame({
                            'Feature': selected_features,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                                    title='Which factors matter most in prediction?',
                                    height=500,
                                    color='Importance',
                                    color_continuous_scale='Viridis')
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **How to interpret feature importance:**
                        - Longer bars = features with stronger influence on predictions
                        - Use this to understand which factors drive your target variable
                        - Consider removing features with very low importance to simplify your model
                        """)
                    else:
                        # For models without direct feature importance
                        st.info(f"{model_name} doesn't provide direct feature importance. Consider using a tree-based model like Random Forest if feature importance is needed.")
                        
                        # For linear models, can use coefficients instead
                        if model_algorithm in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
                            st.subheader("Model Coefficients")
                            coeffs = model.coef_
                            coef_df = pd.DataFrame({
                                'Feature': selected_features,
                                'Coefficient': coeffs
                            }).sort_values('Coefficient', ascending=False)
                            
                            fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                                        title='Model Coefficients (impact on predictions)',
                                        height=500,
                                        color='Coefficient',
                                        color_continuous_scale='RdBu')
                            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                
                # Option to save the model
                if save_model:
                    # Create a unique ID for this model
                    model_id = f"{model_name.replace(' ', '_')}_{target_column.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                    
                    # Store model configurations for future use
                    model_config = {
                        'model_type': 'regression',
                        'target_column': target_column,
                        'features': selected_features,
                        'categorical_features': selected_categorical,
                        'algorithm': model_algorithm,
                        'test_size': test_size,
                        'metrics': {
                            'train_rmse': float(train_rmse),
                            'test_rmse': float(test_rmse),
                            'train_r2': float(train_r2),
                            'test_r2': float(test_r2)
                        },
                        'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Save to session state
                    if 'saved_models' not in st.session_state:
                        st.session_state.saved_models = {}
                    
                    st.session_state.saved_models[model_name] = model_config
                    
                    # Save to file
                    with open("saved_models/saved_models.json", "w") as f:
                        json.dump(st.session_state.saved_models, f, indent=4)
                    
                    st.success(f"Model saved as '{model_name}'!")
                
                # Store model in session state for prediction tab
                st.session_state.regression_model = model
                st.session_state.regression_features = selected_features
                st.session_state.regression_target = target_column
                
            except Exception as e:
                st.error(f"Error during regression prediction: {str(e)}")
    
    # Add prediction tab for making new predictions with the model
    with tabs[3]:
        st.markdown("### Make New Predictions")
        st.write("Use your trained model to predict new values.")
        
        if 'regression_model' in st.session_state:
            # Create input methods for each feature
            st.subheader(f"Enter values for predicting {st.session_state.regression_target}")
            
            prediction_inputs = {}
            
            # Create sliders/inputs for each feature
            for feature in st.session_state.regression_features:
                # Safely handle numeric values
                col_data = df[feature].dropna()
                
                if col_data.empty:
                    st.warning(f"Column '{feature}' contains only missing values")
                    prediction_inputs[feature] = 0
                    continue
                    
                try:
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    
                    # Use median instead of mean for robustness
                    mean_val = float(col_data.median())
                    
                    # Handle if min and max are the same
                    if min_val == max_val:
                        st.info(f"All values in {feature} are {min_val}")
                        prediction_inputs[feature] = min_val
                        continue
                        
                    # Handle if min or max are invalid
                    if pd.isna(min_val) or pd.isna(max_val) or np.isinf(min_val) or np.isinf(max_val):
                        st.warning(f"Cannot create slider for {feature} due to invalid values")
                        prediction_inputs[feature] = mean_val
                        continue
                        
                    # Adjust step size based on the range
                    step = (max_val - min_val) / 100 if max_val > min_val else 0.1
                    
                    # Create a slider for each numeric feature
                    prediction_inputs[feature] = st.slider(
                        f"{feature}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=step
                    )
                except Exception as e:
                    st.warning(f"Error processing feature '{feature}': {str(e)}")
                    # Use median or 0 as fallback
                    try:
                        prediction_inputs[feature] = float(col_data.median())
                    except:
                        prediction_inputs[feature] = 0
            
            # Button to make prediction
            if st.button("Predict"):
                # Create a DataFrame with a single row for prediction
                pred_data = pd.DataFrame([prediction_inputs])
                
                # Make prediction
                prediction = st.session_state.regression_model.predict(pred_data)[0]
                
                # Display prediction
                st.success(f"Predicted {st.session_state.regression_target}: **{prediction:.4f}**")
        else:
            st.info("Train a model first in the 'Results & Analysis' tab before making predictions.")

def run_classification_prediction(df, numeric_columns, categorical_columns, load_saved=False):
    """Run enhanced classification prediction workflow with advanced models and visualizations."""
    st.markdown("### Professional Classification Analysis")
    
    if not categorical_columns:
        st.warning("⚠️ Classification requires at least one categorical column as target. No categorical columns found.")
        return
        
    # Create tabs for different steps of the workflow
    tabs = st.tabs(["Data Selection", "Model Configuration", "Results & Analysis", "Prediction"])
    
    with tabs[0]:
        st.markdown("#### Select Data for Your Model")
        
        # Target selection
        target_column = st.selectbox(
            "Select the category to predict (target):",
            categorical_columns,
            help="This is the category you want your model to predict"
        )
        
        # Feature selection with interactive UI
        available_features = numeric_columns.copy()
        
        if not available_features:
            st.warning("⚠️ No numeric features available for prediction.")
            return
        
        # Option to include other categorical features with one-hot encoding
        include_categorical = st.checkbox("Include other categorical features", value=True, 
                                        help="Convert text/category columns to numbers automatically")
        
        # Feature selection with search and info
        st.markdown("**Select features for your prediction model:**")
        feature_col1, feature_col2 = st.columns([3, 1])
        
        with feature_col1:
            selected_features = st.multiselect(
                "Numeric features:",
                available_features,
                default=available_features[:min(3, len(available_features))]
            )
            
        with feature_col2:
            # Quick selection buttons
            if st.button("Select All"):
                selected_features = available_features
                st.rerun()
                
        available_categorical = [col for col in categorical_columns if col != target_column]
        if include_categorical and available_categorical:
            selected_categorical = st.multiselect(
                "Categorical features:",
                available_categorical,
                default=[]
            )
        else:
            selected_categorical = []
        
        if not selected_features and not selected_categorical:
            st.warning("⚠️ Please select at least one feature for prediction.")
            return
            
        # Data split preview with helpful explanation
        st.markdown("##### Data Splitting")
        st.info("Your data will be split into training data (to build the model) and testing data (to verify accuracy)")
        
        # Train-test split ratio with visual indicator
        col1, col2 = st.columns([3, 1])
        with col1:
            test_size = st.slider(
                "Test set size (%):",
                min_value=10,
                max_value=50,
                value=20,
                step=5
            ) / 100
        
        with col2:
            # Visual representation of the split
            st.markdown(f"Training: **{int((1-test_size)*100)}%**")
            st.markdown(f"Testing: **{int(test_size*100)}%**")
    
    with tabs[1]:
        st.markdown("#### Configure Your Classification Model")
        
        # Model selection with descriptions
        model_options = {
            "Logistic Regression": "Simple, interpretable model for binary classification",
            "Random Forest": "Robust ensemble model for complex classification",
            "Gradient Boosting": "Advanced model for highest accuracy",
            "XGBoost": "State-of-the-art algorithm for tabular data",
            "Support Vector Machine": "Works well for smaller datasets",
            "K-Nearest Neighbors": "Simple, intuitive classification algorithm"
        }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # Create model selection with descriptions
            model_descriptions = [f"{model}: {desc}" for model, desc in model_options.items()]
            selected_model_with_desc = st.radio(
                "Select algorithm:",
                options=model_descriptions
            )
            model_algorithm = selected_model_with_desc.split(":", 1)[0]
            
        with col2:
            # Model complexity/hyperparameter adjustment
            st.markdown("##### Model Settings")
            
            if model_algorithm == "Logistic Regression":
                C = st.slider("Regularization strength:", 0.1, 10.0, 1.0, 0.1,
                            help="Lower values = stronger regularization")
                solver = st.selectbox("Solver:", ["lbfgs", "liblinear", "newton-cg", "sag"])
                
            elif model_algorithm == "Random Forest":
                n_estimators = st.slider("Number of trees:", 10, 200, 100, 10,
                                       help="More trees = better accuracy but slower")
                max_depth = st.slider("Maximum tree depth:", 2, 20, 10, 1,
                                    help="Deeper trees can overfit")
                
            elif model_algorithm == "Gradient Boosting":
                n_estimators = st.slider("Number of stages:", 10, 200, 100, 10)
                learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1, 0.01,
                                       help="Lower values need more trees but may generalize better")
                
            elif model_algorithm == "XGBoost":
                n_estimators = st.slider("Number of boosting rounds:", 10, 500, 100, 10)
                learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1, 0.01)
                max_depth = st.slider("Maximum tree depth:", 3, 10, 6, 1)
                
            elif model_algorithm == "Support Vector Machine":
                kernel = st.selectbox("Kernel type:", ["linear", "rbf", "poly"])
                C = st.slider("Regularization parameter:", 0.1, 10.0, 1.0, 0.1,
                            help="Lower values = stronger regularization")
                
            elif model_algorithm == "K-Nearest Neighbors":
                n_neighbors = st.slider("Number of neighbors:", 3, 20, 5, 1,
                                      help="Number of neighbors to consider")
                weights = st.selectbox("Weight function:", ["uniform", "distance"])
        
        # Cross-validation for more reliable results
        cross_validation = st.checkbox("Use cross-validation", value=True,
                                     help="More reliable performance assessment but slower")
        
        if cross_validation:
            cv_folds = st.slider("Number of cross-validation folds:", 3, 10, 5, 1)
        
        # Create a name for saving the model
        save_model = st.checkbox("Save model for future use", value=False)
        if save_model:
            model_name = st.text_input("Model name:", value=f"{model_algorithm} for {target_column}")
    
    # Train button in a third tab
    with tabs[2]:
        train_model = st.button("Train Model", type="primary")
        
        if train_model:
            try:
                # Prepare data
                X = df[selected_features]
                y = df[target_column]
                
                # Get unique classes
                classes = y.unique()
                n_classes = len(classes)
                
                # Check if we have enough samples per class
                class_counts = y.value_counts()
                min_class_count = class_counts.min()
                
                if min_class_count < 5:
                    st.warning(f"⚠️ Warning: Some classes have very few samples (minimum: {min_class_count}). This may affect model performance.")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                
                # Alert for training
                samples_train = len(X_train)
                samples_test = len(X_test)
                st.info(f"Training model with {samples_train} samples, testing with {samples_test} samples. Target has {n_classes} unique classes.")
                
                # Initialize and train model based on selected algorithm
                if model_algorithm == "Logistic Regression":
                    model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
                    model_name = "Logistic Regression"
                    has_feature_importance = False
                elif model_algorithm == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    model_name = "Random Forest Classifier"
                    has_feature_importance = True
                elif model_algorithm == "Gradient Boosting":
                    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                    model_name = "Gradient Boosting Classifier"
                    has_feature_importance = True
                elif model_algorithm == "XGBoost":
                    model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                    model_name = "XGBoost Classifier"
                    has_feature_importance = True
                elif model_algorithm == "Support Vector Machine":
                    model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
                    model_name = "Support Vector Classifier"
                    has_feature_importance = False
                elif model_algorithm == "K-Nearest Neighbors":
                    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
                    model_name = "K-Nearest Neighbors Classifier"
                    has_feature_importance = False
                
                # Fit the model with progress indication
                with st.spinner(f"Training {model_name} model..."):
                    model.fit(X_train, y_train)
                    
                    # If cross-validation is enabled, calculate CV scores
                    if cross_validation:
                        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
                        mean_cv_score = np.mean(cv_scores)
                        std_cv_score = np.std(cv_scores)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                
                # Classification report
                class_report_test = classification_report(y_test, y_pred_test, output_dict=True)
                
                # Confusion matrix
                conf_matrix = confusion_matrix(y_test, y_pred_test)
                
                # Display summary results
                st.success(f"Model trained successfully! {model_name} is ready for analysis.")
                
                # Professional metrics display
                st.markdown("#### Model Performance Metrics")
                
                # Show cross-validation results if enabled
                if cross_validation:
                    st.info(f"Cross-validation accuracy: {mean_cv_score:.4f} ± {std_cv_score:.4f} (averaged over {cv_folds} folds)")
                
                # Create metrics dashboard
                metrics_cols = st.columns(2)
                with metrics_cols[0]:
                    st.metric("Training Accuracy", f"{train_accuracy:.4f}", 
                            help="How often the model predicts correctly on training data (0-1, higher is better)")
                with metrics_cols[1]:
                    st.metric("Testing Accuracy", f"{test_accuracy:.4f}", 
                            delta=f"{test_accuracy-train_accuracy:.4f}",
                            delta_color="inverse",
                            help="How often the model predicts correctly on unseen data (0-1, higher is better)")
                
                # Explanation of what these metrics mean
                with st.expander("📘 What do these metrics mean? (Click to learn more)"):
                    st.markdown("""
                    **Accuracy**: The proportion of correct predictions (both true positives and true negatives)
                    - 0.9 or higher: Excellent! Your model predicts correctly 90%+ of the time
                    - 0.7-0.9: Good model, correct most of the time
                    - 0.5-0.7: Moderate, better than random guessing
                    - Near 0.5 or lower: Poor, close to random guessing for binary classification
                    
                    **Precision**: How many of the positive predictions were actually correct
                    - High precision: Few false positives
                    - Example: If precision is 0.9, 90% of items predicted as positive are truly positive
                    
                    **Recall**: How many of the actual positives were correctly predicted
                    - High recall: Few false negatives
                    - Example: If recall is 0.8, the model found 80% of all positive cases
                    
                    **F1-Score**: The balance between precision and recall (harmonic mean)
                    - Good for imbalanced datasets
                    - Combines both false positives and false negatives in one metric
                    
                    **Training vs Testing**: 
                    - If testing metrics are much worse than training metrics, your model may be overfitting
                    - Good models have similar performance on both training and testing data
                    """)
                
                # Show detailed metrics by class
                st.markdown("#### Performance by Class")
                
                # Create a dataframe for the classification report
                class_metrics = []
                for cls in class_report_test.keys():
                    if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                        class_metrics.append({
                            'Class': cls,
                            'Precision': class_report_test[cls]['precision'],
                            'Recall': class_report_test[cls]['recall'],
                            'F1-Score': class_report_test[cls]['f1-score'],
                            'Support': class_report_test[cls]['support']
                        })
                
                metrics_df = pd.DataFrame(class_metrics)
                st.dataframe(metrics_df)
                
                # Visualizations for classification results
                st.markdown("#### Prediction Analysis")
                
                # Create tabs for different visualizations
                visual_tabs = st.tabs(["Confusion Matrix", "Classification Metrics", "Feature Importance"])
                
                with visual_tabs[0]:
                    # Create a heatmap of the confusion matrix
                    conf_matrix_df = pd.DataFrame(conf_matrix, 
                                                index=classes, 
                                                columns=classes)
                    
                    fig = px.imshow(conf_matrix_df,
                                    labels=dict(x="Predicted", y="Actual", color="Count"),
                                    x=classes,
                                    y=classes,
                                    title="Confusion Matrix",
                                    color_continuous_scale="Viridis",
                                    text_auto=True,
                                    aspect="auto",
                                    height=500)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **How to interpret the confusion matrix:**
                    - Diagonal cells (top-left to bottom-right) show correct predictions
                    - Off-diagonal cells show where the model made mistakes
                    - Higher numbers on diagonal = better model
                    """)
                
                with visual_tabs[1]:
                    # Create bar charts for precision, recall, f1-score by class
                    metrics_melted = pd.melt(metrics_df, 
                                          id_vars=['Class'], 
                                          value_vars=['Precision', 'Recall', 'F1-Score'],
                                          var_name='Metric', 
                                          value_name='Value')
                    
                    fig = px.bar(metrics_melted, 
                               x='Class', 
                               y='Value', 
                               color='Metric',
                               barmode='group',
                               title='Classification Metrics by Class',
                               height=500)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Class distribution
                    class_dist = pd.DataFrame(y.value_counts()).reset_index()
                    class_dist.columns = ['Class', 'Count']
                    
                    fig2 = px.pie(class_dist, 
                                values='Count', 
                                names='Class',
                                title='Class Distribution in Dataset',
                                height=400)
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
                with visual_tabs[2]:
                    # Feature importance visualization
                    if has_feature_importance:
                        st.subheader("Feature Importance")
                        importances = model.feature_importances_
                        feature_importance = pd.DataFrame({
                            'Feature': selected_features,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                                    title='Which factors matter most in prediction?',
                                    height=500,
                                    color='Importance',
                                    color_continuous_scale='Viridis')
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **How to interpret feature importance:**
                        - Longer bars = features with stronger influence on predictions
                        - Use this to understand which factors drive your classification
                        - Consider removing features with very low importance to simplify your model
                        """)
                    else:
                        # For models without direct feature importance
                        st.info(f"{model_name} doesn't provide direct feature importance. Consider using a tree-based model like Random Forest if feature importance is needed.")
                        
                        # For linear models, can use coefficients instead
                        if model_algorithm == "Logistic Regression":
                            st.subheader("Model Coefficients")
                            if n_classes == 2:  # Binary classification
                                coeffs = model.coef_[0]
                                coef_df = pd.DataFrame({
                                    'Feature': selected_features,
                                    'Coefficient': coeffs
                                }).sort_values('Coefficient', ascending=False)
                                
                                fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                                          title='Model Coefficients (impact on predictions)',
                                          height=500,
                                          color='Coefficient',
                                          color_continuous_scale='RdBu')
                                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Coefficient visualization for multi-class logistic regression is not shown due to complexity.")
                
                # Option to save the model
                if save_model:
                    # Create a unique ID for this model
                    model_id = f"{model_name.replace(' ', '_')}_{target_column.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                    
                    # Store model configurations for future use
                    model_config = {
                        'model_type': 'classification',
                        'target_column': target_column,
                        'target_classes': list(classes),
                        'features': selected_features,
                        'categorical_features': selected_categorical,
                        'algorithm': model_algorithm,
                        'test_size': test_size,
                        'metrics': {
                            'train_accuracy': float(train_accuracy),
                            'test_accuracy': float(test_accuracy)
                        },
                        'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Save to session state
                    if 'saved_models' not in st.session_state:
                        st.session_state.saved_models = {}
                    
                    st.session_state.saved_models[model_name] = model_config
                    
                    # Save to file
                    with open("saved_models/saved_models.json", "w") as f:
                        json.dump(st.session_state.saved_models, f, indent=4)
                    
                    st.success(f"Model saved as '{model_name}'!")
                
                # Store model in session state for prediction tab
                st.session_state.classification_model = model
                st.session_state.classification_features = selected_features
                st.session_state.classification_target = target_column
                st.session_state.classification_classes = classes
                
            except Exception as e:
                st.error(f"Error during classification prediction: {str(e)}")
    
    # Add prediction tab for making new predictions with the model
    with tabs[3]:
        st.markdown("### Make New Predictions")
        st.write("Use your trained model to predict new categories.")
        
        if 'classification_model' in st.session_state:
            # Create input methods for each feature
            st.subheader(f"Enter values for predicting {st.session_state.classification_target}")
            
            prediction_inputs = {}
            
            # Create sliders/inputs for each feature
            for feature in st.session_state.classification_features:
                # Safely handle numeric values
                col_data = df[feature].dropna()
                
                if col_data.empty:
                    st.warning(f"Column '{feature}' contains only missing values")
                    prediction_inputs[feature] = 0
                    continue
                    
                try:
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    
                    # Use median instead of mean for robustness
                    mean_val = float(col_data.median())
                    
                    # Handle if min and max are the same
                    if min_val == max_val:
                        st.info(f"All values in {feature} are {min_val}")
                        prediction_inputs[feature] = min_val
                        continue
                        
                    # Handle if min or max are invalid
                    if pd.isna(min_val) or pd.isna(max_val) or np.isinf(min_val) or np.isinf(max_val):
                        st.warning(f"Cannot create slider for {feature} due to invalid values")
                        prediction_inputs[feature] = mean_val
                        continue
                        
                    # Adjust step size based on the range
                    step = (max_val - min_val) / 100 if max_val > min_val else 0.1
                    
                    # Create a slider for each numeric feature
                    prediction_inputs[feature] = st.slider(
                        f"{feature}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=step
                    )
                except Exception as e:
                    st.warning(f"Error processing feature '{feature}': {str(e)}")
                    # Use median or 0 as fallback
                    try:
                        prediction_inputs[feature] = float(col_data.median())
                    except:
                        prediction_inputs[feature] = 0
            
            # Button to make prediction
            if st.button("Predict"):
                # Create a DataFrame with a single row for prediction
                pred_data = pd.DataFrame([prediction_inputs])
                
                # Make prediction
                prediction = st.session_state.classification_model.predict(pred_data)[0]
                
                # Get prediction probabilities if available
                if hasattr(st.session_state.classification_model, 'predict_proba'):
                    proba = st.session_state.classification_model.predict_proba(pred_data)[0]
                    proba_df = pd.DataFrame({
                        'Class': st.session_state.classification_classes,
                        'Probability': proba
                    }).sort_values('Probability', ascending=False)
                    
                    # Display prediction with probability
                    st.success(f"Predicted {st.session_state.classification_target}: **{prediction}** with {proba_df.iloc[0]['Probability']:.2%} probability")
                    
                    # Show all class probabilities
                    st.subheader("Prediction Probabilities")
                    
                    # Create bar chart of probabilities
                    fig = px.bar(proba_df, x='Class', y='Probability',
                               title='Prediction Probabilities by Class',
                               color='Probability',
                               height=400,
                               color_continuous_scale='Viridis')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # Display prediction only
                    st.success(f"Predicted {st.session_state.classification_target}: **{prediction}**")
        else:
            st.info("Train a model first in the 'Results & Analysis' tab before making predictions.")