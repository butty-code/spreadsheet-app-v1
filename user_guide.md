# Excel Analyzer - Simple User Guide

## What This App Does

The Excel Analyzer makes it easy to understand your Excel data without needing to be a data expert. You can:
- See what's in your Excel sheets
- Clean up messy data
- Filter and sort data
- Create charts and graphs
- Make predictions based on your data
- Export your results

## How to Use

1. Upload your Excel file using the upload button
2. Select which sheet to analyze
3. Use the tabs to perform different operations

## Guide to Each Tab

### 1. Data View
**What it does:** Shows you what's in your Excel sheet and basic statistics.

**What you need:** Any type of Excel data.

**How to use it:**
- Adjust how many rows to see using the slider
- Check "Show basic statistics" to see averages, minimums, maximums, etc. for number columns

### 2. Data Cleaning
**What it does:** Helps fix common data problems before analysis.

**What you need:** Data with issues like wrong data types, missing values, duplicates, etc.

**How to use it:** Choose from five different cleaning operations:

**Data Types Tab**
- Automatically detects columns with wrong data types
- Converts text to numbers, dates, or categories
- Shows you what was changed

**Missing Values Tab**
- Shows which columns have missing data
- Options to:
  - Remove rows with missing values
  - Fill with average (mean) or median values
  - Fill with most common value
  - Fill with zero or custom value

**Duplicate Rows Tab**
- Finds identical rows in your data
- Shows you the duplicates
- Options to keep first, last, or remove all duplicates
- Can also find partial duplicates based on specific columns

**Text Cleaning Tab**
- Fixes messy text data
- Options for:
  - Removing extra spaces
  - Converting to uppercase, lowercase, or title case
  - Removing special characters
  - Custom find and replace

**Outliers Tab**
- Finds unusual values that might skew your analysis
- Uses statistical methods (Z-score or IQR)
- Options to:
  - View the outliers
  - Remove outlier rows
  - Cap outliers at a reasonable value
  - Replace outliers with average or median

After making your changes, click "Replace current data with cleaned data" to save your changes.

### 3. Data Operations

#### Filtering
**What it does:** Lets you narrow down your data based on specific values.

**What you need:** 
- For numeric filters: columns with numbers (prices, quantities, ages, etc.)
- For category filters: columns with text values (names, product types, colors, etc.)

**How to use it:**
- Select "Filter" 
- Choose which columns to filter
- For numbers: use the sliders to set minimum and maximum values
- For categories: select which values to include
- Check "Show filtered data" to see results
- Click "Replace current data" if you want to keep only the filtered data

#### Sorting
**What it does:** Arranges your data in order (smallest to largest, A to Z, etc.)

**What you need:** Any column to sort by

**How to use it:**
- Select "Sort"
- Choose which column to sort by
- Select ascending (A→Z, lowest→highest) or descending (Z→A, highest→lowest)
- Check "Show sorted data" to see results
- Click "Replace current data" if you want to keep the sorted order

### 3. Data Visualization

**What it does:** Creates different charts and graphs to help you see patterns in your data.

**What you need:** Depends on the chart type:

**Bar Chart**
- Needs: At least one category column (like product names) and one number column (like sales)
- Great for: Comparing values across categories
- Example: "Show me total sales for each product"

**Line Chart**
- Needs: Data that changes over time or another continuous value
- Great for: Trends over time
- Example: "Show me how sales changed each month"

**Scatter Plot**
- Needs: Two number columns
- Great for: Seeing relationships between two numbers
- Example: "Does price affect how many units we sell?"

**Histogram**
- Needs: One number column
- Great for: Seeing how values are distributed
- Example: "What price range do most of our products fall into?"

**Box Plot**
- Needs: One number column
- Great for: Seeing the spread of values and finding unusual values
- Example: "What's the typical range of customer spending?"

**Pie Chart**
- Needs: One category column and one number column
- Great for: Showing how a total is divided up
- Example: "What percentage of sales comes from each product?"

**Heatmap**
- Needs: Multiple number columns
- Great for: Seeing relationships between many different measurements
- Example: "Which measurements tend to increase or decrease together?"

### 4. Predictive Analysis

**What it does:** Uses your existing data to make predictions about new data.

#### Regression (predicting a number)

**What it does:** Predicts a number based on other values.

**What you need:**
- Target: A number column you want to predict (like sales)
- Features: Other number columns that might help predict your target (like price, advertising spend)

**How to use it:**
- Select "Regression"
- Choose which number to predict
- Select features that might help predict it
- Choose an algorithm (Linear Regression is simpler, Random Forest usually more accurate)
- Click "Run Prediction"
- Use sliders at the bottom to make a new prediction

**What the results mean:**
- **R² (R-squared):** Shows how much of the variation your model explains (0-1)
  - 0.9 or higher: Excellent! Your model explains 90%+ of what affects your target
  - 0.7-0.9: Good model, captures most patterns
  - 0.5-0.7: Moderate, captures about half the patterns
  - Below 0.5: Weak, misses most patterns
  - Example: R² of 0.8 means 80% of what makes sales go up or down is captured

- **RMSE (Root Mean Square Error):** The typical error in your predictions
  - Lower is better
  - In the same units as your target (dollars, counts, etc.)
  - Example: If predicting house prices in dollars, RMSE of 10,000 means predictions are typically off by about $10,000

- **Training vs Testing metrics:** 
  - Training: How well the model fits data it learned from
  - Testing: How well it works on new data (more important)
  - If testing metrics are much worse than training, your model might be "memorizing" rather than learning

- **Feature Importance:** Shows which columns matter most for predictions

#### Classification (predicting a category)

**What it does:** Predicts which category something belongs to.

**What you need:**
- Target: A category column you want to predict (like "Will customer buy: Yes/No")
- Features: Number columns that might help predict your target (like age, income)

**How to use it:**
- Select "Classification"
- Choose which category to predict
- Select features that might help predict it
- Choose an algorithm (Logistic Regression is simpler, Random Forest usually more accurate)
- Click "Run Prediction"
- Use sliders at the bottom to make a new prediction

**What the results mean:**
- **Accuracy:** The percentage of predictions that are correct
  - 0.90 means 90% of predictions are right
  - Higher is better (above 0.7 is usually good)
  - Example: Accuracy of 0.85 means the model correctly identifies customers who will buy 85% of the time

- **Precision and Recall:** More detailed measures of accuracy
  - **Precision:** When the model predicts "yes," how often is it right?
    - Example: When the model says a customer will buy, is it correct?
  - **Recall:** Out of all actual "yes" cases, how many did the model find?
    - Example: Out of all customers who actually buy, how many did the model identify?

- **Feature Importance:** Shows which columns matter most for predictions
  - Higher values mean that feature strongly influences the prediction
  - Example: If "age" has high importance for predicting purchases, it means age is a key factor

- **Prediction Probabilities:** Shows how confident the model is 
  - Higher probability means more confidence
  - Example: 90% probability of "Yes" means the model is very confident in its prediction

### 5. Export

**What it does:** Saves your current data to a new file.

**What you need:** Any data

**How to use it:**
- Select format (Excel, CSV, or JSON)
- Click "Export Data"
- Download the file

## Tips for Best Results

1. Make sure your Excel has proper column headers (names at the top of each column)
2. For predictions, more data rows usually give better results
3. For visualizations, try different chart types to see which shows your data best
4. Use filtering to focus on just the data you're interested in
5. For best predictions, try to have at least 30 data points (rows)