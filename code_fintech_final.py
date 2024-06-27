import pandas as pd
import numpy as np
import statsmodels.api as sm

# Function to log messages
def log_message(message):
    print(message)

# Load the data
try:
    fdic_data = pd.read_csv('20210303_FDIC.csv', parse_dates=['repdte'], dayfirst=True)
    dgs3mo_data = pd.read_csv('DGS3MO.csv', parse_dates=['DATE'])
    dgs10_data = pd.read_csv('DGS10.csv', parse_dates=['DATE'])
    a191rl1q225sbea_data = pd.read_csv('A191RL1Q225SBEA.csv', parse_dates=['DATE'])
    log_message("Data loaded successfully.")
except Exception as e:
    log_message(f"Error loading data: {e}")

# Convert date columns to datetime and ensure proper format
try:
    fdic_data['repdte'] = pd.to_datetime(fdic_data['repdte'], format='%d.%m.%Y', errors='coerce')
    dgs3mo_data['DATE'] = pd.to_datetime(dgs3mo_data['DATE'], errors='coerce')
    dgs10_data['DATE'] = pd.to_datetime(dgs10_data['DATE'], errors='coerce')
    a191rl1q225sbea_data['DATE'] = pd.to_datetime(a191rl1q225sbea_data['DATE'], errors='coerce')

    # Check for and drop any rows with invalid dates
    fdic_data = fdic_data.dropna(subset=['repdte'])
    dgs3mo_data = dgs3mo_data.dropna(subset=['DATE'])
    dgs10_data = dgs10_data.dropna(subset=['DATE'])
    a191rl1q225sbea_data = a191rl1q225sbea_data.dropna(subset=['DATE'])
    log_message("Date conversion successful.")
except Exception as e:
    log_message(f"Error converting dates: {e}")

# Extract year from date columns for merging purposes
try:
    dgs3mo_data['YEAR'] = dgs3mo_data['DATE'].dt.year
    dgs10_data['YEAR'] = dgs10_data['DATE'].dt.year
    a191rl1q225sbea_data['YEAR'] = a191rl1q225sbea_data['DATE'].dt.year

    # Merge macroeconomic datasets on YEAR
    macro_data = pd.merge(dgs3mo_data[['YEAR', 'DGS3MO']], dgs10_data[['YEAR', 'DGS10']], on='YEAR')
    macro_data = pd.merge(macro_data, a191rl1q225sbea_data[['YEAR', 'A191RL1Q225SBEA']], on='YEAR')

    # Interpolate missing macroeconomic data
    macro_data.interpolate(method='linear', inplace=True)

    # Calculate the slope of the yield curve
    macro_data['SLOPE'] = macro_data['DGS10'] - macro_data['DGS3MO']
    macro_data.rename(columns={'DGS3MO': 'SHORT_RATE', 'A191RL1Q225SBEA': 'GDP_GROWTH'}, inplace=True)
    log_message("Macroeconomic data processed successfully.")
except Exception as e:
    log_message(f"Error processing macroeconomic data: {e}")

# Aggregate FDIC data to annual level
try:
    fdic_data['YEAR'] = fdic_data['repdte'].dt.year

    # Handle missing values in FDIC data using interpolation and forward/backward fill
    fdic_data = fdic_data.groupby('YEAR').apply(lambda group: group.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')).reset_index(drop=True)

    # Ensure no missing values in the relevant columns
    fdic_data_cleaned = fdic_data.dropna(subset=['nimy', 'roa'])

    # Select only numeric columns for aggregation and ensure 'YEAR' is not duplicated
    numeric_columns = fdic_data_cleaned.select_dtypes(include=[np.number]).columns
    numeric_columns = numeric_columns[numeric_columns != 'YEAR']
    fdic_annual_data = fdic_data_cleaned.groupby('YEAR')[numeric_columns].mean().reset_index()
    log_message("FDIC data processed successfully.")
except Exception as e:
    log_message(f"Error processing FDIC data: {e}")

# Save intermediate FDIC data to free up memory
try:
    chunk_size = 100000  # Adjust chunk size based on available memory
    for i, chunk in enumerate(np.array_split(fdic_annual_data, len(fdic_annual_data) // chunk_size + 1)):
        chunk.to_csv(f'fdic_data_cleaned_chunk_{i}.csv', index=False)
    macro_data.to_csv('macro_data.csv', index=False)
    log_message("Intermediate data saved to disk.")
except Exception as e:
    log_message(f"Error saving intermediate data: {e}")

# Reload intermediate data with specified data types
try:
    # Specify data types for each column
    fdic_dtype = {col: 'float' for col in fdic_annual_data.columns}
    chunks = [pd.read_csv(f'fdic_data_cleaned_chunk_{i}.csv', dtype=fdic_dtype) for i in range(len(fdic_annual_data) // chunk_size + 1)]
    fdic_data_cleaned = pd.concat(chunks, ignore_index=True)
    macro_data = pd.read_csv('macro_data.csv')
    log_message("Intermediate data reloaded successfully.")
except Exception as e:
    log_message(f"Error reloading intermediate data: {e}")

# Merge FDIC data with macroeconomic data on YEAR
try:
    final_data = pd.merge(fdic_data_cleaned, macro_data, on='YEAR')
    log_message("Data merged successfully.")
except Exception as e:
    log_message(f"Error merging data: {e}")

# Define the model
try:
    X = final_data[['GDP_GROWTH', 'SLOPE', 'SHORT_RATE']]
    X = sm.add_constant(X)
    y = final_data['nimy']

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Summary of the model
    log_message(model.summary().as_text())
except Exception as e:
    log_message(f"Error fitting the model: {e}")

# Additional analysis for structural break
try:
    final_data['post_zero_rates'] = (final_data['YEAR'] >= 2009).astype(int)
    final_data['interaction'] = final_data['SLOPE'] * final_data['post_zero_rates']

    # Redefine the model with interaction term
    X_interact = final_data[['GDP_GROWTH', 'SLOPE', 'SHORT_RATE', 'post_zero_rates', 'interaction']]
    X_interact = sm.add_constant(X_interact)
    model_interact = sm.OLS(y, X_interact).fit()

    # Summary of the interaction model
    log_message(model_interact.summary().as_text())
except Exception as e:
    log_message(f"Error with interaction model: {e}")

# Analysis of ROA relationship
try:
    y_roa = final_data['roa']
    model_roa = sm.OLS(y_roa, X).fit()
    log_message(model_roa.summary().as_text())

    # Save summaries to files
    with open('model_interact_summary.txt', 'w') as f:
        f.write(model_interact.summary().as_text())
    
    with open('model_roa_summary.txt', 'w') as f:
        f.write(model_roa.summary().as_text())
except Exception as e:
    log_message(f"Error with ROA analysis: {e}")
