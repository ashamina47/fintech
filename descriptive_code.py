import pandas as pd

# Load the data
fdic_data = pd.read_csv('20210303_FDIC.csv', parse_dates=['repdte'], dayfirst=True)
dgs3mo_data = pd.read_csv('DGS3MO.csv', parse_dates=['DATE'])
dgs10_data = pd.read_csv('DGS10.csv', parse_dates=['DATE'])
a191rl1q225sbea_data = pd.read_csv('A191RL1Q225SBEA.csv', parse_dates=['DATE'])

# Convert date columns to datetime
fdic_data['repdte'] = pd.to_datetime(fdic_data['repdte'], format='%d.%m.%Y', errors='coerce')
dgs3mo_data['DATE'] = pd.to_datetime(dgs3mo_data['DATE'], errors='coerce')
dgs10_data['DATE'] = pd.to_datetime(dgs10_data['DATE'], errors='coerce')
a191rl1q225sbea_data['DATE'] = pd.to_datetime(a191rl1q225sbea_data['DATE'], errors='coerce')

# Extract year from date columns
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

# Descriptive statistics for FDIC data
fdic_descriptive_stats = fdic_data[['nimy', 'roa']].describe()

# Descriptive statistics for FRED (macroeconomic) data
fred_descriptive_stats = macro_data[['GDP_GROWTH', 'SLOPE', 'SHORT_RATE']].describe()

# Display the descriptive statistics
print("FDIC Data Descriptive Statistics:\n", fdic_descriptive_stats)
print("\nFRED Data Descriptive Statistics:\n", fred_descriptive_stats)
