import pandas as pd
import numpy as np

"""
Read in data
"""

DATA_PATH = "/Users/elliehuang/Desktop/capstone/data/"  # Replace with your file path

load_level = pd.read_csv(DATA_PATH + "load_level_shipment_records_chainalytics.csv")
service_performance = pd.read_csv(DATA_PATH + "service_performance.csv")
lane_breadth = pd.read_excel(DATA_PATH + "lane_breadth.xlsx")
commitment_vs_take = pd.read_csv(DATA_PATH + "commitment_vs_take.csv")
carrier_tenure = pd.read_csv(DATA_PATH + "carrier_tenure.csv")
ftr_index_data = pd.read_csv(DATA_PATH + "ftr_index_data.csv")
parent_ids = pd.read_csv(DATA_PATH + "Carrier Parent Company.csv")

"""
Join files with parent_ids by CARRIER_SKEY from parent_ids
"""

parent_load_level = load_level.merge(parent_ids, how='inner', on='CARRIER_SKEY')
parent_service_performance = service_performance.merge(parent_ids, how='inner', on='CARRIER_SKEY')
parent_lane_breadth = lane_breadth.merge(parent_ids, how='inner', on='CARRIER_SKEY')
parent_commitment_vs_take = commitment_vs_take.merge(parent_ids, how='inner', on='CARRIER_SKEY')
parent_carrier_tenure = carrier_tenure.merge(parent_ids, how='inner', on='CARRIER_SKEY')

"""
Make date columns DATETIME type, not string (for later)
"""
parent_load_level['DROP_ACTUAL_DATE'] = pd.to_datetime(
    parent_load_level['DROP_ACTUAL_DATE'], errors='coerce'
)
parent_load_level['PICK_ACTUAL_DATE'] = pd.to_datetime(
    parent_load_level['PICK_ACTUAL_DATE'], errors='coerce'
)
parent_load_level['DROP_DUE_DATE'] = pd.to_datetime(
    parent_load_level['DROP_DUE_DATE'], errors='coerce'
)
parent_load_level['PICK_DUE_DATE'] = pd.to_datetime(
    parent_load_level['PICK_DUE_DATE'], errors='coerce'
)

"""
Compute month-over-month percent change in FTR spot rate (for Volatility_2)
"""

spot_rate = ftr_index_data[
    ftr_index_data['DATA_LABEL'] == 'Total TL: Spot Rate (exc. FSC, SA)'
].copy()
spot_rate['DATE'] = pd.to_datetime(spot_rate['DATE'])
spot_rate = spot_rate.sort_values('DATE')
spot_rate['SPOT_RATE_PCT_CHANGE'] = spot_rate['VALUE'].pct_change() # Compute month-over-month percent change in spot rate
spot_rate = spot_rate[['DATE', 'SPOT_RATE_PCT_CHANGE']].dropna()


"""
Compute each parent carrier's spot load proportion by month (for Volatility_2)
"""

parent_load_level['IS_SPOT'] = parent_load_level['AWARD_TYPE'].isin(
    ['Carrier not in waterfall', 'No waterfall for lane']
).astype(int)

# Create year-month column for monthly grouping using pick-up date
parent_load_level['YEAR_MONTH'] = (
    parent_load_level['PICK_ACTUAL_DATE'].dt.to_period('M').dt.to_timestamp()
)

# Aggregate spot and total load counts per carrier per month
monthly_spot = (
    parent_load_level
    .groupby(['PARENT_COMPANY_ID', 'YEAR_MONTH'])
    .agg(
        SPOT_COUNT=('IS_SPOT', 'sum'),
        TOTAL_COUNT=('IS_SPOT', 'size')
    )
    .reset_index()
)

# Compute proportion of loads that were spot in each month
monthly_spot['SPOT_PROP'] = monthly_spot['SPOT_COUNT'] / monthly_spot['TOTAL_COUNT']

monthly_spot = monthly_spot.sort_values(['PARENT_COMPANY_ID', 'YEAR_MONTH'])
monthly_spot['SPOT_PROP_CHANGE'] = (
    monthly_spot.groupby('PARENT_COMPANY_ID')['SPOT_PROP'].diff()
)

# Join carrier monthly data with FTR spot rate changes on matching month
monthly_spot = monthly_spot.merge(
    spot_rate,
    left_on='YEAR_MONTH',
    right_on='DATE',
    how='inner' # Only keep months where FTR data exists
)

monthly_spot['VOL2_MONTHLY'] = (
    monthly_spot['SPOT_PROP_CHANGE'] * monthly_spot['SPOT_RATE_PCT_CHANGE']
)

# Average across all months per carrier to get final Volatility_2 score
volatility_2_df = (
    monthly_spot
    .groupby('PARENT_COMPANY_ID')['VOL2_MONTHLY']
    .mean()
    .reset_index()
    .rename(columns={'VOL2_MONTHLY': 'Volatility_2'})
)

"""
Aggregate parent_load_level data at parent carrier level (PARENT_COMPANY_ID)

Features of interest:
- ON_TIME_PICK | load_level
- ON_TIME_DROP | load_level
- AWARD_TYPE | load_level
- QUANTITY | load_level
- TRANSIT_TIME_STANDARD | load_level
- DROP_ACTUAL_DATE | load_level
- PICK_ACTUAL_DATE | load_level
- DROP_DUE_DATE | load_level
- PICK_DUE_DATE | load_level
- TEMPERATURE_ZONE | load_level
- PAID_LINEHAUL | load_level
- MILEAGE | load_level
- TOTAL LOADS | load_level
"""

parent_agg_load_level = parent_load_level.groupby('PARENT_COMPANY_ID').agg(
    ON_TIME_PICK_YES_COUNT=(
        'ON_TIME_PICK',
        lambda x: (x == 1).sum()  # Count 1s (on-time)
    ),
    ON_TIME_DROP_YES_COUNT=(
        'ON_TIME_DROP',
        lambda x: (x == 1).sum()  # Count 1s (on-time)
    ),
    AWARD_TYPE_PRIMARY_COUNT=(
        'AWARD_TYPE',
        lambda x: (x == 'Primary').sum()  # Count Primary loads
    ),
    AWARD_TYPE_WATERFALL_COUNT=(
        'AWARD_TYPE',
        lambda x: (x.isin(['Waterfall #2', 'Waterfall #3', 'Waterfall #4', 'Waterfall #5', 'Waterfall #6']).sum())  # Count all Waterfall loads
    ),
    AWARD_TYPE_SPOT_COUNT=(
        'AWARD_TYPE',
        lambda x: (x.isin(['Carrier not in waterfall', 'No waterfall for lane']).sum())
    ),
    QUANTITY_TOTAL=(
        'QUANTITY',
        lambda x: x.sum()
    ),
    TRANSIT_TIME_STANDARD_TRUE_COUNT=(
        'TRANSIT_TIME_STANDARD',
        lambda x: x.isin(['Y', 'TRUE', 'STANDARD']).sum()
    ),
    ACTUAL_DIFF_TOTAL=(
        'DROP_ACTUAL_DATE',
        lambda x: (x - parent_load_level.loc[x.index, 'PICK_ACTUAL_DATE']).dt.days.sum()  # Get sum of differences between actual pickup and dropoff
    ),
    DUE_DIFF_TOTAL=(
        'DROP_DUE_DATE',
        lambda x: (x - parent_load_level.loc[x.index, 'PICK_DUE_DATE']).dt.days.sum()  # Get sum of differences between expected pickup and dropoff
    ),
    TEMPERATURE_REQ_DRY_COUNT=(  # Dry loads
        'TEMPERATURE_ZONE',
        lambda x: (x == 'DRY').sum()
    ),
    TEMPERATURE_REQ_SENSITIVE_COUNT=(
        'TEMPERATURE_ZONE',
        lambda x: (x == 'TEMP CONTROLLED').sum()  # Temperature controlled loads
    ),
    PAID_LINEHAUL_TOTAL=(
        'PAID_LINEHAUL',
        lambda x: x.sum()
    ),
    PAID_LINEHAUL_MEAN=(
        'PAID_LINEHAUL',
        lambda x: round(x.mean(), 2)
    ),
    PAID_LINEHAUL_STD=(
        'PAID_LINEHAUL',
        lambda x: round(x.std(), 2)
    ),
    MILEAGE_TOTAL=(
        'MILEAGE',
        lambda x: x.sum()
    ),
    MILEAGE_MEAN=(
        'MILEAGE',
        lambda x: round(x.mean(), 2)
    ),
    MILEAGE_STD=(
        'MILEAGE',
        lambda x: round(x.std(), 2)
    ),
    TOTAL_LOADS=(
        'QUANTITY',  # could be any column
        'size'
    )
)

# Filter for total loads > 50
parent_agg_load_level = parent_agg_load_level[parent_agg_load_level['TOTAL_LOADS'] >= 50]

print(parent_agg_load_level)


"""
Aggregate parent_commitment_vs_take data at parent carrier level (PARENT_COMPANY_ID)

Features of interest:

- ACTUAL_QUANTITY | commitment_vs_take
"""
parent_agg_commitment_vs_take = parent_commitment_vs_take.groupby('PARENT_COMPANY_ID').agg(
    ACTUAL_QUANTITY_TOTAL=(
        'ACTUAL_QUANTITY',
        lambda x: x[x != 0].sum()  # Not including rows where ACTUAL_QUANTITY == 0 in total
    ),
    ACTUAL_QUANTITY_MEAN=(
        'ACTUAL_QUANTITY',
        lambda x: round(x[x != 0].mean(), 2)  # Not including rows where ACTUAL_QUANTITY == 0 in mean
    ),
    ACTUAL_QUANTITY_STD=(
        'ACTUAL_QUANTITY',
        lambda x: round(x[x != 0].std(), 2)  # Not including rows where ACTUAL_QUANTITY == 0 in std
    ),
    TOTAL_LANES=(
        'ACTUAL_QUANTITY',  # Could be any column
        'size'
    )
)
print(parent_agg_commitment_vs_take)


"""
Aggregate parent_carrier_tenure data at parent carrier level (PARENT_COMPANY_ID)

Features of interest:

- TENURE_YEARS | carrier_tenure
"""

parent_agg_carrier_tenure = parent_carrier_tenure.groupby('PARENT_COMPANY_ID').agg(
    PARENT_TENURE_YEARS=(
        'TENURE_YEARS',
        lambda x: round(x.mean(), 2)
    )
)


"""
Aggregate parent_service_performance data at parent carrier level (PARENT_COMPANY_ID)

Features of interest:

- CLAIM_TYPE_CD | service_performance
"""
parent_agg_service_performance = parent_service_performance.groupby('PARENT_COMPANY_ID').agg(
    CLAIM_TYPE_CD_COUNT=(
        'CLAIM_TYPE_CD',
        lambda x: (x != '').sum()
    )
)


"""
Export parent aggregates as .csv
"""

"""
parent_agg_load_level.to_csv('/Users/vaibhavjha/Documents/Capstone/Data/parent_agg_load_level_shipment_record_chainalytics.csv',
                             index=True,
                             index_label="PARENT_COMPANY_ID")
parent_agg_commitment_vs_take.to_csv('/Users/vaibhavjha/Documents/Capstone/Data/parent_agg_commitment_vs_take.csv',
                                     index=True,
                                     index_label="PARENT_COMPANY_ID")
parent_agg_carrier_tenure.to_csv('/Users/vaibhavjha/Documents/Capstone/Data/parent_agg_carrier_tenure.csv',
                                     index=True,
                                     index_label="PARENT_COMPANY_ID")
parent_agg_service_performance.to_csv('/Users/vaibhavjha/Documents/Capstone/Data/parent_agg_service_performance.csv',
                                     index=True,
                                     index_label="PARENT_COMPANY_ID")
"""

"""
Merge all parent-level aggregates into one dataframe
"""

parent_agg_load_level = parent_agg_load_level.reset_index()  # Converts the index of a DataFrame back into a normal column
parent_agg_commitment_vs_take = parent_agg_commitment_vs_take.reset_index()
parent_agg_carrier_tenure = parent_agg_carrier_tenure.reset_index()
parent_agg_service_performance = parent_agg_service_performance.reset_index()

parent_features = (
    parent_agg_load_level
    .merge(parent_agg_commitment_vs_take, how='left', on='PARENT_COMPANY_ID')
    .merge(parent_agg_carrier_tenure, how='left', on='PARENT_COMPANY_ID')
    .merge(parent_agg_service_performance, how='left', on='PARENT_COMPANY_ID')
)

"""
Create parent-level engineered features by cohort
"""

# Consistency
parent_features['Consistency_1'] = (
    parent_features['ACTUAL_QUANTITY_TOTAL'] / parent_features['QUANTITY_TOTAL']
)

parent_features['Consistency_2'] = (
    (parent_features['TOTAL_LOADS'] - parent_features['CLAIM_TYPE_CD_COUNT']) /
    parent_features['TOTAL_LOADS']
)

parent_features['Consistency_3'] = (
    parent_features['ON_TIME_PICK_YES_COUNT'] / parent_features['TOTAL_LOADS']
)

parent_features['Consistency_4'] = (
    parent_features['ON_TIME_DROP_YES_COUNT'] / parent_features['TOTAL_LOADS']
)

parent_features['Consistency_5'] = (
    parent_features['TRANSIT_TIME_STANDARD_TRUE_COUNT'] / parent_features['TOTAL_LOADS']
)

parent_features['Consistency_6'] = (
    parent_features['AWARD_TYPE_WATERFALL_COUNT'] /
    (parent_features['AWARD_TYPE_PRIMARY_COUNT'] + parent_features['AWARD_TYPE_WATERFALL_COUNT'])
)

parent_features['Consistency_7'] = (
    (parent_features['DUE_DIFF_TOTAL'] - parent_features['ACTUAL_DIFF_TOTAL']) /
    parent_features['TOTAL_LOADS']
)

# Volatility
parent_features['Volatility_1'] = (
    (parent_features['AWARD_TYPE_SPOT_COUNT'] / parent_features['TOTAL_LOADS']) *
    (parent_features['ACTUAL_QUANTITY_TOTAL'] / parent_features['QUANTITY_TOTAL'])
)

parent_features['Volatility_2'] = parent_features['PARENT_COMPANY_ID'].map(
    volatility_2_df.set_index('PARENT_COMPANY_ID')['Volatility_2']
)

parent_features['Volatility_3'] = (
    125 - parent_features['PARENT_TENURE_YEARS']
)

# Adaptability
parent_features['Adaptability_1'] = (
    (parent_features['TEMPERATURE_REQ_DRY_COUNT'] * parent_features['TEMPERATURE_REQ_SENSITIVE_COUNT']) /
    (parent_features['TOTAL_LOADS'] ** 2)
)

parent_features['Adaptability_2'] = (
    parent_features['MILEAGE_MEAN'] / parent_features['MILEAGE_STD']
)

parent_features['Adaptability_3'] = (
    parent_features['ACTUAL_QUANTITY_MEAN'] / parent_features['ACTUAL_QUANTITY_STD']
)

parent_features['Adaptability_4'] = (
    (parent_features['AWARD_TYPE_PRIMARY_COUNT'] * parent_features['AWARD_TYPE_WATERFALL_COUNT']) /
    (parent_features['TOTAL_LOADS'] ** 2)
)

parent_features['Adaptability_5'] = (
    parent_features['PAID_LINEHAUL_MEAN'] / parent_features['PAID_LINEHAUL_STD']
)

# Service Capacity
parent_features['ServiceCapacity_1'] = (
    parent_features['ACTUAL_QUANTITY_TOTAL']
)

parent_features['ServiceCapacity_2'] = (
    parent_features['TOTAL_LOADS'] / parent_features['TOTAL_LOADS'].sum()
)

parent_features['ServiceCapacity_3'] = (
    parent_features['MILEAGE_TOTAL']
)

parent_features['ServiceCapacity_4'] = (
    parent_features['PAID_LINEHAUL_TOTAL']
)

"""
Replace inf values from division-by-zero with NaN
"""

parent_features = parent_features.replace([np.inf, -np.inf], np.nan)

"""
Standardize engineered feature columns
Append new standardized columns to parent_features
"""

feature_cols_to_standardize = [
    'Consistency_1',
    'Consistency_2',
    'Consistency_3',
    'Consistency_4',
    'Consistency_5',
    'Consistency_6',
    'Consistency_7',
    'Volatility_1',
    'Volatility_2',
    'Volatility_3',
    'Adaptability_1',
    'Adaptability_2',
    'Adaptability_3',
    'Adaptability_4',
    'Adaptability_5',
    'ServiceCapacity_1',
    'ServiceCapacity_2',
    'ServiceCapacity_3',
    'ServiceCapacity_4'
]

for col in feature_cols_to_standardize:
    col_mean = parent_features[col].mean()
    col_std = parent_features[col].std()

    parent_features[col + '_STD'] = (
        (parent_features[col] - col_mean) / col_std
    )
