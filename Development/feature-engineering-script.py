import pandas as pd
import numpy as np

"""
Part 2: Feature Engineering
"""

"""
Read in data
"""

DATA_PATH = "/Users/elliehuang/Desktop/capstone/data/"  # Replace with your file path

parent_features = pd.read_csv(DATA_PATH + "parent_raw_features.csv")
volatility_2_df = pd.read_csv(DATA_PATH + "volatility_2.csv")


"""
Create parent-level engineered features by cohort
"""

# Consistency
parent_features['Consistency_1'] = (
    parent_features['ACTUAL_QUANTITY_TOTAL'] / parent_features['QUANTITY_TOTAL']
)

parent_features['Consistency_2'] = (
    parent_features['CLAIM_TYPE_CD_FALSE_COUNT'] / parent_features['TOTAL_LOADS']
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
    (1 - parent_features['ACTUAL_QUANTITY_TOTAL'] / parent_features['QUANTITY_TOTAL']).clip(lower=0)  # Positive part: proportion of unmet commitment
)

# Map Volatility_2 from pre-computed monthly spot/rate change calculation
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
    parent_features['MILEAGE_STD'] / parent_features['MILEAGE_MEAN']
)

parent_features['Adaptability_3'] = (
    parent_features['ACTUAL_QUANTITY_STD'] / parent_features['ACTUAL_QUANTITY_MEAN']
)

parent_features['Adaptability_4'] = (
    (parent_features['AWARD_TYPE_PRIMARY_COUNT'] * parent_features['AWARD_TYPE_WATERFALL_COUNT']) /
    (parent_features['TOTAL_LOADS'] ** 2)
)

parent_features['Adaptability_5'] = (
    parent_features['PAID_LINEHAUL_STD'] / parent_features['PAID_LINEHAUL_MEAN']
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

# Economical
parent_features['Economical_1'] = (
    parent_features['NAP_LINEHAUL_CONTRACT_TOTAL'] - parent_features['PAID_LINEHAUL_CONTRACT_TOTAL']
)

parent_features['Economical_2'] = (
    (parent_features['CONTRACT_LINEHAUL_50_TOTAL'] - parent_features['PAID_LINEHAUL_CONTRACT_TOTAL']) /
    (parent_features['CONTRACT_LINEHAUL_75_TOTAL'] - parent_features['CONTRACT_LINEHAUL_25_TOTAL'])
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
    'ServiceCapacity_4',
    'Economical_1',
    'Economical_2'
]

for col in feature_cols_to_standardize:
    col_mean = parent_features[col].mean()
    col_std = parent_features[col].std()

    parent_features[col + '_STD'] = (
        (parent_features[col] - col_mean) / col_std
    )

"""
Export parent_features as .csv
"""

parent_features.to_csv(DATA_PATH + 'parent_features_engineered.csv',
                       index=False)
