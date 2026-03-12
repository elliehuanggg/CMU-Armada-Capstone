import pandas as pd

"""
Read in data
"""

load_level = pd.read_csv('/Users/vaibhavjha/Documents/Capstone/Data/load_level_shipment_records_chainalytics.csv')
service_performance = pd.read_csv("/Users/vaibhavjha/Documents/Capstone/Data/service_performance.csv")
lane_breadth = pd.read_excel("/Users/vaibhavjha/Documents/Capstone/Data/lane_breadth.xlsx")
commitment_vs_take = pd.read_csv("/Users/vaibhavjha/Documents/Capstone/Data/commitment_vs_take.csv")
carrier_tenure = pd.read_csv("/Users/vaibhavjha/Documents/Capstone/Data/carrier_tenure.csv")
parent_ids = pd.read_csv('/Users/vaibhavjha/Documents/Capstone/Data/Carrier Parent Company.csv')

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
        lambda x: (x == 1).sum()  # count 1s (on-time)
    ),
    ON_TIME_DROP_YES_COUNT=(
        'ON_TIME_DROP',
        lambda x: (x == 1).sum()  # count 1s (on-time)
    ),
    AWARD_TYPE_PRIMARY_COUNT=(
        'AWARD_TYPE',
        lambda x: (x == 'Primary').sum()  # count Primary loads
    ),
    AWARD_TYPE_WATERFALL_COUNT=(
        'AWARD_TYPE',
        lambda x: (x.isin(['Waterfall #2', 'Waterfall #3', 'Waterfall #4', 'Waterfall #5', 'Waterfall #6']).sum())  # count all Waterfall loads
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
        lambda x: (x - parent_load_level.loc[x.index, 'PICK_ACTUAL_DATE']).dt.days.sum()  # get sum of differences between actual pickup and dropoff
    ),
    DUE_DIFF_TOTAL=(
        'DROP_DUE_DATE',
        lambda x: (x - parent_load_level.loc[x.index, 'PICK_DUE_DATE']).dt.days.sum()  # get sum of differences between expected pickup and dropoff
    ),
    TEMPERATURE_REQ_DRY_COUNT=(  # dry loads
        'TEMPERATURE_ZONE',
        lambda x: (x == 'DRY').sum()
    ),
    TEMPERATURE_REQ_SENSITIVE_COUNT=(
        'TEMPERATURE_ZONE',
        lambda x: (x == 'TEMP CONTROLLED').sum()  # temperature controlled loads
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
print(parent_agg_load_level)

"""
Aggregate parent_commitment_vs_take data at parent carrier level (PARENT_COMPANY_ID)

Features of interest:

- ACTUAL_QUANTITY | commitment_vs_take
"""
parent_agg_commitment_vs_take = parent_commitment_vs_take.groupby('PARENT_COMPANY_ID').agg(
    ACTUAL_QUANTITY_TOTAL=(
        'ACTUAL_QUANTITY',
        lambda x: round(x.sum(), 2)
    ),
    ACTUAL_QUANTITY_MEAN=(
        'ACTUAL_QUANTITY',
        lambda x: round(x.mean(), 2)
    ),
    ACTUAL_QUANTITY_STD=(
        'ACTUAL_QUANTITY',
        lambda x: round(x.std(), 2)
    )
)

print(parent_agg_commitment_vs_take)

"""
Export parent aggregates as .csv
"""
parent_agg_load_level.to_csv('/Users/vaibhavjha/Documents/Capstone/Data/parent_agg_load_level_shipment_record_chainalytics.csv',
                             index=True,
                             index_label="PARENT_COMPANY_ID")
parent_agg_commitment_vs_take.to_csv('/Users/vaibhavjha/Documents/Capstone/Data/parent_agg_commitment_vs_take.csv',
                                     index=True,
                                     index_label="PARENT_COMPANY_ID")
