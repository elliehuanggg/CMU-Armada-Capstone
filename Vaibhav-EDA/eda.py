import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Read in data
"""
load_level = pd.read_csv("/Users/vaibhavjha/Documents/Capstone/Data/load_level_shipment_records.csv")
service_performance = pd.read_csv("/Users/vaibhavjha/Documents/Capstone/Data/service_performance.csv")

# Loop through DataFrames
data = [load_level, service_performance]
for i in data:
    print(i.shape)

"""
Dimensions:
- load_level: 439,171 x 62
- service_performance: 464,060 x 24
"""


# Assess Freeze/Refrigerate/Dry with `TEMPERATURE_REQ`
print(load_level['TEMPERATURE_REQ'].nunique())
print(load_level['CARRIER_SKEY'].nunique())


"""
Scatterplot of `TEMPERATURE_ZONE` aggregated by `CARRIER_SKEY` with Dry on x-axis and Temp Controlled on y-axis
"""

carrier_temp = pd.crosstab(
    load_level['CARRIER_SKEY'],
    load_level['TEMPERATURE_ZONE']
)

plt.scatter(carrier_temp['DRY'], carrier_temp['TEMP CONTROLLED'], color='navy', alpha=0.7)
plt.xlim(0, 3000)
plt.ylim(0, 5000)
plt.title('Carrier Loads By Temperature Requirement')
plt.suptitle('Carriers tend to be dry-heavy or temperature-controlled heavy in loads, not both.',
             y=0.98)
plt.xlabel('Number of Dry Loads')
plt.ylabel('Number of Temperature Controlled Loads')
#plt.show()
plt.close()


"""
Contingency table of on-time pick vs. on-time drop
"""
timeliness_na = [2.0]
load_level_time = load_level[~load_level['ON_TIME_PICK'].isin(timeliness_na)]
load_level_time = load_level_time[~load_level_time['ON_TIME_DROP'].isin(timeliness_na)]

carrier_timeliness = pd.crosstab(
    load_level_time['ON_TIME_PICK'],
    load_level_time['ON_TIME_DROP'],
    normalize='all'
)

print(carrier_timeliness)

fig, ax = plt.subplots()
sns.heatmap(carrier_timeliness, annot=True, cmap='Blues', ax=ax)
plt.title('Timeliness of Load From Pick to Drop')
plt.suptitle('Loads tend to be equally delayed in pick-up and drop-off, with 89% being on-time.',
             y=0.98)
plt.xlabel('On-Time Pick-Up')
plt.ylabel('On-Time Drop-Off')
ax.set_xticklabels(['No', 'Yes'])
ax.set_yticklabels(['No', 'Yes'])
ax.tick_params(axis='y', rotation=360)
#plt.show()
plt.close()


"""
Assessing value of primary vs. waterfall paid by carrier
EDITED after suggestion from Matthew
"""

# Subset load_level for relevant loads
awards_of_interest = ['Primary', 'Waterfall #2']
carrier_reliable = load_level[load_level['AWARD_TYPE'].isin(awards_of_interest)]

# Subset primary vs. waterfall #2
primary_carrier = carrier_reliable[carrier_reliable['AWARD_TYPE'] == 'Primary']
waterfall_carrier = carrier_reliable[carrier_reliable['AWARD_TYPE'] == 'Waterfall #2']

# Aggregate means for each
agg_primary_carrier = primary_carrier.groupby('CARRIER_SKEY', as_index=False).agg(avg_payment=('TOTAL_PAYMENT_AMOUNT', 'mean'))
agg_waterfall_carrier = waterfall_carrier.groupby('CARRIER_SKEY', as_index=False).agg(avg_payment=('TOTAL_PAYMENT_AMOUNT', 'mean'))

# Merge
plot_df = (agg_primary_carrier.merge(agg_waterfall_carrier,
                                     on='CARRIER_SKEY',
                                     how='inner',
                                     suffixes=('_primary', '_waterfall')))

# Scatterplot weight x payment
plt.scatter(plot_df['avg_payment_primary'], plot_df['avg_payment_waterfall'], c='navy')
plt.xlim(0,)
plt.ylim(0,)
plt.title('Primary vs. Waterfall #2 Load Value, Aggregated by Carrier')
plt.suptitle('Among carriers delivering primary and waterfall loads, there is variation\nin the value they take on.', y=0.98)
plt.xlabel("Average Primary Load Payment Amount ($USD)")
plt.ylabel("Average Waterfall Load Payment Amount ($USD)")
plt.figtext(0.5, 0.005, 'One point = one carrier. Each carrier plotted logged at least one primary and one waterfall load.',
            ha='center', fontsize=8, style='italic')
plt.show()
plt.close()


"""
Finding percentiles of loads taken on by carriers
"""
carrier_frequency = load_level['CARRIER_SKEY'].value_counts()
Q1 = carrier_frequency.quantile(0.25)  # 232.75
Q2 = carrier_frequency.median()  # 823.5
Q3 = carrier_frequency.quantile(0.75)  # 2088.75
print(Q1)
print(Q2)
print(Q3)


"""
Histogram of loads taken on by carriers
"""
plt.hist(carrier_frequency, bins=15, range=(0, 10000), color='navy')
plt.axvline(Q1, color='gold', linestyle='--')
plt.axvline(Q2, color='gold', linestyle='--')
plt.axvline(Q3, color='gold', linestyle='--')
plt.title('Histogram of Number of Loads One Carrier Delivered')
plt.suptitle('The histogram of loads delivered by carriers is skewed to the right,\nwith few carriers taking many loads.',
             y=0.98)
plt.xlabel("Number of Loads Delivered")
plt.ylabel("Frequency")
#plt.figtext(0.5, 0.005, 'The range is (1, 10,000). There are select carriers with more loads.',
            #ha='center', fontsize=8, style='italic')
#plt.show()
plt.close()


"""
Subset data by carrier's frequency
Get rid of 0s
"""
load_counts = load_level.groupby('CARRIER_SKEY').size()

carrier_firstq = load_level[load_level['CARRIER_SKEY'].map(load_counts).between(1, Q1)]
carrier_secondq = load_level[load_level['CARRIER_SKEY'].map(load_counts).between(Q1+1, Q2)]
carrier_thirdq = load_level[load_level['CARRIER_SKEY'].map(load_counts).between(Q2+1, Q3)]
carrier_fourthq = load_level[load_level['CARRIER_SKEY'].map(load_counts) > Q3]

# Get rid of 0s
carrier_firstq = carrier_firstq[(carrier_firstq['CONTRACT_LINEHAUL'] > 100) & (carrier_firstq['PAID_LINEHAUL'] > 100)]
carrier_secondq = carrier_secondq[(carrier_secondq['CONTRACT_LINEHAUL'] > 100) & (carrier_secondq['PAID_LINEHAUL'] > 100)]
carrier_thirdq = carrier_thirdq[(carrier_thirdq['CONTRACT_LINEHAUL'] > 100) & (carrier_thirdq['PAID_LINEHAUL'] > 100)]
carrier_fourthq = carrier_fourthq[(carrier_fourthq['CONTRACT_LINEHAUL'] > 100) & (carrier_fourthq['PAID_LINEHAUL'] > 100)]


"""
Comparing paid rate vs. market rate
Make scatterplot of paid v. market rate for each quartile
# ACCOUNT FOR 0S, MAKE CONDITIONAL ON IF CONTRACT OR SPOT
"""
quartiles = [carrier_firstq, carrier_secondq, carrier_thirdq, carrier_fourthq]
for i in quartiles:
    plt.scatter(i['CONTRACT_LINEHAUL'], i['PAID_LINEHAUL'], color='navy', alpha=0.5, s=30)
    xlim = plt.xlim()
    plt.plot(xlim, xlim, color='gold', linestyle='--', lw=1)
    plt.title('Carrier Pay Evaluation')
    plt.xlabel("Contract Benchmark Price")
    plt.ylabel("Actual Price")
    #plt.show()
    plt.close()


# Duplicates issue
"""
ASK THEO:

Merge two DataFrames

load = pd.merge(load_level, service_performance, on='LOAD_ID', how='outer')
print(load.shape)

Dimensions:
- load: 464,160 x 85
"""

load = pd.merge(load_level, service_performance, on='LOAD_ID', how='inner')
print(f"Length of load_level: {len(load_level)}")
print(f"Length of service_performance: {len(service_performance)}")
print(f"Length of join table: {len(load)}")

print(f'Number of duplicated load ids in the join table: {load['LOAD_ID'].duplicated().sum()}')
print(f'Number of duplicated load ids in the load_level_shipment_record table: {load_level['LOAD_ID'].duplicated().sum()}')
print(f'Number of duplicated load ids in the service_performance table: {service_performance['LOAD_ID'].duplicated().sum()}')


"""
Plot for Vraj
Histogram of carriers' total cost / mile
"""

load_level['Cost Per Mile'] = load_level['TOTAL_PAYMENT_AMOUNT'] / load_level['MILEAGE']
load_level['Cost Per Mile'] = load_level['Cost Per Mile'].replace([np.inf, -np.inf], np.nan)
cpm = load_level['Cost Per Mile'].dropna()
upper = cpm.quantile(0.90)



print(load_level['TOTAL_PAYMENT_AMOUNT'].isna().sum())  # 0 missing vals
print(load_level['MILEAGE'].isna().sum())  # 21446 missing vals

print(load_level['Cost Per Mile'].isna().sum())  # 26446 missing vals
print(len(load_level))  # 439,171 total rows


# Histogram of aggregates
plt.hist(cpm, bins=20, color='navy', range=(0, upper))
plt.xlim(0, upper)
plt.title('Distribution of Armada true per-mile rates')
plt.suptitle("Armada's loads' cost per mile skew further to the right in comparison to DAT benchmarks.", y=0.98)
plt.xlabel("Cost Per Mile")
plt.ylabel("Number of Loads")
plt.show()
plt.close()
