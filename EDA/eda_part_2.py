import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Read in data
"""
load_level = pd.read_csv("load_level_shipment_records.csv")  # May have to specify file path
service_performance = pd.read_csv("service_performance.csv")
parent_ids = pd.read_csv("parent_carrier_company_ids(in).csv")

# Loop through DataFrames
data = [load_level, service_performance, parent_ids]
for i in data:
    print(i.shape)

"""
Dimensions:
- load_level: 439,171 x 62
- service_performance: 464,060 x 24
- parent_ids: 486 x 2
"""

print(parent_ids.sort_values(by="PARENT_COMPANY_ID").head(15))
print(load_level['CARRIER_SKEY'].drop_duplicates().sort_values().head(15))

"""
Compare CARRIER_SKEY with PARENT_COMPANY_ID
"""

overlap = set(load_level['CARRIER_SKEY']) & set(parent_ids['PARENT_COMPANY_ID'])
print(len(overlap)) # 56?
