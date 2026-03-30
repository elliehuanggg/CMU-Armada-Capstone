"""
Import packages
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, ConfirmatoryFactorAnalyzer, ModelSpecificationParser


"""
Read in data: parent_features_engineered.csv
"""

df = pd.read_csv("/Users/vaibhavjha/Documents/Capstone/Data/parent_features_engineered.csv")
features = df.iloc[:, 52:]

clean_features = features.dropna()

# Specify loadings to confirm
model_dict = {
    "Consistency": ["Consistency_1_STD", "Consistency_2_STD", "Consistency_3_STD", "Consistency_4_STD", "Consistency_5_STD", "Consistency_6_STD", "Consistency_7_STD"],
    "Volatility": ["Volatility_1_STD", "Volatility_2_STD", "Volatility_3_STD"],
    "Adaptability": ["Adaptability_1_STD", "Adaptability_2_STD", "Adaptability_3_STD", "Adaptability_4_STD", "Adaptability_5_STD"],
    "ServiceCapacity": ["ServiceCapacity_1_STD", "ServiceCapacity_2_STD", "ServiceCapacity_3_STD", "ServiceCapacity_4_STD"],
    "Economical": ["Economical_1_STD", "Economical_2_STD"]
}

# Specify model
model_spec = ModelSpecificationParser.parse_model_specification_from_dict(clean_features, model_dict)

# Define CFA
cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
# Run CFA
cfa.fit(clean_features.values)

# Print output
print(cfa.loadings_**2)
print(cfa.factor_varcovs_)
# print(cfa.error_vars_)
