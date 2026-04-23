"""
Import packages
"""
import pandas as pd
import semopy

"""
Read in data: parent_features_engineered.csv
"""
df = pd.read_csv("/Users/vaibhavjha/Documents/Capstone/Data/parent_features_engineered.csv")
features = df.iloc[:, 52:]

clean_features = features.dropna()

"""
Define model
"""

model_spec = """
Consistency =~ Consistency_1_STD +Consistency_2_STD +Consistency_3_STD +Consistency_4_STD +Consistency_5_STD +Consistency_6_STD +Consistency_7_STD
Volatility =~ Volatility_1_STD +Volatility_2_STD +Volatility_3_STD
Adaptability =~ Adaptability_1_STD +Adaptability_2_STD +Adaptability_3_STD +Adaptability_4_STD +Adaptability_5_STD
ServiceCapacity =~ ServiceCapacity_1_STD +ServiceCapacity_2_STD +ServiceCapacity_3_STD +ServiceCapacity_4_STD
Economical =~ Economical_1_STD +Economical_2_STD
"""

"""
Fit the model
"""
model = semopy.Model(model_spec)
result = model.fit(clean_features)

"""
Inspect output of model
"""
params = model.inspect(std_est=True)

# Keep only factor loadings (op == "~")
loadings = params[params['op'] == '~'][['lval', 'rval', 'Estimate', 'Est. Std', 'Std. Err', 'z-value', 'p-value']]
loadings.columns = ['Item', 'Factor', 'Unstd. Loading', 'Std. Loading', 'Std. Err', 'z-value', 'p-value']
numeric_cols = ['Unstd. Loading', 'Std. Loading', 'Std. Err', 'z-value', 'p-value']
loadings[numeric_cols] = loadings[numeric_cols].apply(pd.to_numeric, errors='coerce')
loadings = loadings.round(2).reset_index(drop=True)

# Model performance
stats = semopy.calc_stats(model)
print("\nModel fit statistics:")
print(stats.T)

print(loadings)
# loadings.to_csv('/Users/vaibhavjha/Documents/Capstone/Data/cfa_loadings.csv')
# RMSEA: 0.115830