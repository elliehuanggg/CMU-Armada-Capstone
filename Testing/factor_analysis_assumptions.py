import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
import scipy.stats as stats

"""
Read in data
"""
df = pd.read_csv('/Users/vaibhavjha/Documents/Capstone/Data/parent_features_engineered.csv')


# Assumption: Multivariate Normality
"""
Univariate standardized features' distributions
"""
print(df.columns)
cols_of_interest = ['Consistency_1_STD', 'Consistency_2_STD', 'Consistency_3_STD', 'Consistency_4_STD', 'Consistency_5_STD', 'Consistency_6_STD', 'Consistency_7_STD', 
                    'Volatility_1_STD', 'Volatility_2_STD', 
                    'Adaptability_1_STD', 'Adaptability_2_STD', 'Adaptability_3_STD', 'Adaptability_4_STD', 'Adaptability_5_STD', 
                    'ServiceCapacity_1_STD', 'ServiceCapacity_2_STD', 'ServiceCapacity_3_STD', 'ServiceCapacity_4_STD']
"""
for i in cols_of_interest:
    plt.close()
    plt.hist(df[i], bins=15)
    plt.xlim(-4,4)
    plt.title(f"Histogram of {i}")
    plt.xlabel("Standardized value (z-score)")
    plt.ylabel("Frequency")
    plt.show()
"""

# Largely skewed

"""
Henze–Zirkler Test
"""
df_of_interest = df[['Consistency_1_STD', 'Consistency_2_STD', 'Consistency_3_STD', 'Consistency_4_STD', 'Consistency_5_STD', 'Consistency_6_STD', 'Consistency_7_STD',
                    'Volatility_1_STD', 'Volatility_2_STD', 
                    'Adaptability_1_STD', 'Adaptability_2_STD', 'Adaptability_3_STD', 'Adaptability_4_STD', 'Adaptability_5_STD', 
                    'ServiceCapacity_1_STD', 'ServiceCapacity_2_STD', 'ServiceCapacity_3_STD', 'ServiceCapacity_4_STD']]

print(pg.multivariate_normality(df_of_interest, alpha=0.05))

"""
QQPlot of Mahalanobis distances
"""
X = df_of_interest.dropna().values
n, p = X.shape
mean = np.mean(X, axis=0)
cov = np.cov(X, rowvar=False)
S_inv = np.linalg.inv(cov)
Xc = X - mean

# Mahalanobis distances
md = np.array([x @ S_inv @ x.T for x in Xc])

# Q-Q plot against chi-square
stats.probplot(md, dist="chi2", sparams=(p,), plot=plt)
plt.title("Q-Q Plot (Mahalanobis distances)")
plt.show()