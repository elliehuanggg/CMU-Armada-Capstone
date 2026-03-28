"""
Import packages
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

"""
Read in data: parent_features_engineered.csv
"""

df = pd.read_csv("/Users/vaibhavjha/Documents/Capstone/Data/parent_features_engineered.csv")
features = df.iloc[:, 31:52]

"""
Determine whether data are suitable for factor analysis
"""
# Assess correlation between columns (features)

corr = features.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle

labels = corr.applymap(lambda x: f'{x:.2f}' if abs(x) >= 0.3 else '')  # annotate correlations where abs value is >= 0.3
plt.figure(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=labels, fmt='', cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
# plt.show()
plt.close()

print(corr.round(2))


# Conduct Bartlett's sphericity test

clean_features = features.dropna()
chi_square, p_value = calculate_bartlett_sphericity(clean_features)
print(f'Barlett p-value: {p_value}')
# Output: 0.0


# Conduct KMO test

kmo_per_variable, kmo_total = calculate_kmo(clean_features)
print(f'KMO score: {kmo_total}')
# Output: 0.72


"""
Load factors
"""
n_factors = 7
fa = FactorAnalyzer(n_factors=n_factors, rotation='oblimin')  # maybe consider varimax/oblimin if we believe factors are uncorrelated/correlated
fa.fit(clean_features)

"""
Choose number of factors
"""
# Kaiser criterion
ev, cf = fa.get_eigenvalues()
print(ev)

# Proportion of variance by factor
print(fa.get_factor_variance())

# Scree plot
plt.plot(range(1, len(ev) + 1), ev, marker='o')
plt.axhline(y=1, color='r', linestyle='--', label='Kaiser criterion')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot - Oblique Rotation')
plt.legend()
plt.show()
plt.close()


"""
Factor loadings heatmap
"""
loadings = pd.DataFrame(fa.loadings_, index=clean_features.columns,
                        columns=[f'Factor {i+1}' for i in range(n_factors)])

plt.figure(figsize=(10, 8))
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            vmin=-1, vmax=1)
plt.title('Factor Loadings - Oblique Rotation')
plt.tight_layout()
plt.show()
