"""
Import packages
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity


"""
Read in data: parent_features_engineered.csv
"""

df = pd.read_csv("/Users/elliehuang/Desktop/capstone/data/parent_features_engineered.csv")
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


"""
Load factors
"""
n_factors = 7
fa = FactorAnalyzer(n_factors=n_factors, rotation='oblimin')  # maybe consider varimax/oblimin if we believe factors are uncorrelated/correlated
fa.fit(clean_features)


# Kaiser criterion
ev, cf = fa.get_eigenvalues()
print(ev)

# Proportion of variance by factor
variance, prop_var, cumulative_var = fa.get_factor_variance()

x = np.arange(1, len(ev) + 1)

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(x[:8], ev[:8], marker='o', linewidth=2)
plt.plot(x[7:], ev[7:], marker='o', linewidth=2, color='0.65')
plt.axhline(y=1, color='r', linestyle='--', linewidth=1.8)

plt.xlabel('Factor Number', fontsize=16)
plt.ylabel('Eigenvalue', fontsize=16)
plt.title('Scree Plot - Oblique Rotation', fontsize=18)

plt.text(10.5, 1.05, 'Kaiser criterion', color='r', fontsize=12)

ax = plt.gca()
plt.xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21], fontsize=13)
plt.yticks(fontsize=13)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
# plt.show()
plt.close()

"""
Visualization: cumulative proportion of variance explained with each added factor
"""

plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o')
plt.xlabel('Factor Number')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.title('Cumulative Proportion of Variance Explained Across Factors')
plt.suptitle('Seven factors correspond to approx. 57% variance explained.',
             y=0.98, fontsize=14)
# plt.show()
plt.close()


"""
Visualization: factor loadings heatmap
"""
loadings = pd.DataFrame(fa.loadings_, index=clean_features.columns,
                        columns=[f'Factor {i+1}' for i in range(n_factors)])

plt.figure(figsize=(10, 8))
ax = sns.heatmap(loadings, annot=True, fmt='.2f',
                 cmap='coolwarm', center=0, vmin=-1, vmax=1, annot_kws={"size": 14})
plt.xlabel('Factors', fontsize=12)
plt.ylabel('Raw Features', fontsize=12)
plt.title('Factor Loadings - Oblique Rotation', fontsize=12)
plt.suptitle('The factor loadings for the first seven factors align with the behavioral cohorts we developed.',
             y=0.98, fontsize=14)
plt.tight_layout()
# plt.show()
plt.close()
