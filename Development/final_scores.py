import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv("parent_features_engineered.csv")
df = df.rename(columns={"PARENT_COMPANY_ID": "carrier_id"})

# ------------------------------------------------------------
# Helper: weighted score with dynamic renormalization
# ------------------------------------------------------------
def weighted_score_with_missing(row, weight_dict):
    """
    Compute a weighted score for one row.
    If some features are NaN, drop them and renormalize
    remaining weights to sum to 1.
    """
    values = {}
    weights = {}

    for col, w in weight_dict.items():
        val = row.get(col, np.nan)
        if pd.notna(val):
            values[col] = val
            weights[col] = w

    if not values:  # all features missing
        return np.nan

    total_w = sum(weights.values())
    norm_weights = {c: w / total_w for c, w in weights.items()}

    return sum(values[c] * norm_weights[c] for c in values.keys())

# ------------------------------------------------------------
# Define base weights for each cohort (as given)
# ------------------------------------------------------------

consistency_weights = {
    "Consistency_1_STD": 0.40,
    "Consistency_2_STD": 0.20,
    "Consistency_3_STD": 0.10,
    "Consistency_4_STD": 0.10,
    "Consistency_5_STD": 0.10,
    "Consistency_6_STD": 0.05,
    "Consistency_7_STD": 0.05,
}

volatility_weights = {
    "Volatility_1_STD": 0.45,
    "Volatility_2_STD": 0.45,
    "Volatility_3_STD": 0.10,
}

adaptability_weights = {
    "Adaptability_1_STD": 0.30,
    "Adaptability_2_STD": 0.30,
    "Adaptability_3_STD": 0.15,
    "Adaptability_4_STD": 0.15,
    "Adaptability_5_STD": 0.10,
}

service_capacity_weights = {
    "ServiceCapacity_1_STD": 0.50,
    "ServiceCapacity_2_STD": 0.30,
    "ServiceCapacity_3_STD": 0.10,
    "ServiceCapacity_4_STD": 0.10,
}

economical_weights = {
    "Economical_1_STD": 0.60,
    "Economical_2_STD": 0.40,
}

# ------------------------------------------------------------
# Compute cohort scores
# ------------------------------------------------------------

df["consistency_score"] = df.apply(
    weighted_score_with_missing, axis=1, weight_dict=consistency_weights
)

df["volatility_score"] = df.apply(
    weighted_score_with_missing, axis=1, weight_dict=volatility_weights
)

df["adaptability_score"] = df.apply(
    weighted_score_with_missing, axis=1, weight_dict=adaptability_weights
)

df["service_capacity_score"] = df.apply(
    weighted_score_with_missing, axis=1, weight_dict=service_capacity_weights
)

df["economical_score"] = df.apply(
    weighted_score_with_missing, axis=1, weight_dict=economical_weights
)

# ------------------------------------------------------------
# Compute percentile ranks for each cohort score
# ------------------------------------------------------------

def percentile_col(series):
    return series.rank(pct=True) * 100

df["consistency_percentile"] = percentile_col(df["consistency_score"])
df["volatility_percentile"] = percentile_col(df["volatility_score"])
df["adaptability_percentile"] = percentile_col(df["adaptability_score"])
df["service_capacity_percentile"] = percentile_col(df["service_capacity_score"])
df["economical_percentile"] = percentile_col(df["economical_score"])

# ------------------------------------------------------------
# Save behavioral_scores.csv
# ------------------------------------------------------------

behavioral_scores = df[
    [
        "carrier_id",
        "consistency_score",
        "volatility_score",
        "adaptability_score",
        "service_capacity_score",
        "economical_score",
        "consistency_percentile",
        "volatility_percentile",
        "adaptability_percentile",
        "service_capacity_percentile",
        "economical_percentile",
    ]
].copy()

behavioral_scores.to_csv("behavioral_scores.csv", index=False)

# ------------------------------------------------------------
# Plot distributions for all five scores
# ------------------------------------------------------------

def plot_score_distribution(series, title):
    plt.figure(figsize=(7, 4))
    plt.hist(series.dropna(), bins=30, color="navy")
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Number of carriers")
    plt.tight_layout()
    plt.show()

plot_score_distribution(
    behavioral_scores["consistency_score"],
    "The bulk of carriers are similar in reliability, but some are exceptionally reliable."
)

plot_score_distribution(
    behavioral_scores["volatility_score"],
    "Most carriers are commited, but some are highly opportunistic."
)

plot_score_distribution(
    behavioral_scores["adaptability_score"],
    "There are two clusters of flexible and specialist carriers. Some carriers are incredibly flexible."
)

plot_score_distribution(
    behavioral_scores["service_capacity_score"],
    "Relatively few carriers are truly heavy-duty."
)

plot_score_distribution(
    behavioral_scores["economical_score"],
    "Only a few carriers are overly costly."
)

