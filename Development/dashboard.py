import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Armada Carrier Behavior Weight Explorer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    div[data-testid="column"]:first-child .stSlider { margin-bottom: 0.1rem; }
    .stSlider label { font-size: 0.78rem !important; }
    div[data-testid="metric-container"] { padding: 0.3rem 0.5rem; }
    .eff-weight {
        display: inline-block;
        background: #e8f0fe;
        color: #1a56db;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 6px;
        padding: 1px 7px;
        margin-top: 2px;
        margin-bottom: 6px;
        white-space: nowrap;
    }
    .eff-weight-changed {
        background: #fef3c7;
        color: #b45309;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('parent_features_engineered.csv')
    df = df.set_index('PARENT_COMPANY_ID')
    return df


def compute_weighted_score(df, feature_cols, weights):
    """Weighted average of _STD columns; renormalize over non-NaN features per row."""
    weight_arr = np.array(weights, dtype=float)
    scores = []
    for _, row in df.iterrows():
        vals = np.array([row.get(f, np.nan) for f in feature_cols], dtype=float)
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            scores.append(np.nan)
        else:
            w = weight_arr[mask]
            w = w / w.sum()
            scores.append((vals[mask] * w).sum())
    return np.array(scores)


FEATURE_NAMES = {
    'Consistency_1_STD': 'Quantity fulfillment',
    'Consistency_2_STD': 'No-claim rate',
    'Consistency_3_STD': 'On-time pickup rate',
    'Consistency_4_STD': 'On-time drop-off rate',
    'Consistency_5_STD': 'Transit-standard rate',
    'Consistency_6_STD': 'Waterfall share',
    'Consistency_7_STD': 'Timing difference',
    'Volatility_1_STD': 'Spot exposure',
    'Volatility_2_STD': 'Market alignment',
    'Volatility_3_STD': 'Reversed tenure',
    'Adaptability_1_STD': 'Temperature-zone balance',
    'Adaptability_2_STD': 'Mileage CV',
    'Adaptability_3_STD': 'Actual-quantity CV',
    'Adaptability_4_STD': 'Award-type balance',
    'Adaptability_5_STD': 'Linehaul CV',
    'ServiceCapacity_1_STD': 'Actual quantity moved',
    'ServiceCapacity_2_STD': 'Load share',
    'ServiceCapacity_3_STD': 'Total mileage',
    'ServiceCapacity_4_STD': 'Total paid linehaul',
    'Economical_1_STD': 'NAP benchmark gap',
    'Economical_2_STD': 'DAT benchmark gap',
}


def cohort_section(title, feature_cols, default_weights, df):
    st.markdown(f"#### {title}")

    left_col, hist_col, table_col = st.columns([1.8, 2, 2])

    with left_col:
        # Read current slider values from session_state (available after first render)
        current_raws = []
        for i in range(len(feature_cols)):
            key = f"{title}_{i}"
            val = st.session_state.get(key, default_weights[i])
            current_raws.append(val)

        cur_total = sum(current_raws)
        cur_needs_norm = cur_total > 0 and abs(cur_total - 1.0) > 0.01
        cur_eff = [w / cur_total for w in current_raws] if cur_total > 0 else list(current_raws)

        raw_weights = []
        for i, feat in enumerate(feature_cols):
            label = FEATURE_NAMES.get(feat, feat.replace('_STD', '').replace('_', ' '))
            s_col, b_col = st.columns([3, 1])
            with s_col:
                w = st.slider(label, 0.0, 1.0, default_weights[i], 0.01,
                              format="%.2f", key=f"{title}_{i}")
                raw_weights.append(w)
            with b_col:
                css_class = "eff-weight eff-weight-changed" if cur_needs_norm else "eff-weight"
                st.markdown(
                    f'<span class="{css_class}">↳ {cur_eff[i]:.1%}</span>',
                    unsafe_allow_html=True
                )

        total = sum(raw_weights)
        needs_norm = total > 0 and abs(total - 1.0) > 0.01
        eff_weights = [w / total for w in raw_weights] if total > 0 else list(raw_weights)

        if needs_norm:
            st.warning(f"Weights were normalized because they do not add to 1 (sum was {total:.2f})")

    scores = pd.Series(
        compute_weighted_score(df, feature_cols, eff_weights),
        index=df.index
    ).dropna()

    with hist_col:
        fig = px.histogram(
            x=scores,
            nbins=25,
            title="Weighted Score Distribution",
            labels={'x': 'Weighted Score', 'y': 'Count'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=280, margin=dict(t=35, b=20, l=20, r=10),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        top = scores.nlargest(len(scores)).reset_index()
        top.columns = ['Carrier ID', 'Score']
        top['Score'] = top['Score'].round(3)
        bot = scores.nsmallest(len(scores)).reset_index()
        bot.columns = ['Carrier ID', 'Score']
        bot['Score'] = bot['Score'].round(3)

        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**🏆 Top Carriers**")
            st.dataframe(top, use_container_width=True, hide_index=True, height=210)
        with t2:
            st.markdown("**📉 Bottom Carriers**")
            st.dataframe(bot, use_container_width=True, hide_index=True, height=210)

    st.divider()


def main():
    st.markdown('<h1 class="main-header">Armada Carrier Behavior Weight Explorer</h1>',
                unsafe_allow_html=True)

    df = load_data()

    cohorts = {
        "Consistency (Variable ↔ Reliable)": {
            "features": ['Consistency_1_STD', 'Consistency_2_STD', 'Consistency_3_STD',
                         'Consistency_4_STD', 'Consistency_5_STD', 'Consistency_6_STD',
                         'Consistency_7_STD'],
            "weights": [0.40, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05]
        },
        "Volatility (Committed ↔ Opportunistic)": {
            "features": ['Volatility_1_STD', 'Volatility_2_STD', 'Volatility_3_STD'],
            "weights": [0.45, 0.45, 0.10]
        },
        "Adaptability (Specialist ↔ Flexible)": {
            "features": ['Adaptability_1_STD', 'Adaptability_2_STD', 'Adaptability_3_STD',
                         'Adaptability_4_STD', 'Adaptability_5_STD'],
            "weights": [0.30, 0.30, 0.15, 0.15, 0.10]
        },
        "Service Capacity (Fringe ↔ Heavy-Duty)": {
            "features": ['ServiceCapacity_1_STD', 'ServiceCapacity_2_STD',
                         'ServiceCapacity_3_STD', 'ServiceCapacity_4_STD'],
            "weights": [0.50, 0.30, 0.10, 0.10]
        },
        "Economical (Costly ↔ Friendly)": {
            "features": ['Economical_1_STD', 'Economical_2_STD'],
            "weights": [0.60, 0.40]
        }
    }

    with st.sidebar:
        st.header("🎛️ How to Use")
        st.markdown("""
        1. Adjust sliders per feature  
        2. Histogram shows the **weighted score** distribution  
        3. Weights auto-renormalize for missing values  
        4. Top/Bottom 5 carriers update live  
        """)
        st.info(f"**{len(df)} carriers** loaded")
        if st.button("Reset All Weights"):
            for t, cfg in cohorts.items():
                for i, default_w in enumerate(cfg["weights"]):
                    st.session_state[f"{t}_{i}"] = default_w
            st.rerun()

    for title, cfg in cohorts.items():
        cohort_section(title, cfg["features"], cfg["weights"], df)


if __name__ == "__main__":
    main()