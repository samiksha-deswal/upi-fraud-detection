import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from fraud_rules import apply_all_rules

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide"
)

# ── Load & process data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 10000

    timestamps = [
        datetime.now() - timedelta(
            days=int(np.random.randint(0, 30)),
            hours=int(np.random.randint(0, 24)),
            minutes=int(np.random.randint(0, 60))
        )
        for _ in range(n)
    ]

    df = pd.DataFrame({
        'transaction_id':    [f'TXN{i:06d}' for i in range(n)],
        'timestamp':         timestamps,
        'amount':            np.random.lognormal(5, 2, n).round(2),
        'user_id':           [f'USER{np.random.randint(1, 500):04d}' for _ in range(n)],
        'merchant_category': np.random.choice(['Food', 'Shopping', 'Bills', 'Transfer', 'Entertainment'], n),
        'device_id':         [f'DEVICE{np.random.randint(1, 600):04d}' for _ in range(n)],
        'location':          np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Hyderabad'], n),
    })

    # Inject fraud — 5%
    fraud_indices = np.random.choice(n, size=int(n * 0.05), replace=False)
    df.loc[fraud_indices, 'amount'] = (
        df.loc[fraud_indices, 'amount'] * np.random.uniform(3, 10, size=len(fraud_indices))
    ).round(2)
    df['is_fraud'] = df.index.isin(fraud_indices)

    df = apply_all_rules(df)
    return df

df = load_data()
flagged = df[df['risk_score'] > 0]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🚨 UPI Transaction Fraud Detection Dashboard")
st.caption("Rule-based real-time fraud detection · 10,000 synthetic transactions")

# KPI strip
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Flagged",
            f"{len(flagged):,}",
            delta=f"{len(flagged)/len(df)*100:.1f}%",
            delta_color="inverse")
col3.metric("High Risk",
            f"{len(df[df['risk_label']=='High']):,}",
            delta_color="inverse")
col4.metric("Fraud Injected", f"{df['is_fraud'].sum():,}")
col5.metric("Avg Risk Score", f"{flagged['risk_score'].mean():.0f}/100")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🚨 Real-time Alerts", "📊 Pattern Analysis", "🔍 User Drill-down"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Real-time Alerts
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Flagged Transactions")

    c1, c2, c3 = st.columns(3)
    risk_filter = c1.multiselect(
        "Risk Level",
        ['High', 'Medium', 'Low'],
        default=['High', 'Medium']
    )
    min_amount = c2.number_input("Min Amount (₹)", value=0, step=1000)
    sort_by    = c3.selectbox("Sort by", ['risk_score', 'amount', 'timestamp'])

    view = df[
        (df['risk_label'].isin(risk_filter)) &
        (df['amount'] >= min_amount)
    ].sort_values(sort_by, ascending=False)

    def color_risk(val):
        colors = {
            'High':   'background-color:#ff4b4b; color:white',
            'Medium': 'background-color:#ffa500; color:white',
            'Low':    'background-color:#2ecc71; color:white'
        }
        return colors.get(val, '')

    display_cols = [
        'transaction_id', 'timestamp', 'user_id', 'amount',
        'merchant_category', 'location', 'risk_score', 'risk_label',
        'flag_velocity', 'flag_amount', 'flag_location',
        'flag_late_night', 'flag_new_device'
    ]

    st.dataframe(
        view[display_cols].style.applymap(color_risk, subset=['risk_label']),
        use_container_width=True,
        height=450
    )

    st.caption(f"Showing {len(view):,} transactions matching filters")

    csv = view[display_cols].to_csv(index=False)
    st.download_button(
        "⬇️ Download Flagged Transactions",
        csv,
        "flagged_transactions.csv",
        "text/csv"
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Pattern Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Fraud Patterns")

    col_a, col_b = st.columns(2)

    # Chart 1: Fraud by hour
    with col_a:
        df['hour'] = df['timestamp'].dt.hour
        hourly = df.groupby('hour')['risk_score'].agg(
            flagged=lambda x: (x > 0).sum(),
            total='count'
        ).reset_index()
        hourly['rate'] = (hourly['flagged'] / hourly['total'] * 100).round(1)

        fig1 = px.bar(
            hourly, x='hour', y='flagged',
            color='rate', color_continuous_scale='Reds',
            title="🕐 Flagged Transactions by Hour of Day",
            labels={'hour': 'Hour', 'flagged': 'Flagged Count', 'rate': 'Flag Rate %'}
        )
        fig1.add_vrect(
            x0=2, x1=5,
            fillcolor="red", opacity=0.1,
            annotation_text="Late Night Risk Zone"
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Fraud by category
    with col_b:
        cat_stats = df.groupby('merchant_category').agg(
            total=('transaction_id', 'count'),
            flagged=('risk_score', lambda x: (x > 0).sum())
        ).reset_index()
        cat_stats['rate'] = (cat_stats['flagged'] / cat_stats['total'] * 100).round(1)

        fig2 = px.bar(
            cat_stats, x='merchant_category', y='rate',
            color='rate', color_continuous_scale='Oranges',
            title="📂 Flag Rate by Merchant Category (%)",
            labels={'merchant_category': 'Category', 'rate': 'Flag Rate %'}
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    # Chart 3: Amount distribution — fraud vs normal
    with col_c:
        sample_normal  = df[df['risk_score'] == 0]['amount'].sample(500, random_state=42)
        sample_flagged = df[df['risk_score'] >  0]['amount'].sample(
            min(500, len(df[df['risk_score'] > 0])), random_state=42
        )

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=sample_normal, name='Normal',
            opacity=0.6, marker_color='steelblue', nbinsx=40
        ))
        fig3.add_trace(go.Histogram(
            x=sample_flagged, name='Flagged',
            opacity=0.7, marker_color='red', nbinsx=40
        ))
        fig3.update_layout(
            barmode='overlay',
            title="💰 Amount Distribution: Normal vs Flagged",
            xaxis_title="Amount (₹)",
            yaxis_title="Count"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Rules breakdown pie
    with col_d:
        rule_counts = {
            'Velocity':    int(df['flag_velocity'].sum()),
            'Amt Anomaly': int(df['flag_amount'].sum()),
            'Travel':      int(df['flag_location'].sum()),
            'Late Night':  int(df['flag_late_night'].sum()),
            'New Device':  int(df['flag_new_device'].sum()),
        }
        fig4 = px.pie(
            values=list(rule_counts.values()),
            names=list(rule_counts.keys()),
            title="🔍 Flags Triggered by Rule",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Chart 5: Fraud by city
    st.subheader("🗺️ Flagged Transactions by City")
    city_stats = df.groupby('location').agg(
        total=('transaction_id', 'count'),
        flagged=('risk_score', lambda x: (x > 0).sum()),
        high_risk=('risk_label', lambda x: (x == 'High').sum())
    ).reset_index()
    city_stats['flag_rate'] = (city_stats['flagged'] / city_stats['total'] * 100).round(1)

    fig5 = px.bar(
        city_stats, x='location', y=['flagged', 'high_risk'],
        barmode='group',
        title="Flagged vs High-Risk by City",
        labels={'value': 'Count', 'location': 'City'}
    )
    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — User Drill-down
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("User Risk Profile")

    default_user = df[df['risk_score'] > 0]['user_id'].value_counts().idxmax()
    user_input = st.text_input("Enter User ID (e.g. USER0042)", value=default_user)

    user_df = df[df['user_id'] == user_input.strip().upper()]

    if user_df.empty:
        st.warning("⚠️ No transactions found for this user ID.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", len(user_df))
        c2.metric("Flagged",
                  f"{(user_df['risk_score'] > 0).sum()} "
                  f"({(user_df['risk_score'] > 0).mean()*100:.0f}%)")
        c3.metric("Avg Transaction", f"₹{user_df['amount'].mean():,.0f}")
        c4.metric("Max Risk Score",  f"{user_df['risk_score'].max()}/100")

        if 'High' in user_df['risk_label'].values:
            st.error("⛔ HIGH RISK USER — Recommend immediate account review")
        elif 'Medium' in user_df['risk_label'].values:
            st.warning("⚠️ MEDIUM RISK USER — Monitor closely")
        else:
            st.success("✅ LOW RISK USER — No immediate action needed")

        st.markdown("#### Transaction Timeline")
        fig6 = px.scatter(
            user_df.sort_values('timestamp'),
            x='timestamp', y='amount',
            color='risk_label',
            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
            size='risk_score',
            size_max=20,
            hover_data=['transaction_id', 'merchant_category', 'location', 'device_id'],
            title=f"Transaction Timeline — {user_input.upper()}",
        )
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown("#### All Transactions")
        show_cols = [
            'transaction_id', 'timestamp', 'amount', 'merchant_category',
            'location', 'device_id', 'risk_score', 'risk_label',
            'flag_velocity', 'flag_amount', 'flag_location',
            'flag_late_night', 'flag_new_device'
        ]
        st.dataframe(
            user_df[show_cols].sort_values('timestamp', ascending=False),
            use_container_width=True
        )

        st.markdown("#### 🔍 Rules Triggered for This User")
        user_flags = {
            'Velocity':    int(user_df['flag_velocity'].sum()),
            'Amt Anomaly': int(user_df['flag_amount'].sum()),
            'Travel':      int(user_df['flag_location'].sum()),
            'Late Night':  int(user_df['flag_late_night'].sum()),
            'New Device':  int(user_df['flag_new_device'].sum()),
        }
        fig7 = px.bar(
            x=list(user_flags.keys()),
            y=list(user_flags.values()),
            color=list(user_flags.values()),
            color_continuous_scale='Reds',
            title=f"Flag Triggers — {user_input.upper()}",
            labels={'x': 'Rule', 'y': 'Times Triggered'}
        )
        st.plotly_chart(fig7, use_container_width=True)