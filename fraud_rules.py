import pandas as pd
import numpy as np

def apply_all_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Run all 5 fraud rules. Returns df with risk scores and flags."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    df['flag_velocity']    = rule_velocity(df)
    df['flag_amount']      = rule_amount_anomaly(df)
    df['flag_location']    = rule_impossible_travel(df)
    df['flag_late_night']  = rule_late_night(df)
    df['flag_new_device']  = rule_new_device_high_amount(df)

    # Risk score: each flag = 20 points → max 100
    flag_cols = ['flag_velocity', 'flag_amount', 'flag_location',
                 'flag_late_night', 'flag_new_device']
    df['risk_score'] = df[flag_cols].sum(axis=1) * 20
    df['risk_label'] = pd.cut(
        df['risk_score'],
        bins=[-1, 20, 60, 100],
        labels=['Low', 'Medium', 'High']
    )
    return df


# ── Rule 1: Velocity ────────────────────────────────────────────────────────
def rule_velocity(df: pd.DataFrame) -> pd.Series:
    """Flag users with >5 transactions in any rolling 10-minute window."""
    flagged = set()
    for user, group in df.groupby('user_id'):
        group = group.sort_values('timestamp')
        times = group['timestamp'].values
        for i in range(len(times)):
            window = times[
                (times >= times[i]) &
                (times <= times[i] + np.timedelta64(10, 'm'))
            ]
            if len(window) > 5:
                flagged.update(group.index.tolist())
                break
    return df.index.isin(flagged).astype(int)


# ── Rule 2: Amount Anomaly ───────────────────────────────────────────────────
def rule_amount_anomaly(df: pd.DataFrame) -> pd.Series:
    """Flag transactions >3 std deviations above user's mean amount."""
    user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std']).reset_index()
    user_stats.columns = ['user_id', 'user_mean', 'user_std']
    merged = df.merge(user_stats, on='user_id', how='left')
    merged['user_std'] = merged['user_std'].fillna(1)
    threshold = merged['user_mean'] + 3 * merged['user_std']
    return (merged['amount'] > threshold).astype(int).values


# ── Rule 3: Impossible Travel ────────────────────────────────────────────────
def rule_impossible_travel(df: pd.DataFrame) -> pd.Series:
    """Flag: same user in 2 different cities within 30 minutes."""
    flagged = set()
    for user, group in df.groupby('user_id'):
        group = group.sort_values('timestamp').reset_index()
        for i in range(len(group) - 1):
            time_diff = (group.loc[i+1, 'timestamp'] - group.loc[i, 'timestamp']).seconds / 60
            if time_diff < 30 and group.loc[i, 'location'] != group.loc[i+1, 'location']:
                flagged.add(group.loc[i,   'index'])
                flagged.add(group.loc[i+1, 'index'])
    return df.index.isin(flagged).astype(int)


# ── Rule 4: Late Night ───────────────────────────────────────────────────────
def rule_late_night(df: pd.DataFrame) -> pd.Series:
    """Flag transactions between 2 AM – 5 AM."""
    hour = pd.to_datetime(df['timestamp']).dt.hour
    return ((hour >= 2) & (hour < 5)).astype(int)


# ── Rule 5: New Device + High Amount ────────────────────────────────────────
def rule_new_device_high_amount(df: pd.DataFrame) -> pd.Series:
    """Flag: first-ever transaction from a device AND amount > ₹10,000."""
    first_seen = df.groupby('device_id')['timestamp'].transform('min')
    is_first   = df['timestamp'] == first_seen
    return (is_first & (df['amount'] > 10000)).astype(int)