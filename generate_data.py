import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n = 10000

print("Generating synthetic UPI transaction data...")

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

# Inject fraud — 5% of transactions
fraud_indices = np.random.choice(n, size=int(n * 0.05), replace=False)
df.loc[fraud_indices, 'amount'] = (
    df.loc[fraud_indices, 'amount'] * np.random.uniform(3, 10, size=len(fraud_indices))
).round(2)
df['is_fraud'] = df.index.isin(fraud_indices)

df.to_csv('transactions.csv', index=False)
print(f"✅ Saved transactions.csv ({len(df)} rows, {df['is_fraud'].sum()} fraud cases)")