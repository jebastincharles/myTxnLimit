"""
Full pipeline (Option B):
 - Generate customer profiles
 - Generate transaction table with injected behavior
 - Compute transaction-level flags (burst, recurring, geo_velocity, etc.)
 - Aggregate to customer-level features
 - Synthesize daily_limit target
 - Train 3 models (GBR, XGBoost, MLP) and evaluate
 - Save outputs (CSV + feature importance PNGs)
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------------
# Config / Seedou
# -------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Size / performance knobsiiscan ypu send me full log
N_CUSTOMERS = 1000                # reduce if you want faster runs
AVG_TXNS_PER_CUSTOMER = 6000       # average transactions per customer
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)

# -------------------------
# 1) Customer profiles
# -------------------------
customer_ids = [f"CUST{i:05d}" for i in range(1, N_CUSTOMERS+1)]
account_tenure = np.random.randint(0, 20, N_CUSTOMERS)  # years
annual_income = np.random.normal(60000, 25000, N_CUSTOMERS).clip(10000, 300000)
credit_score = np.random.normal(680, 60, N_CUSTOMERS).clip(300, 850).round().astype(int)

# monthly incomes for last 12 months -> income stability
income_monthly = [np.random.normal(inc/12, (abs(np.random.normal(inc/12))*0.1)+50, 12) for inc in annual_income]
income_stability = np.array([np.std(m) for m in income_monthly])

# loan defaults, device change rate, ip risk score
loan_defaults = np.random.poisson(0.05, N_CUSTOMERS)
device_change_rate = np.random.beta(1.2, 10, N_CUSTOMERS)
ip_risk_score = np.random.beta(1, 12, N_CUSTOMERS)

customer_df = pd.DataFrame({
    "customer_id": customer_ids,
    "account_tenure_years": account_tenure,
    "annual_income": annual_income.round(2),
    "credit_score": credit_score,
    "income_stability": income_stability.round(2),
    "loan_defaults": loan_defaults,
    "device_change_rate": device_change_rate.round(3),
    "ip_risk_score": ip_risk_score.round(3)
})

# -------------------------
# 2) Transaction generation (with behavior)
# -------------------------
merchant_categories = ["Groceries", "Electronics", "Travel", "Dining", "Clothing",
                       "Healthcare", "Fuel", "Entertainment", "Utilities", "Luxury"]
cities = ["New York", "London", "Singapore", "Mumbai", "Sydney", "Dubai",
          "Berlin", "Toronto", "São Paulo", "Johannesburg"]

# mark some cities as having slightly higher fraud risk
country_risk = {c: np.random.choice([0,1], p=[0.9,0.1]) for c in cities}

all_txns = []

for idx, row in customer_df.iterrows():
    cid = row["customer_id"]
    inc = row["annual_income"]
    # profile based on income
    if inc > 120000:
        profile = np.random.choice(["high","moderate","low"], p=[0.4,0.5,0.1])
    elif inc < 30000:
        profile = np.random.choice(["high","moderate","low"], p=[0.05,0.45,0.5])
    else:
        profile = np.random.choice(["high","moderate","low"], p=[0.2,0.6,0.2])

    mean_txn = {"high": AVG_TXNS_PER_CUSTOMER*1.5,
                "moderate": AVG_TXNS_PER_CUSTOMER,
                "low": AVG_TXNS_PER_CUSTOMER*0.6}[profile]
    txn_count = max(5, int(np.random.poisson(mean_txn)))

    # choose 1-2 home cities
    home_cities = random.sample(cities, k=2) if np.random.rand() < 0.2 else [random.choice(cities)]
    last_txn_time = None
    last_txn_city = home_cities[0]

    for t in range(txn_count):
        txn_id = f"TXN_{cid}_{t+1}"
        delta_days = (END_DATE - START_DATE).days
        txn_date = START_DATE + timedelta(days=np.random.randint(0, delta_days),
                                          hours=np.random.randint(0,24),
                                          minutes=np.random.randint(0,60))
        base_scale = inc / 200 if profile=="moderate" else (inc/120 if profile=="high" else inc/400)
        amount = np.random.exponential(scale=base_scale)
        if np.random.rand() < 0.005:
            amount *= np.random.uniform(5, 20)
        amount = round(min(amount, inc*0.6),2)

        merchant = np.random.choice(merchant_categories,
                                    p=[0.18,0.08,0.06,0.12,0.10,0.06,0.07,0.12,0.12,0.09])

        if np.random.rand() < 0.9:
            city = last_txn_city if np.random.rand() < 0.8 else home_cities[0]
        else:
            city = random.choice([c for c in cities if c != home_cities[0]])

        geo_velocity = 0.0
        if last_txn_time is not None and city != last_txn_city:
            hours_diff = abs((txn_date - last_txn_time).total_seconds())/3600.0 + 1e-6
            if hours_diff < 24:
                geo_velocity = np.random.uniform(500, 10000) / hours_diff

        last_txn_time = txn_date
        last_txn_city = city

        fraud_prob = 0.01
        if geo_velocity > 300:
            fraud_prob += 0.15
        if country_risk.get(city,0) == 1:
            fraud_prob += 0.03
        if amount > inc * 0.25:
            fraud_prob += 0.05

        is_fraud = np.random.rand() < fraud_prob
        is_night = txn_date.hour < 6 or txn_date.hour > 22
        is_weekend = txn_date.weekday() >= 5

        all_txns.append({
            "transaction_id": txn_id,
            "customer_id": cid,
            "transaction_date": txn_date,
            "amount": amount,
            "merchant_category": merchant,
            "city": city,
            "geo_velocity": round(geo_velocity,2),
            "is_fraud": int(is_fraud),
            "is_night": int(is_night),
            "is_weekend": int(is_weekend)
        })

transactions = pd.DataFrame(all_txns).sort_values(["customer_id","transaction_date"]).reset_index(drop=True)

# -------------------------
# 3) Transaction flags: bursts, recurring
# -------------------------
transactions['txn_time_minute'] = transactions['transaction_date'].dt.floor('min')
burst_counts = transactions.groupby(['customer_id', 'txn_time_minute'])['transaction_id'].count().reset_index(name='burst_count')
transactions = transactions.merge(burst_counts, on=['customer_id','txn_time_minute'], how='left')
transactions['burst_flag'] = (transactions['burst_count'] >= 3).astype(int)

# recurring merchant: merchant with >=3 txns for a customer (simple proxy)
merchant_counts = transactions.groupby(['customer_id','merchant_category'])['transaction_id'].count().reset_index(name='mcount')
merchant_counts['is_recurring_merchant'] = (merchant_counts['mcount'] >= 3).astype(int)
transactions = transactions.merge(merchant_counts[['customer_id','merchant_category','is_recurring_merchant']],
                                  on=['customer_id','merchant_category'], how='left')

# -------------------------
# 4) Aggregate to customer-level features
# -------------------------
agg_funcs = {
    'amount': ['mean','std','max','sum'],
    'transaction_id': 'count',
    'is_fraud': ['sum','mean'],
    'geo_velocity': ['max','mean'],
    'burst_flag': 'sum',
    'is_night': 'mean',
    'is_weekend': 'mean',
    'is_recurring_merchant': 'mean'
}
txn_agg = transactions.groupby('customer_id').agg(agg_funcs)
txn_agg.columns = ['_'.join(col).strip() for col in txn_agg.columns.values]
txn_agg.reset_index(inplace=True)

# derived metrics
txn_agg['spend_volatility'] = transactions.groupby('customer_id')['amount'].std().fillna(0).values
recent_spike = []
for cid, group in transactions.groupby('customer_id'):
    amounts = group['amount'].values
    if len(amounts) < 5:
        recent_spike.append(0.0)
        continue
    split = int(len(amounts)*0.9)
    prior_avg = amounts[:max(1,split)].mean()
    recent_avg = amounts[split:].mean() if len(amounts[split:])>0 else prior_avg
    recent_spike.append(max(0.0, (recent_avg - prior_avg) / (prior_avg+1e-6)))
txn_agg['recent_spike_ratio'] = recent_spike

merchant_div = transactions.groupby('customer_id')['merchant_category'].nunique().reset_index(name='merchant_diversity')
txn_agg = txn_agg.merge(merchant_div, on='customer_id', how='left')

transactions['high_risk_country'] = transactions['city'].map(country_risk).fillna(0).astype(int)
high_risk_ratio = transactions.groupby('customer_id')['high_risk_country'].mean().reset_index(name='high_risk_country_ratio')
txn_agg = txn_agg.merge(high_risk_ratio, on='customer_id', how='left')

txn_agg['txn_per_day'] = txn_agg['transaction_id_count'] / 365.0
txn_agg['peer_spend_percentile'] = txn_agg['amount_sum'].rank(pct=True)
txn_agg.fillna(0, inplace=True)

# -------------------------
# 5) Merge features into full dataset
# -------------------------
full = customer_df.merge(txn_agg, on='customer_id', how='left')
full.fillna(0, inplace=True)

# -------------------------
# 6) Synthesize daily_limit (target)
# -------------------------
def synth_daily_limit(row):
    base = row['annual_income'] / 12 * 0.12
    credit_factor = (row['credit_score'] - 300) / 550
    tenure_factor = min(1.5, 1 + row['account_tenure_years'] / 20)
    fraud_penalty = 1 - min(0.8, row.get('is_fraud_sum', 0) * 0.2)
    geo_penalty = 1 - (0.4 if row.get('geo_velocity_max', 0) > 200 else 0)
    ip_penalty = 1 - min(0.5, row.get('ip_risk_score',0) * 0.8)
    spike_penalty = 1 - min(0.6, row.get('recent_spike_ratio',0) * 0.8)
    volatility_penalty = 1 - min(0.6, (row.get('spend_volatility',0) / (row.get('amount_mean',1)+1e-6)) * 0.5)
    merchant_factor = 1 + (row.get('merchant_diversity',0) / 10.0)
    peer_factor = 1 + (row.get('peer_spend_percentile',0) - 0.5)
    limit = base * (0.8 + credit_factor * 0.8) * tenure_factor * fraud_penalty * ip_penalty * spike_penalty * volatility_penalty * merchant_factor * peer_factor
    limit = np.clip(limit, 500, 100000)
    if row.get('loan_defaults',0) > 0:
        limit *= 0.7 ** row['loan_defaults']
    return round(limit * np.random.uniform(0.9, 1.1), 2)

full['daily_limit'] = full.apply(synth_daily_limit, axis=1)

# Optionally save intermediate outputs
full.to_csv("./dataset/model2/synthetic_customers_with_features_and_limits.csv", index=False)
transactions.to_csv("./dataset/model2/synthetic_transactions.csv", index=False)
customer_df.to_csv("./dataset/model2/synthetic_customers.csv", index=False)

# -------------------------
# 7) Prepare data for ML
# -------------------------
model_df = full.copy()
drop_cols = ['customer_id']
X = model_df.drop(columns=drop_cols + ['daily_limit'])
y = model_df['daily_limit']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Scale for NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 8) Train models
# -------------------------
gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=SEED)
gbr.fit(X_train, y_train)

xgb = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=SEED, verbosity=0)
xgb.fit(X_train, y_train)

nn = MLPRegressor(hidden_layer_sizes=(128,64), activation='relu', solver='adam', max_iter=400, random_state=SEED)
nn.fit(X_train_scaled, y_train)

# -------------------------
# 9) Evaluate
# -------------------------
def evaluate_model(model, X_eval, y_eval, name, scaled=False):
    preds = model.predict(X_eval)
    rmse = np.sqrt(mean_squared_error(y_eval, preds))
    r2 = r2_score(y_eval, preds)
    print(f"{name}: RMSE={rmse:.2f}, R2={r2:.3f}")
    return preds

print("Evaluating on test set:")
gbr_preds = evaluate_model(gbr, X_test, y_test, "GradientBoosting")
xgb_preds = evaluate_model(xgb, X_test, y_test, "XGBoost")
nn_preds = evaluate_model(nn, X_test_scaled, y_test, "NeuralNet (scaled)")

# -------------------------
# 10) Feature importance + savewe
# -------------------------
feature_names = X.columns.tolist()

# GBR
gbr_imps = gbr.feature_importances_
gbr_imp_df = pd.DataFrame({"feature": feature_names, "importance": gbr_imps}).sort_values("importance", ascending=False)
plt.figure(figsize=(10,6))
plt.bar(gbr_imp_df['feature'], gbr_imp_df['importance'])
plt.xticks(rotation=90)
plt.title("GBR Feature Importances")
plt.tight_layout()
plt.savefig("gbr_feature_importance.png")
plt.close()

# XGB
xgb_imps = xgb.feature_importances_
xgb_imp_df = pd.DataFrame({"feature": feature_names, "importance": xgb_imps}).sort_values("importance", ascending=False)
plt.figure(figsize=(10,6))
plt.bar(xgb_imp_df['feature'], xgb_imp_df['importance'])
plt.xticks(rotation=90)
plt.title("XGB Feature Importances")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png")
plt.close()

# Save predictions
preds_df = pd.DataFrame({
    "index": X_test.index,
    "actual_daily_limit": y_test.values,
    "gbr_pred": gbr_preds,
    "xgb_pred": xgb_preds,
    "nn_pred": nn_preds
})
preds_df.to_csv("model_predictions.csv", index=False)

print("\nSaved files:")
print(" - synthetic_customers.csv")
print(" - synthetic_transactions.csv")
print(" - synthetic_customers_with_features_and_limits.csv")
print(" - gbr_feature_importance.png")
print(" - xgb_feature_importance.png")
print(" - model_predictions.csv")

# Show top features textually
print("\nTop GBR features:")
print(gbr_imp_df.head(10).to_string(index=False))
print("\nTop XGB features:")
print(xgb_imp_df.head(10).to_string(index=False))

# Show small sample
print("\nSample of final dataset:")
print(full.sample(5, random_state=SEED).head().to_string(index=False))
