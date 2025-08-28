import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score,root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate Customer Profiles
# -----------------------------
def generate_customer_profiles(n_customers=500):
    customers = pd.DataFrame({
        "customer_id": range(1, n_customers + 1),
        "account_tenure_years": np.random.randint(1, 20, n_customers),
        "credit_score": np.random.randint(300, 850, n_customers),
        "income": np.random.randint(20000, 200000, n_customers),
        "loan_default_history": np.random.binomial(1, 0.1, n_customers),
    })
    return customers


# -----------------------------
# 2. Generate Transactions
# -----------------------------
def generate_transactions(customers, max_txn_per_cust=200):
    records = []
    for _, row in customers.iterrows():
        n_txns = np.random.randint(50, max_txn_per_cust)
        start_date = datetime(2023, 1, 1)
        for i in range(n_txns):
            txn_date = start_date + timedelta(days=np.random.randint(0, 180))
            amount = np.random.exponential(scale=row["income"]/1000)  # spending ~ income
            records.append([
                row["customer_id"],
                txn_date,
                amount,
                random.choice(["POS", "Online", "ATM", "Travel", "Shopping"]),
                random.choice(["US", "IN", "SG", "UK"])
            ])
    txns = pd.DataFrame(records, columns=["customer_id", "date", "amount", "merchant_type", "country"])
    return txns


# -----------------------------
# 3. Time-Series Features
# -----------------------------
def add_time_series_features(transactions_df):
    transactions_df = transactions_df.sort_values(["customer_id", "date"])

    transactions_df["7d_avg_spend"] = (
        transactions_df.groupby("customer_id")["amount"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    transactions_df["7d_txn_count"] = (
        transactions_df.groupby("customer_id")["amount"]
        .transform(lambda x: x.rolling(7, min_periods=1).count())
    )

    transactions_df["spike_ratio"] = (
        transactions_df["amount"] / (transactions_df["7d_avg_spend"] + 1e-6)
    )

    transactions_df["days_since_last_txn"] = (
        transactions_df.groupby("customer_id")["date"]
        .diff()
        .dt.days
        .fillna(0)
    )

    return transactions_df


def aggregate_customer_features(transactions_df):
    agg_funcs = {
        "amount": ["mean", "max", "sum"],
        "7d_avg_spend": "mean",
        "7d_txn_count": "mean",
        "spike_ratio": "mean",
        "days_since_last_txn": "mean",
    }

    customer_features = (
        transactions_df.groupby("customer_id")
        .agg(agg_funcs)
    )

    customer_features.columns = ["_".join(col).strip() for col in customer_features.columns.values]
    customer_features = customer_features.reset_index()
    return customer_features


# -----------------------------
# 4. Generate Ground Truth Daily Limit
# -----------------------------
def generate_daily_limit(customers, features):
    df = customers.merge(features, on="customer_id", how="left")

    df["daily_limit"] = (
        0.3 * df["income"]
        + 50 * df["credit_score"]
        + 1000 * df["account_tenure_years"]
        - 20000 * df["loan_default_history"]
        + 2 * df["amount_mean"]
        - 500 * df["spike_ratio_mean"]
        - 1000 * df["days_since_last_txn_mean"]
    )

    # Add noise
    df["daily_limit"] = df["daily_limit"] + np.random.normal(0, 5000, len(df))
    df["daily_limit"] = df["daily_limit"].clip(lower=1000)

    return df


# -----------------------------
# 5. Train Models
# -----------------------------
def train_models(df):
    X = df.drop(columns=["customer_id", "daily_limit"])
    y = df["daily_limit"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "GradientBoosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=200),
        "NeuralNet": MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        #rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"rmse": rmse,"mae":mae, "r2": r2}
        print(f"{name}: RMSE={rmse:.2f},MAE={mae:.2f},  R2={r2:.3f}")

        # Feature importance for tree models
        if name in ["GradientBoosting", "XGBoost"]:
            importance = model.feature_importances_
            plt.figure(figsize=(8,5))
            plt.barh(X.columns, importance)
            plt.title(f"Feature Importance - {name}")
            plt.show()



# -----------------------------
# Run Full Pipeline
# -----------------------------
customers = generate_customer_profiles(300)
transactions = generate_transactions(customers, max_txn_per_cust=15000)
transactions = add_time_series_features(transactions)
features = aggregate_customer_features(transactions)
df = generate_daily_limit(customers, features)

results = train_models(df)