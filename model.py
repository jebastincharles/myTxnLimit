import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

transactions = pd.read_csv("./dataset/transactions.csv", parse_dates=["transaction_date"])
customer_with_limits = pd.read_csv("./dataset/customer_profiles_with_limits.csv")
# ---------------------------
# 1. Transaction Feature Engineering
# ---------------------------
# Aggregate features from transaction dataset
transactions['amount'] = pd.to_numeric(transactions['amount'], errors='coerce')
#transactions['is_fraud'] = transactions['is_fraud'].astype(int) # Assumes it's boolean or can be converted to 0/1
transactions['is_fraud'] = pd.to_numeric(transactions['is_fraud'], errors='coerce').astype('Int64')

txn_features = transactions.groupby("customer_id").agg({
    "amount": ["mean", "std", "max", "sum"],
    "transaction_id": "count",
    "is_fraud": "mean"
})

print(txn_features.head(10))

# Flatten multi-level column names
txn_features.columns = ["avg_amount", "std_amount", "max_amount",
                        "total_spent", "txn_count", "fraud_ratio"]
txn_features.reset_index(inplace=True)

# ---------------------------
# 2. Merge with Customer Profiles + Daily Limit
# ---------------------------
full_data = customer_with_limits.merge(txn_features, on="customer_id", how="left")
full_data.to_csv("./dataset/full_customer_data.csv", index=False)

# Fill missing values (if some customers have no transactions)
full_data.fillna(0, inplace=True)

# ---------------------------
# 3. Prepare Features & Target
# ---------------------------
X = full_data.drop(columns=["customer_id", "daily_limit"])
y = full_data["daily_limit"]

# One-hot encode categorical spending_profile
X = pd.get_dummies(X, columns=["spending_profile"], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 4. Train Models
# ---------------------------

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gbr.fit(X_train, y_train)

# XGBoost Regressor
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)

# Neural Network Regressor (MLP)
nn = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                  solver='adam', max_iter=500, random_state=42)
nn.fit(X_train, y_train)

# ---------------------------
# 5. Evaluation Function
# ---------------------------
def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name}: RMSE={rmse:.2f}, R²={r2:.3f}")
    return rmse, r2

print("\n--- Model Performance ---")
evaluate(gbr, X_test, y_test, "Gradient Boosting")
evaluate(xgb, X_test, y_test, "XGBoost")
evaluate(nn, X_test, y_test, "Neural Net")