import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_transactions_with_patterns(customer_df, transactions_per_customer=50,
                                        start_date="2024-01-01", end_date="2024-12-31"):
    np.random.seed(42)
    random.seed(42)

    merchants = ["Groceries", "Electronics", "Travel", "Dining", "Clothing", 
                 "Healthcare", "Fuel", "Entertainment"]
    locations = ["New York", "London", "Singapore", "Mumbai", "Sydney", "Dubai", "Berlin", "Toronto"]

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days_range = (end - start).days

    all_txns = []

    for _, row in customer_df.iterrows():
        cust_id = row["customer_id"]
        base_income = row["annual_income"]

        # Assign spending profile
        profile = np.random.choice(["high", "moderate", "low"], p=[0.2, 0.6, 0.2])

        # Adjust txn count and scale by profile
        if profile == "high":
            txn_count = int(transactions_per_customer * 1.5)
            scale_factor = base_income / 80
        elif profile == "low":
            txn_count = int(transactions_per_customer * 0.5)
            scale_factor = base_income / 300
        else:  # moderate
            txn_count = transactions_per_customer
            scale_factor = base_income / 150

        # Generate transactions
        for i in range(txn_count):
            txn_id = f"TXN_{cust_id}_{i+1}"

            txn_date = start + timedelta(days=np.random.randint(0, days_range),
                                         hours=np.random.randint(0, 24),
                                         minutes=np.random.randint(0, 60))

            # Base spend amount
            amount = np.random.exponential(scale=scale_factor)
            amount = round(min(amount, base_income/2), 2)

            merchant = random.choice(merchants)
            location = random.choice(locations)

            # Fraud probability slightly higher for spikes & geo velocity
            is_fraud = np.random.choice([0, 1], p=[0.985, 0.015])

            all_txns.append([txn_id, cust_id, txn_date, amount, merchant, location,
                             profile, is_fraud])

        # Inject behavior: spending spike (rare, 5% of customers)
        if np.random.rand() < 0.05:
            for j in range(3):
                txn_id = f"SPIKE_{cust_id}_{j+1}"
                txn_date = start + timedelta(days=np.random.randint(0, days_range))
                amount = round(np.random.uniform(base_income*0.3, base_income*0.6), 2)
                all_txns.append([txn_id, cust_id, txn_date, "Luxury Purchase", "LuxuryStore",
                                 profile, 1])  # often fraudulent

        # Inject geo-velocity anomaly (rare, 3% of customers)
        if np.random.rand() < 0.03:
            txn_time = start + timedelta(days=np.random.randint(0, days_range), hours=10)
            for k, loc in enumerate(["New York", "Singapore"]):  # far apart
                txn_id = f"GEO_{cust_id}_{k+1}"
                all_txns.append([txn_id, cust_id, txn_time + timedelta(hours=k*2),
                                 round(np.random.uniform(50, 500), 2),
                                 "Travel", loc, profile, 1])

    txn_df = pd.DataFrame(all_txns, columns=[
        "transaction_id", "customer_id", "transaction_date", "amount",
        "merchant_category", "location", "spending_profile", "is_fraud"
    ])

    return txn_df

# Example usage
customer_data = pd.read_csv("./dataset/customer_profile_v1.csv")
transactions = generate_transactions_with_patterns(customer_data, transactions_per_customer=1000)
print(transactions.head(15))

transactions.to_csv("./dataset/transactions.csv", index=False)