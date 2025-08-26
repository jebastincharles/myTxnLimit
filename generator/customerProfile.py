
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of synthetic customers
n_customers = 1000

# Generate synthetic customer data
customer_data = pd.DataFrame({
    "customer_id": [f"CUST{i:04d}" for i in range(1, n_customers+1)],

    # Years with the bank
    "account_years": np.random.randint(1, 20, n_customers),

    # Annual income distribution (centered ~60k, spread 20k, clipped to reasonable bounds)
    "annual_income": np.random.normal(60000, 20000, n_customers).clip(20000, 200000),

    # Percentage income change from last year (-50% to +50%)
    "income_change_pct": np.random.normal(0, 0.1, n_customers).clip(-0.5, 0.5),

    # Payment history: ratio of on-time payments
    "on_time_payment_ratio": np.random.beta(9, 1.5, n_customers),

    # Count of late payments
    "late_payment_count": np.random.poisson(2, n_customers),

    # Credit score distribution (FICO-like range 300–850, centered ~680)
    "credit_score": np.random.normal(680, 50, n_customers).clip(300, 850)
})

# Round appropriate columns
customer_data["annual_income"] = customer_data["annual_income"].round(2)
customer_data["income_change_pct"] = (customer_data["income_change_pct"] * 100).round(2)  # in %
customer_data["on_time_payment_ratio"] = customer_data["on_time_payment_ratio"].round(2)
customer_data["credit_score"] = customer_data["credit_score"].round().astype(int)

# Preview first 10 rows
print(customer_data.head(10))

# Save to CSV (optional)
customer_data.to_csv("./dataset/customer_profile_v1.csv", index=False)