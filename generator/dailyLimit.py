import pandas as pd
import numpy as np
def generate_daily_limits(customer_df):
    limits = []

    for _, row in customer_df.iterrows():
        income = row["annual_income"]
        credit = row["credit_score"]
        on_time_ratio = row["on_time_payment_ratio"]
        late_payments = row["late_payment_count"]

        # Base = ~10% of monthly income
        base_limit = income / 12 * 0.1

        # Credit score adjustment
        if credit > 750:
            credit_factor = 1.3
        elif credit > 650:
            credit_factor = 1.0
        else:
            credit_factor = 0.7

        # Payment history adjustment
        payment_factor = 0.5 + on_time_ratio  # 0.5–1.5 scaling

        # Late payments penalty
        late_factor = max(0.7, 1 - (late_payments * 0.05))  # at most reduce 30%

        # Spending profile adjustment
        profile = np.random.choice(["high", "moderate", "low"], p=[0.2, 0.6, 0.2])
        if profile == "high":
            profile_factor = 1.3
        elif profile == "low":
            profile_factor = 0.8
        else:
            profile_factor = 1.0

        # Final limit
        limit = base_limit * credit_factor * payment_factor * late_factor * profile_factor

        # Enforce realistic bounds
        limit = min(max(limit, 1000), 50000)

        limits.append((row["customer_id"], profile, round(limit, 2)))

    limit_df = pd.DataFrame(limits, columns=["customer_id", "spending_profile", "daily_limit"])
    return limit_df

# Example usage
customer_data = pd.read_csv("./dataset/customer_profile_v1.csv")
daily_limits = generate_daily_limits(customer_data)
customer_with_limits = customer_data.merge(daily_limits, on="customer_id", how="left")
print(daily_limits.head(10))
customer_with_limits.to_csv("./dataset/customer_profiles_with_limits.csv", index=False)