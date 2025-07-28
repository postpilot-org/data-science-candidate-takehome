import numpy as np
import pandas as pd

def generate_data(random_seed=42, n_customers=15000):

    campaigns = {
        'RETENTION_SUMMER': '2025-07-01',
        'RETENTION_FALL': '2025-10-01',
        'RETENTION_WINTER': '2026-01-01'
    }

    campaign_ids = np.random.choice(list(campaigns.keys()), size=n_customers, p=[0.4, 0.35, 0.25])
    campaign_dates = pd.to_datetime([campaigns[cid] for cid in campaign_ids])

    # Treatment group assignment (imbalanced)
    treatment_group = np.random.choice([1, 0], size=n_customers, p=[5/6, 1/6])

    # Customer attributes (untransformed)
    historical_lifetime_spend = np.random.gamma(2.0, 500, n_customers)
    brand_affinity_score = np.clip(np.random.beta(2, 5, n_customers), 0, 1)
    web_activity_score = np.random.normal(50, 15, n_customers)
    days_since_last_purchase = np.random.exponential(90, n_customers)
    price_sensitivity = np.random.beta(5, 2, n_customers)  # negatively correlated predictor
    email_open_rate = np.random.uniform(0, 1, n_customers)  # noise
    customer_tenure_days = np.random.randint(30, 2000, n_customers)  # noise

    # Seasonal effect on spend
    seasonality_effect = np.array([0.1 if 'SUMMER' in c else 0.2 if 'FALL' in c else 0.4 for c in campaign_ids])

    # Hidden transformations and interactions used only for generating the response (not provided to candidates)
    log_spend = (
        3
        + seasonality_effect
        + 0.5 * np.log1p(historical_lifetime_spend)
        + 1.2 * brand_affinity_score
        + 0.03 * web_activity_score
        - 2.0 * price_sensitivity
        + 0.00001 * historical_lifetime_spend ** 2
        - 0.000001 * web_activity_score ** 3
        + 5 / (days_since_last_purchase + 1)
        + 0.02 * (brand_affinity_score * web_activity_score)
        - 0.0005 * (historical_lifetime_spend * price_sensitivity)
        + 0.6 * treatment_group
        + np.random.normal(0, 1, n_customers)
    )

    # Convert log-spend back to original scale
    total_30d_post_campaign_spend = np.expm1(log_spend)

    # Final dataset (omitting transformed/interactions)
    df = pd.DataFrame({
        'customer_id': [f'CUST_{i:05d}' for i in range(n_customers)],
        'campaign_id': campaign_ids,
        'campaign_send_date': campaign_dates,
        'treatment_group': treatment_group,
        'historical_lifetime_spend': historical_lifetime_spend,
        'brand_affinity_score': brand_affinity_score,
        'web_activity_score': web_activity_score,
        'days_since_last_purchase': days_since_last_purchase,
        'price_sensitivity': price_sensitivity,
        'email_open_rate': email_open_rate,
        'customer_tenure_days': customer_tenure_days,
        'total_30d_post_campaign_spend': total_30d_post_campaign_spend
    })

    return df

if __name__ == '__main__':
    df = generate_data()
    df.to_csv('data.csv', index=False)
    