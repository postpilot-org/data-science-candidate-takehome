import numpy as np
import pandas as pd
from scipy.special import expit 
import hashlib

def generate_data(random_seed=42, n_customers=15000):

    np.random.seed(random_seed)
    treatment_group = np.random.choice([1, 0], size=n_customers, p=[5/6, 1/6])
    campaigns = {
        'RETENTION_SUMMER': '2024-07-01',
        'RETENTION_FALL': '2024-10-01',
        'RETENTION_WINTER': '2025-01-01',
        'RETENTION_SUMMER_TEST': '2025-09-01',
        'RETENTION_FALL_TEST': '2025-11-01',
        'RETENTION_WINTER_TEST': '2026-12-01'
    }
    campaign_ids = np.random.choice(list(campaigns.keys()), size=n_customers, p=[0.35, 0.25, 0.2, 0.1, 0.05, 0.05])
    campaign_dates = pd.to_datetime([campaigns[cid] for cid in campaign_ids])
    test_group = np.array(['TEST' in id for id in campaign_ids])

    historical_lifetime_spend = np.random.gamma(2.0, 500, n_customers)
    brand_affinity_score = np.clip(np.random.beta(2, 5, n_customers), 0, 1)
    web_activity_score = np.random.normal(50, 4, n_customers)
    days_since_last_purchase = np.random.exponential(90, n_customers)
    price_sensitivity = np.random.beta(5, 2, n_customers)
    email_open_rate = np.random.uniform(0, 1, n_customers)  
    customer_tenure_days = np.random.randint(30, 2000, n_customers) 

    # Seasonal effect on spend
    seasonality_effect = np.array([0.1 if 'SUMMER' in c else 0.2 if 'FALL' in c else 0.4 for c in campaign_ids])
    hashes = [hashlib.md5(row.tobytes()).hexdigest() for row in campaign_ids]

    # Hidden transformations and interactions used only for generating the response (not provided to candidates)
    # Generate baseline spend (log-scale)
    log_spend = (
        5  # Increased baseline
        + seasonality_effect
        + 0.7 * np.log1p(historical_lifetime_spend)
        + 1.5 * brand_affinity_score
        + 0.04 * web_activity_score
        - 2.5 * price_sensitivity
        + 6 / (days_since_last_purchase + 1)
        + 0.8 * treatment_group
        + np.random.normal(0, 0.75, n_customers)
    )

    raw_spend = np.exp(log_spend)

    # Apply scaling to realistically cap spend near ~50,000
    max_desired_spend = 50000
    scaled_spend = raw_spend / np.percentile(raw_spend, 99.5) * max_desired_spend
    scaled_spend = np.clip(scaled_spend, 0, max_desired_spend)
    scaled_spend[scaled_spend == max_desired_spend] = max_desired_spend + np.random.normal(0, 4, sum(scaled_spend == max_desired_spend))

    # Simulate zero-spend indicator from Bernoulli
    logit_zero_prob = (
        -1.5
        + 2.0 * price_sensitivity
        - 1.2 * brand_affinity_score
        - 0.7 * treatment_group
    )
    zero_spend_prob = expit(logit_zero_prob)
    zero_spend_indicator = np.random.binomial(1, zero_spend_prob)

    final_spend = scaled_spend * (1 - zero_spend_indicator)

    df = pd.DataFrame({
        'customer_id': [ hashlib.md5(f'CUST_{i:05d}'.encode()).hexdigest() for i in range(n_customers)],
        'campaign_id': hashes,
        'campaign_send_date': campaign_dates,
        'treatment_group': treatment_group,
        'historical_lifetime_spend': historical_lifetime_spend,
        'brand_affinity_score': brand_affinity_score,
        'web_activity_score': web_activity_score,
        'days_since_last_purchase': days_since_last_purchase,
        'price_sensitivity': price_sensitivity,
        'email_open_rate': email_open_rate,
        'customer_tenure_days': customer_tenure_days,
        'total_30d_post_campaign_spend': final_spend
    })

    train_df = df[test_group == False]
    test_df = df[test_group == True]
    return train_df, test_df

if __name__ == '__main__':
    train_df, test_df = generate_data()
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    