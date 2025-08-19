# PostPilot iROAS Modeling — Submission Summary

> **Goal:** Estimate **incremental return on ad spend (iROAS)** for each campaign and serve predictions via an API.


## Table of Contents
- [1) Business Framing To Modeling Target](#1-business-framing-to-modeling-target)
- [2) Data & EDA (summary)](#2-data--eda-summary)
- [3) Methodology](#3-methodology)
- [4) Feature Engineering](#4-feature-engineering)
- [5) Evaluation & Diagnostics](#5-evaluation--diagnostics)
- [6) Serving API](#6-serving-api)
  
---

## 1) Business Framing To Modeling Target

**Goal:** For each historic campaign, estimate **iROAS** (incremental Return on Ad Spend):

$$\text{iROAS} = \frac{\sum_{i \in T} Y_i - \sum_{j \in C} Y_j}{0.70 \times |T|}$$

where $Y$ is 30-day post-campaign spend, $T$ and $C$ are the treated and control groups respectively, and $0.70 is the per-postcard cost.

### Modeling Approach: From Direct Prediction to Causal Estimation

The straightforward approach to calculating iROAS is to predict individual 30-day spend and compute the difference between treated and control group totals. This constitutes an **S-learner** in causal inference terminology—a single model trained on all data that relies on treatment as a feature to capture differential effects.

However, reframing this as a **causal uplift problem** allows us to systematically address selection bias and directly target the quantity of interest: individual treatment effects. I explored two meta-learning approaches:
- **X-learner**: Explicitly models treatment and control outcomes separately before estimating uplift
- **Doubly-Robust X-learner (DR-X)**: Augments X-learner with propensity weighting for improved bias correction

While DR-X improved upon standard X-learner, both methods **consistently underestimated** iROAS. This systematic underestimation makes sense given the heavy-tailed spend distribution—models trained to minimize average error naturally produce conservative predictions for rare high-spenders, yet these "whales" disproportionately impact campaign-level totals.

To address this bias, I developed a **blended ensemble** combining:
- The uplift models (X-learner, DR-X) which underestimate but provide robust causal estimates
- The direct outcome model which overestimates but better captures the upper tail of the distribution

While this ensemble approach sacrifices some interpretability, it leverages the complementary biases of each model class to produce campaign-level iROAS estimates that more closely track ground truth. The blend weights were tuned to minimize prediction error on validation campaigns, effectively learning to correct for the systematic biases observed in each component model.

**Important note:** Since campaigns are executed sequentially without temporal overlap (each campaign has a distinct 'campaign_send_date' with a 30-day measurement windows), there is no interference between campaigns, This means the campaign-level iROAS estimates is essentially the overall performance metrics. We don't need to account for cross-campaign spillover effects or adjustment for simulataneous treatments.

---

## 2) Data & EDA (summary)

- **Data quality:** No missing values or duplicates in either train or test sets
- **Covariate balance:** Treatment and control groups are well-balanced across features (SMD < 0.1 for 13/14 feature-split combinations). Only `customer_tenure_days` shows minor imbalance in test set (SMD = -0.11)
- **Target distribution:** `total_30d_post_campaign_spend` exhibits heavy-tailed behavior with many low/zero values and a few extreme high-spenders ("whales"). This right-skewed distribution poses modeling challenges:
  - Models minimizing squared error will likely underpredict the tail
  - Campaign-level aggregations are sensitive to these rare high-spenders
  - Log transformation considered to stabilize variance
- **Feature distributions:** Numeric features show consistent distributions across splits with no concerning skewness. Minor distributional shifts in `customer_tenure_days` and `email_open_rate` in test set are within acceptable ranges
- **Implications for modeling:** The well-balanced treatment assignment supports causal inference, but the heavy-tailed outcome requires careful handling (log transformation, robust loss functions, or ensemble methods)

Full EDA: `notebooks/02_eda.ipynb`.

---

## 3) Methodology

### Model Architecture
All models use **log1p(spend)** transformation with **Duan smearing estimator** for retransformation to dollar scale, addressing the heavy-tailed distribution identified in EDA.

#### Primary: Doubly-Robust X-Learner (DR-X)
- **Stage 1:** Separate CatBoost regressors for treated/control groups on log1p(y)
- **Stage 2:** Pseudo-outcome construction: $D_1^i = Y_i - \hat{m}_0(X_i)$ for treated, $D_0^j = \hat{m}_1(X_j) - Y_j$ for control
- **Stage 3:** CatBoost models on pseudo-outcomes to estimate $\tau_1(x)$ and $\tau_0(x)$
- **Stage 4:** Propensity-weighted blending: $\hat{\tau}(x) = g(x)\tau_0(x) + (1-g(x))\tau_1(x)$
- **Robustness:** Training-time outcome capping + prediction-time winsorization (99th percentile within campaign)

#### Secondary: Standard X-Learner
- Identical architecture to DR-X but using uniform weights (g=0.5) instead of propensity scores
- Provides ensemble diversity through different bias-variance tradeoff

#### Tertiary: S-Learner (Direct Prediction)
- Single CatBoost on pooled data with strong regularization
- Counterfactuals via treatment toggle: $\hat{\tau}_i = \hat{Y}_i(T=1) - \hat{Y}_i(T=0)$
- Excluded campaign_id to ensure generalization to new campaigns

> **Leakage prevention:** `campaign_id` excluded from all feature sets; used only for campaign-level aggregation post-prediction

### Ensemble Rationale
The three models exhibit complementary biases:
- **DR-X/X-Learner:** Conservative estimates, robust to selection bias
- **S-Learner:** Optimistic estimates, better tail capture

Optimal blend weights (determined via campaign-level MAE minimization on validation set):
- Achieves better calibration than any individual model
- Reduces systematic under/over-estimation observed in component models

### Uncertainty Quantification
- **Method:** Stratified bootstrap (1000 iterations, sampling within campaigns)
- **Output:** 95% confidence intervals for campaign iROAS via percentile method
- **Computation:** $\text{iROAS}_b = \sum_{i \in T_b} \hat{\tau}_i^b / (0.70 \times |T_b|)$ for each bootstrap sample $b$


## Future Improvements

With additional time, the following enhancements would strengthen the solution:

1. Cross-validation framework: K-fold CV stratified by campaign for more robust performance estimates
2. Hyperparameter optimization: Bayesian optimization for CatBoost parameters (depth, learning_rate, regularization)
3. Feature engineering: Interaction terms, spend velocity features, customer segmentation
4. Alternative algorithms: Neural uplift models (DragonNet), causal forests for comparison
5. Calibration analysis: Reliability plots for uplift predictions at different spend levels


---

## 4) Feature Engineering

Temporal features (derived from campaign_send_date):
- Day of week, month, and quarter to capture seasonal patterns

Log transformations for skewed distributions:
- log_hist_spend = log1p(historical_lifetime_spend) - stabilizes heavy-tailed spend
- log_tenure = log1p(customer_tenure_days) - normalizes tenure distribution
- inv_days_since = 1/(days_since_last_purchase + 1) - emphasizes recency with decay

Composite engagement metrics:
- engagement_score = 0.5 × web_activity_score + 0.5 × email_open_rate (linear combination)
- engagement_mult = web_activity_score × email_open_rate (interaction term)

Price sensitivity features:
- price_sensitivity_sq = price_sensitivity² (non-linear response)
- affinity_over_price = brand_affinity_score / (price_sensitivity + 1e-3) (ratio capturing value perception)

Cross-feature interactions:
- affinity_x_recency = brand_affinity_score × inv_days_since (combines loyalty with recency)

Discretization:
- recency_bin = categorical bins for days_since_last_purchase using fixed quantile boundaries

Note: All transformations are deterministic and consistently applied across training, validation, and serving to prevent data leakage.

---

## 5) Evaluation & Diagnostics

**Primary metric (business-aligned):** per-campaign **iROAS absolute error**.  
We also report **winsorized MAE/RMSE** (e.g., 99.5th percentile) to de-emphasize whales.

**Diagnostics:**
- Scatter plots: `iROAS_actual` vs `iROAS_pred_*` with y=x.
- Whale-aware plots (highlight campaigns with extreme spenders).
- Residual checks by seasonality & features.

**Observed behavior:**  
DR-X is conservative; Plain overshoots; **Blend** balances bias–variance and tracks closely with ground truth.

---

## 6) Serving API

A lightweight **FastAPI** app loads a single artifact — a small class `BlendedTauDF` that encapsulates the three base models, their weights, and the full feature pipeline.

### Run the API

```bash
# (optional) set a token to require Authorization: Bearer <token>
export API_TOKEN="mytoken"

# start API
uvicorn app:app --reload
# http://127.0.0.1:8000/health

# inference
curl -X POST http://127.0.0.1:8000/predict_tau \
  -H "Content-Type: application/json" \
  -d '{
    "records":[
      {"campaign_id":"A","treatment_group":1,"campaign_send_date":"2025-05-10",
       "historical_lifetime_spend":1200,"days_since_last_purchase":30,
       "brand_affinity_score":0.5,"web_activity_score":0.2,"price_sensitivity":0.3,
       "email_open_rate":0.15,"customer_tenure_days":400},
      {"campaign_id":"A","treatment_group":0,"campaign_send_date":"2025-05-10",
       "historical_lifetime_spend":100,"days_since_last_purchase":120,
       "brand_affinity_score":0.4,"web_activity_score":0.1,"price_sensitivity":0.6,
       "email_open_rate":0.05,"customer_tenure_days":60}
    ],
    "winsor_low": 0.01,
    "winsor_high": 0.99
  }'
```
