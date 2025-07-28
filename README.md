# Data Science Candidate Repo
PostPilot specializes in personalized direct-mail marketing campaigns to grow customer base and increase customer lifetime value. One of our key business goals is identifying consumers having the highest total spend in a specific promotional window following a campaign. Accurately modeling the anticipated spend of consumers allows us to optimize targeting, allocate marketing budgets efficiently, and maximize return on ad spend (ROAS).


## Objectives
Your objectives are to train a model to predict `total_30d_post_campaign_spend` and update the provided Flask API script to serve predictions made by your model. Implement your solution in Python, following professional coding practices (PEP8 compliance, modularity, clear function/class definitions, type hints). Provide either a script or Jupyter notebook that includes your technical analyses and findings. Your submission should include artifacts from the following:

1. Clearly define the mathematical/statistical framing of the business problem.

2. Conduct exploratory data analysis to uncover meaningful patterns, distributions, and relationships. Visualize key distributions (e.g., target variable distribution demonstrating its heavy-tailed nature), and identify critical modeling challenges (e.g., data skewness, zero-inflation, heteroscedasticity, etc.).

3. Methodology and Modeling Choices

Be prepared to clearly explain:
- the rationale behind your chosen modeling approach
- trade-offs between predictive accuracy and interpretability
- uncertainty quantification and robustness considerations

4. Feature Engineering & Selection
Clearly document your feature-engineering process, especially any new features derived from raw variables (e.g., recency-weighted lifetime spend, interaction terms, embedding representations of categorical variables). _Provide justifications for the chosen features_.

5. Model Evaluation & Diagnostics
Evaluate your model using appropriate metrics; conduct and visualize residual diagnostics to identify model strengths and weaknesses.

6. Software Engineering Practices
Organize your work within a Git repository using best practices (meaningful commits, branches, clear PR messages).


## Dataset

The CSV dataset (data.csv) consists of the following columns:

| Column Name                                | Data Type | Description                                                                    |
| ------------------------------------------ | --------- | ------------------------------------------------------------------------------ |
| `customer_id`                              | `string`  | Unique customer identifier                                                     |
| `campaign_id`                              | `string`  | Campaign identifier (`RETENTION_SUMMER`, `RETENTION_FALL`, `RETENTION_WINTER`) |
| `campaign_send_date`                       | `date`    | Date campaign started                                                          |
| `treatment_group`                          | `integer` | 1 if customer received campaign, 0 if control (imbalanced)                     |
| `historical_lifetime_spend`                | `float`   | Total past spend prior to campaign                                             |
| `brand_affinity_score`                     | `float`   | Brand affinity score (0–1)                                                     |
| `web_activity_score`                       | `float`   | Recent web browsing activity score                                             |
| `days_since_last_purchase`                 | `float`   | Days since last purchase                                                       |
| `price_sensitivity`                        | `float`   | Customer sensitivity to price                                                  |
| `email_open_rate`                 | `float`   | Email open rate (0–1)
| `customer_tenure_days`          | `integer` | Number of days since first purchase                        |
| `total_30d_post_campaign_spend` *(target)* | `float`   | Spend in 30 days post-campaign (response variable)                             |



## Evaluation
We will evaluate your work sample by using your prediction script to make predictions
against a test set that is hidden from you. Your code should be environment agnostic; 
it should be possible to run it on any machine with a recent version of Python installed.
Please be sure to give careful instructions explaining how to run your code
to make these predictions.  

You will be evaluated via the following criteria:
- Readability and cleanliness of your code
- Ease of running your code to make new predictions
- Quality of analysis
- Improvement over provided model
