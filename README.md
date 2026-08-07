# Forest Fire Risk Modeling: Regression, Regularization & Classification

A complete CRISP-DM machine learning project analyzing environmental and geographic
drivers of wildfire severity, and comparing regression vs. classification approaches
for operational fire-risk decision-making.

**Score: 100/100** — Flatiron School AI Engineering, Summative Lab (Regression)

[**View the rendered report →**](https://your-project.vercel.app) &nbsp;|&nbsp; [Notebook](./C06M08Lab.ipynb)

---

## Business Context

As a junior data scientist supporting a forestry management team, the objective was to
help emergency response and resource-allocation decisions by answering two questions:

1. **What conditions drive wildfire size and severity?** (temperature, humidity, wind,
   and other environmental/geographic factors)
2. **Can we reliably predict which fires will become high-risk**, so resources can be
   deployed before a fire escalates rather than after?

## Approach

The project follows the CRISP-DM framework end-to-end:

- **EDA** — distribution of burned area, correlation analysis, and scatterplots across
  key predictors (temperature, wind, humidity) to surface relationships before modeling.
- **Regression modeling** — baseline multiple linear regression, extended with a
  quadratic temperature term and a temperature × wind interaction term, compared on
  adjusted R², AIC, and BIC.
- **Model diagnostics** — residuals-vs-fitted and Q-Q plots to check regression
  assumptions, plus Cook's Distance to flag influential observations.
- **Regularization** — Ridge and Lasso regression to address multicollinearity and
  compare generalization vs. interpretability trade-offs.
- **Classification** — the continuous target was converted into a binary "high-risk
  fire" label (above/below median burned area) and evaluated with a logistic regression
  model using accuracy, precision, recall, F1, and a confusion matrix.
- **Assumption checks** — Variance Inflation Factor (VIF) analysis to confirm low
  multicollinearity among the standardized predictors.

## Key Findings

- Ridge regression offered the best bias/variance trade-off among the regression
  models, achieving the lowest MSE (≈0.70) while retaining all predictors.
- Lasso produced comparable performance while zeroing out several coefficients —
  useful when a simpler, more interpretable model is the priority.
- The logistic regression classifier reached **~77% accuracy** on the binary high-risk
  task, with low multicollinearity (VIF ≈ 1 across features), and its coefficients
  offer a directly interpretable, actionable signal for resource-allocation decisions.
- **Recommendation:** use the classifier for operational triage (should this fire get
  immediate resources?) and the regression model for research/planning contexts that
  need a continuous estimate of expected burned area.

## Tools & Techniques

`Python` · `pandas` · `NumPy` · `statsmodels` (OLS regression, diagnostics) ·
`scikit-learn` (Ridge, Lasso, Logistic Regression, train/test split, scaling) ·
`matplotlib` / `seaborn` (EDA and diagnostic visualizations) · `ucimlrepo` (dataset)

## Dataset

[UCI Machine Learning Repository — Forest Fires Data Set](https://archive.ics.uci.edu/dataset/162/forest+fires)
(fetched programmatically via `ucimlrepo`).

## Repo Contents

```
├── C06M08Lab.ipynb   # Full notebook: EDA → regression → regularization → classification
└── README.md
```

---

*Built as part of the AI Engineering apprenticeship at [Flatiron School](https://flatironschool.com).*
