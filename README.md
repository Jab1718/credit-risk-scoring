# Credit Risk Scoring

A web application for credit risk assessment using a **LightGBM** model, with integrated **SHAP (SHapley Additive exPlanations)** to provide transparent explanations for loan approval and rejection decisions.

---

## Features

- **Single prediction** — manually enter customer information and receive an instant result
- **Batch prediction** — upload a CSV file to process multiple applications at once
- **Automated credit terms** — calculates interest rate, credit limit, and counter-offer based on predicted probability of default (EV-based pricing)
- **SHAP explanations** — displays the top risk factors and strengths for each application
- **Adjustable rejection threshold** — slider to tune the PD threshold according to risk appetite

---

## SHAP Explanation

After each prediction, the system displays:

| | Description |
|---|---|
| **Risk factors** | Top 5 features pushing the probability of default upward |
| **Strengths** | Top 3 features pulling the probability of default downward |
| **Base rate** | Average probability of default in the training set (reference point) |

A positive SHAP value means the feature **increases** risk. A negative SHAP value means the feature **reduces** risk. Bar length represents relative impact compared to the most influential feature.

---

## Project Structure

```
├── app.py                  # Flask backend + SHAP logic
├── models/
│   └── lightGBM_model.pkl  # Trained model (sklearn Pipeline)
└── templates/
    ├── index.html          # Input interface
    └── result.html         # Results page with SHAP visualization
```

---

## Installation & Usage

**1. Install dependencies**

```bash
pip install flask pandas scikit-learn lightgbm shap
```

**2. Run the application**

```bash
python app.py
```

**3. Open in browser**

```
http://localhost:5000
```

---

## Data

The training and test datasets are sourced from the **[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)** competition on Kaggle.

**20 input features:**

| Group | Features |
|---|---|
| Credit scores | `EXT_SOURCE_1/2/3`, `EXT_SOURCE_MEAN` |
| Personal information | `AGE`, `YEARS_EMPLOYED`, `EMPLOYMENT_STABILITY`, `INCOME_PER_PERSON` |
| Administrative history | `CONVERTED_DAYS_ID_PUBLISH`, `CONVERTED_DAYS_LAST_PHONE_CHANGE`, `CONVERTED_DAYS_REGISTRATION`, `REGION_POPULATION_RELATIVE` |
| Loan details | `LOG_AMT_CREDIT`, `LOG_AMT_ANNUITY`, `LOG_AMT_GOODS_PRICE`, `ANNUITY LENGTH` |
| Financial ratios | `CREDIT_TO_GOODS_RATIO`, `ANNUITY_BY_INCOME`, `CREDIT_BY_INCOME`, `GOODS_PRICE_BY_INCOME` |

---

## Risk Classification

| Tier | Probability of Default | Decision |
|---|---|---|
| Priority | < 20% | Approved — preferential rate |
| Standard | 20% – 35% | Approved — standard rate |
| Cautious | 35% – 43.75% | Conditionally approved / Counter-offer |
| Rejected | >= 43.75% | Rejected |

The default rejection threshold of `43.75%` is derived from `MAX_RATE / (MAX_RATE + LGD)` = `35% / (35% + 45%)`. It can be adjusted via the slider on the interface.
