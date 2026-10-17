# Churn Prediction — Healthcare Member Retention

Predictive modeling pipeline to identify healthcare members at risk of disenrollment, enabling proactive retention interventions.

## Business Problem

Healthcare payers lose significant revenue from member churn. Early identification of at-risk members allows targeted outreach before disenrollment, improving retention rates and lifetime value. This project builds a binary classification model to predict churn risk.

## Approach

1. **Synthetic Data Generation** — 5,000 members with utilization, engagement, cost, satisfaction, and churn labels (15% churn rate)
2. **Exploratory Data Analysis** — Feature distributions, churn correlations, class imbalance assessment
3. **Feature Engineering** — Derived features for engagement ratios, cost burden, and utilization patterns
4. **Model Training** — Logistic Regression, Random Forest, and Gradient Boosting with cross-validation
5. **Evaluation** — ROC-AUC, precision-recall, confusion matrix, feature importance
6. **Risk Scoring** — Probability-based risk tiers for operational use

## Model Performance

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| Logistic Regression | ~0.78 | ~0.45 | ~0.65 | ~0.53 |
| Random Forest | ~0.82 | ~0.50 | ~0.60 | ~0.55 |
| Gradient Boosting | ~0.84 | ~0.52 | ~0.68 | ~0.59 |

## Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **Models** — LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
- **Visualization** — matplotlib, seaborn
- **Notebook** — Jupyter for interactive analysis

## Project Structure

```
├── README.md
├── requirements.txt
├── data/
│   └── member_churn_data.csv
├── src/
│   ├── generate_data.py
│   ├── features.py
│   └── model.py
└── notebooks/
    └── churn_prediction.ipynb
```

## How to Run

```bash
pip install -r requirements.txt
python src/generate_data.py
jupyter notebook notebooks/churn_prediction.ipynb
```

## Skills Demonstrated

- Supervised classification for business outcomes
- Handling imbalanced datasets
- Feature engineering from healthcare operational data
- Model comparison and selection
- Translating model output into actionable risk tiers
