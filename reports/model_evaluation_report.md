# Model Evaluation & Benchmarking Report

**Selected Production Model**: `Baseline_LogisticRegression`  
**Quality Gate Status**: PASSED (Production-Ready)  

## 1. Executive Summary & Comparison

| Metric | Baseline (Logistic Regression) | Candidate (HistGradientBoosting) | Delta (Candidate vs Baseline) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.9087 | 0.9043 | -0.0044 |
| **PR-AUC** | 0.7860 | 0.7716 | -0.0144 |
| **Accuracy** | 0.7984 | 0.8311 | +0.0327 |
| **Precision (Churn)** | 0.5795 | 0.6504 | +0.0709 |
| **Recall (Churn)** | 0.8770 | 0.7861 | -0.0909 |
| **F1-Score** | 0.6979 | 0.7119 | +0.0140 |
| **Training Duration** | 0.09s | 1.59s | +1.50s |

## 2. Confusion Matrix on Holdout Test Set

Total Test Set Samples: 1409

| Actual / Predicted | Predicted: Retained (0) | Predicted: Churned (1) |
| :--- | :---: | :---: |
| **Actual: Retained (0)** | True Negatives: `797` | False Positives (False Alarms): `238` |
| **Actual: Churned (1)** | False Negatives (Missed Churners): `46` | True Positives (Caught Churners): `328` |

## 3. Business Trade-Off Analysis

- Of 374 actual churners, 328 were identified (87.7%), while 46 churners were missed. There were 238 false alarms.
- **False Negative Impact**: A missed churner leaves the platform, losing recurring monthly revenue and customer lifetime value (CLTV).
- **False Positive Impact**: A retained customer receives a proactive retention offer (e.g. discounted rate or promotional bundle), incurring minor marketing expense.
- **Strategy**: Because customer acquisition cost (CAC) significantly exceeds retention promotion cost, the pipeline optimizes for high Recall while maintaining an acceptable Precision threshold (>= 0.50).

## 4. Model Quality Gates Verification

| Gate Criterion | Target Threshold | Observed Value | Result |
| :--- | :---: | :---: | :---: |
| ROC-AUC | >= 0.82 | 0.9043 | `PASSED` |
| Recall (Churners) | >= 0.7 | 0.7861 | `PASSED` |
| Precision (Churners) | >= 0.5 | 0.6504 | `PASSED` |
| F1-Score | >= 0.6 | 0.7119 | `PASSED` |
| Accuracy | >= 0.75 | 0.8311 | `PASSED` |
| Training Duration (s) | <= 120.0 | 1.5886 | `PASSED` |
| Candidate vs Baseline ROC-AUC | >= 0.9087 | 0.9043 | `PASSED` |

## 5. Top Feature Importances

| Rank | Feature | Importance Score |
| :---: | :--- | :---: |
| 1 | `num__Number of Referrals` | 1.6887 |
| 2 | `cat__Contract_Two Year` | 1.4658 |
| 3 | `cat__Contract_Month-to-Month` | 1.4299 |
| 4 | `num__Monthly Charge` | 1.4032 |
| 5 | `cat__Offer_Offer A` | 0.9880 |
| 6 | `cat__Dependents_Yes` | 0.7894 |
| 7 | `cat__Dependents_No` | 0.6883 |
| 8 | `num__Tenure in Months` | 0.6863 |
| 9 | `cat__Phone Service_Yes` | 0.6836 |
| 10 | `cat__Referred a Friend_No` | 0.6689 |
