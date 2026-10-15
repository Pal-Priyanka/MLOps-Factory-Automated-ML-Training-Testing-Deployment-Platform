# Model Card — Customer Churn Risk Classifier

## 1. Model Details
- **Model Name**: Telco Customer Churn Classifier (`Baseline_LogisticRegression` / `Candidate_HistGradientBoosting`)
- **Version**: 1.0.0
- **Framework**: Scikit-Learn 1.9.1 / Python 3.12
- **Artifact Format**: Serialized Scikit-Learn Pipeline (`model.joblib`) with SHA-256 cryptographic checksum
- **Release Date**: October 2026
- **License**: MIT

---

## 2. Intended Use
- **Primary Use Case**: Early identification of telecommunication subscribers at elevated risk of terminating their service subscription within the upcoming billing cycle.
- **Target Beneficiaries**: Customer Success, Retention Marketing, and Operations teams for targeted customer outreach and personalized retention offers.
- **Out-of-Scope Uses**: 
  - Automated service termination or credit scoring.
  - Deploying to commercial domains with non-subscription pricing structures without domain adaptation.
  - Making punitive credit or contractual decisions based on model predictions.

---

## 3. Training & Evaluation Data
- **Dataset**: IBM Telco Customer Churn (Enriched 50-column dataset, 7,043 customer accounts).
- **Target Variable**: `Churn Label` (`Yes`: 1,869 / 26.54%, `No`: 5,174 / 73.46%).
- **Feature Set**: 36 features (14 numeric, 22 categorical) spanning customer tenure, service subscriptions, contract types, and billing summaries.
- **Leakage Prevention**: Excluded 14 identifier and post-event columns (`Customer ID`, `Satisfaction Score`, `Churn Reason`, `Churn Category`, `Churn Score`, etc.).
- **Data Splitting**: Stratified holdout split:
  - Training Set: 64% (4,507 samples)
  - Validation Set: 16% (1,127 samples)
  - Holdout Test Set: 20% (1,409 samples)

---

## 4. Empirical Evaluation Metrics (Holdout Test Set)

The following metrics were calculated on the unbiased 20% holdout test set (1,409 samples):

| Metric | Selected Model (`LogisticRegression`) | Candidate Model (`HistGradientBoosting`) |
| :--- | :---: | :---: |
| **ROC-AUC** | **0.9087** | 0.9043 |
| **PR-AUC** | **0.7860** | 0.7716 |
| **Accuracy** | 0.7984 | **0.8311** |
| **Recall (Churners)** | **0.8770** | 0.7861 |
| **Precision (Churners)** | 0.5795 | **0.6504** |
| **F1-Score** | 0.6979 | **0.7119** |
| **Training Latency** | **0.09s** | 1.59s |

### Confusion Matrix (Selected Model: 1,409 Test Samples)
- **True Negatives (Retained correctly predicted)**: 797
- **False Positives (False Alarms / Unnecessary outreach)**: 238
- **False Negatives (Missed Churners)**: 46
- **True Positives (Identified Churners)**: 328

---

## 5. Business Trade-Off & Decision Thresholds

### The Churn Retention Asymmetry:
- **Cost of False Negative ($C_{FN}$)**: A churning customer is missed entirely. The business loses all future recurring subscription revenue and incurs high customer acquisition costs (CAC) to replace them.
- **Cost of False Positive ($C_{FP}$)**: A loyal customer is flagged as high risk. The business offers a promotional bundle or loyalty discount. The marginal cost is minimal (often under $15-$25).
- **Decision Rule**: The platform defaults to balanced class weights, achieving **87.7% Recall** on actual churners. For businesses wishing to minimize promotion spend, probability thresholds can be adjusted from the default `0.50` up to `0.65`.

---

## 6. Top Predictive Factors

Feature importance extraction identifies the following primary drivers of subscriber churn:
1. **Number of Referrals**: Customers with zero or few referrals exhibit significantly higher churn propensity.
2. **Contract Type (`Month-to-Month` vs `Two Year`)**: Month-to-month contracts lack lock-in and exhibit highest churn hazard.
3. **Monthly Charge**: High monthly bills without multi-product discounts correlate with dissatisfaction.
4. **Tenure in Months**: New subscribers (< 6 months) face the highest hazard rate before stabilizing into brand loyalty.
5. **Fiber Optic Internet without Tech Support**: Higher-priced service combined with lack of technical support leads to competitor migration.

---

## 7. Ethical Considerations & Limitations
- **Demographic Parity**: Features include `Gender`, `Age`, `Senior Citizen`, `Dependents`. The model is utilized solely for customer retention outreach (offering discounts), not for service denial or punitive measures.
- **Data Freshness**: Customer behaviors change with competitor pricing shifts. Ongoing statistical drift monitoring (via PSI and KS tests) is required to detect market shifts.
- **Unverified Accuracy in Production**: Without live ground-truth churn labels (which only manifest after billing cycle completion), operational telemetry must rely on input drift monitoring rather than real-time accuracy claims.
