# Data Quality & Leakage Audit Report

**Total Records**: `7043`  
**Total Columns**: `50`  
**Duplicate Rows**: `0`  
**Overall Validation Status**: `VALID`  

## 1. Target Distribution

| Churn Class | Count | Percentage |
| :--- | :---: | :---: |
| `No` | 5174 | 73.46% |
| `Yes` | 1869 | 26.54% |

## 2. Leakage Audit & Excluded Columns

The following columns were identified and strictly removed from the model feature matrix to prevent data leakage and artificial inflation of evaluation scores:

| Excluded Column | Category | Justification |
| :--- | :--- | :--- |
| `Customer ID` | Identifier | Primary key; causes identity memorization rather than generalization. |
| `Country`, `State`, `Quarter` | Zero Variance | Constant across all records (`United States`, `California`, `Q3`). |
| `City`, `Zip Code`, `Latitude`, `Longitude` | High Cardinality | Spatial identifiers; `Population` provides clean demographic sizing without overfitting. |
| `Customer Status` | Target Duplicate | Contains ground-truth statuses: `Stayed`, `Churned`, `Joined`. |
| `Churn Category` | Post-Event Leakage | Explains why customer churned; populated exclusively for churned accounts (73.5% missing on retained accounts). |
| `Churn Reason` | Post-Event Leakage | Exit interview reason; populated exclusively for churned accounts. |
| `Churn Score` | Target Leakage | Vendor-generated predictive risk score (0-100) precalculated by IBM. |
| `Satisfaction Score` | Post-Event Leakage | Exit interview survey score (1-5); scores 1 & 2 are 100% churned, scores 4 & 5 are 100% retained. |

## 3. Configured Features

- **Numeric Features (14)**: `Age, Number of Dependents, Population, Number of Referrals, Tenure in Months, Avg Monthly Long Distance Charges, Avg Monthly GB Download, Monthly Charge, Total Charges, Total Refunds, Total Extra Data Charges, Total Long Distance Charges, Total Revenue, CLTV`
- **Categorical Features (22)**: `Gender, Under 30, Senior Citizen, Married, Dependents, Referred a Friend, Offer, Phone Service, Multiple Lines, Internet Service, Internet Type, Online Security, Online Backup, Device Protection Plan, Premium Tech Support, Streaming TV, Streaming Movies, Streaming Music, Unlimited Data, Contract, Paperless Billing, Payment Method`
