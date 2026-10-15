# Technical Demonstration Script (Evaluation & Portfolio Presentation)

This script is designed for an undergraduate technical viva, senior engineering review, or portfolio demonstration. It provides a structured, 5-to-7 minute demonstration showcasing the complete end-to-end MLOps lifecycle.

---

## 1. Introduction & Executive Pitch (1 Minute)

> *"Good morning / afternoon. Today I am presenting **MLOps Factory**, an industry-grade, end-to-end machine learning operations platform built on top of telecommunications subscriber retention data.*  
>  
> *Rather than treating machine learning as an isolated Jupyter notebook, this project provides a production-grade engineering platform featuring: automated data discovery and leakage auditing, reproducible scikit-learn pipelines with quality acceptance gates, SHA-256 artifact governance, sub-50ms REST API inference using FastAPI, and real-time operational drift monitoring using Population Stability Index (PSI) and Kolmogorov-Smirnov statistics."*

---

## 2. Data Engineering & Leakage Audit (1.5 Minutes)

**Action**: Open `reports/data_quality_report.md` or run validation.

> *"First, in the data engineering layer, we ingest the 50-column IBM Telco Churn dataset (7,043 customer accounts). Many naive ML tutorials achieve artificial 95%+ accuracy because of severe retrospective data leakage. In our platform, we implemented an automated leakage detection audit that identifies and strictly removes 14 identifier and post-event columns:*  
> - *Primary keys like `Customer ID`.*  
> - *Direct target duplicates like `Customer Status`.*  
> - *Post-churn exit reasons like `Churn Reason` and `Churn Category` which are populated exclusively after a customer cancels.*  
> - *Exit interview ratings like `Satisfaction Score` which artificially separate churners.*  
> - *Pre-existing predictive scores like `Churn Score`.*  
>  
> *By stripping these columns, our feature matrix uses strictly leak-free operational predictors (36 features) fitted exclusively on training splits."*

---

## 3. Automated Training & Quality Gates (1.5 Minutes)

**Action**: Run or showcase `scripts/run_pipeline.py`.

```bash
python scripts/run_pipeline.py
```

> *"Next, let's execute the reproducible training orchestrator. The pipeline automatically fits both a linear baseline (`LogisticRegression` with balanced class weights) and a non-linear tree candidate (`HistGradientBoostingClassifier`).*  
>  
> *Model selection occurs on a dedicated validation split, and final evaluation is executed on an untouched holdout test set (1,409 accounts):*  
> - *Our baseline achieves **0.9087 ROC-AUC** and an **87.7% Recall** on actual churners.*  
> - *Our candidate achieves **0.9043 ROC-AUC** and **83.1% Accuracy**.*  
>  
> *Before any artifact can be released, it must pass automated **Model Quality Gates**: requiring ROC-AUC >= 0.82, Recall >= 0.70, and Precision >= 0.50. When gates pass, the pipeline saves the fitted pipeline (`model.joblib`), metadata manifest (`metadata.json`), and cryptographic SHA-256 checksums for provenance."*

---

## 4. Serving & Live Real-Time Inference (1.5 Minutes)

**Action**: Start the API server and show Swagger UI.

```bash
python scripts/run_api.py
```
Open browser to: `http://localhost:8000/docs`

> *"Here is our production serving engine built on FastAPI:*  
> - *`GET /health` provides liveness status and service uptime.*  
> - *`GET /ready` checks that the model artifact is loaded and verified.*  
> - *`GET /metadata` displays complete model lineage, feature schemas, and training metrics.*  
> - *`POST /predict` accepts typed Pydantic payloads, formats them via our scikit-learn preprocessor, and returns the churn probability, risk tier (`Low`, `Medium`, `High`), and actionable retention recommendations in under 50 milliseconds.*  
> - *`POST /predict/batch` processes batches up to 1,000 customers with bulk vectorization.*  
>  
> *Notice that invalid payloads (such as negative ages or malformed data types) are caught immediately with HTTP 422 structured error messages without exposing system internals."*

---

## 5. Operational Telemetry & Statistical Drift Monitoring (1 Minute)

**Action**: Run drift simulation or show `http://localhost:8000/metrics`.

```bash
python scripts/run_drift_simulation.py
```

> *"Finally, we maintain operational reliability through real-time monitoring:*  
> - *`GET /metrics/prometheus` exposes latency percentiles (`p50`, `p95`, `p99`) and prediction class counts.*  
> - *Our drift detection engine computes the **Population Stability Index (PSI)** and **Kolmogorov-Smirnov two-sample statistics** to detect distribution shifts in incoming inference data (e.g. inflation-driven changes in monthly charges or contract duration).*  
> - *Importantly, our system explicitly distinguishes statistical input drift from verified model accuracy degradation, since true ground-truth churn labels only become available at billing cycle completion.*  
>  
> *The entire codebase is verified by 32 automated unit and integration tests passing in under 2 seconds, and is fully containerized with Docker and automated via GitHub Actions CI."*
