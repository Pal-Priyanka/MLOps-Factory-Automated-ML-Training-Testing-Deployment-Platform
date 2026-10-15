# System Architecture & Technical Design

## 1. Executive Overview

**MLOps Factory** is an automated machine learning training, validation, deployment, and operational monitoring platform designed for tabular customer retention workflows. The platform enforces strict separation of concerns, guarantees reproducible data transformations, eliminates data leakage, and provides production-grade API inference with operational telemetry.

---

## 2. Architecture Diagram

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data & Discovery Layer"]
        RAW["telco.csv\n(Local Raw Dataset)"]
        DISC["Data Discovery &\nSHA-256 Fingerprint"]
        VAL["Schema & Missingness\nValidator"]
        AUD["Leakage Audit\n(Drops 14 Leakage Cols)"]
        RAW --> DISC --> VAL --> AUD
    end

    subgraph Feature_Layer ["2. Feature Preprocessing"]
        CLEAN["Clean Feature Matrix\nX (36 Features) & y (Target)"]
        SPLIT["Stratified Holdout Split\n(Train 64% / Val 16% / Test 20%)"]
        PREP["Scikit-Learn ColumnTransformer\n(Numeric: Median + Scaler)\n(Categorical: Imputer + OHE)"]
        AUD --> CLEAN --> SPLIT --> PREP
    end

    subgraph Modeling_Layer ["3. Model Training & Evaluation"]
        BASE["Baseline Classifier\n(Logistic Regression)"]
        CAND["Candidate Classifier\n(HistGradientBoosting)"]
        EVAL_VAL["Validation Selection\n(Higher ROC-AUC)"]
        EVAL_TEST["Holdout Test Evaluation\n(ROC-AUC, Rec, Prec, F1, PR-AUC)"]
        GATES["Model Quality Acceptance Gates\n(ROC-AUC >= 0.82, Recall >= 0.70)"]

        PREP --> BASE
        PREP --> CAND
        BASE --> EVAL_VAL
        CAND --> EVAL_VAL
        EVAL_VAL --> EVAL_TEST --> GATES
    end

    subgraph Artifact_Layer ["4. Governance & Versioned Artifacts"]
        ART_M["artifacts/model.joblib\n(Fitted Pipeline)"]
        ART_META["artifacts/metadata.json\n(Provenance & SHA-256)"]
        ART_METR["artifacts/metrics.json\n(Test Metrics)"]
        GATES --> ART_M
        GATES --> ART_META
        GATES --> ART_METR
    end

    subgraph Serving_Layer ["5. Serving & Inference API (FastAPI)"]
        API["FastAPI Inference Server\n(:8000)"]
        EP_HEALTH["GET /health (Liveness)"]
        EP_READY["GET /ready (Readiness)"]
        EP_META["GET /metadata"]
        EP_PRED["POST /predict"]
        EP_BATCH["POST /predict/batch"]
        EP_METRICS["GET /metrics & /prometheus"]

        ART_M -.->|Verified Loading| API
        ART_META -.->|Metadata Verification| API
        API --> EP_HEALTH
        API --> EP_READY
        API --> EP_META
        API --> EP_PRED
        API --> EP_BATCH
        API --> EP_METRICS
    end

    subgraph Monitoring_Layer ["6. Operational Monitoring & Drift"]
        TRACK["In-Memory Telemetry Tracker\n(Request counts, Latency p50/p95)"]
        DRIFT["Statistical Drift Engine\n(PSI & Kolmogorov-Smirnov)"]
        REP_DRIFT["reports/drift_report.json"]

        EP_PRED --> TRACK
        EP_BATCH --> TRACK
        API -.-> DRIFT --> REP_DRIFT
    end
```

---

## 3. Component Breakdown

### 3.1 Data Ingestion & Discovery (`mlops_factory.data`)
- **Discovery (`discovery.py`)**: Recursively discovers dataset CSV files while ignoring `.git`, virtual environments, and generated directories. Calculates deterministic SHA-256 fingerprints to verify data provenance.
- **Validation (`validator.py`)**: Enforces required schema columns, checks binary target integrity, and executes a strict audit to detect and flag 14 post-event and identifier leakage columns.

### 3.2 Feature Preprocessing (`mlops_factory.features`)
- **Preprocessor (`preprocessor.py`)**: Constructs an encapsulated scikit-learn `ColumnTransformer`. 
  - **Numeric pipeline**: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
  - **Categorical pipeline**: `SimpleImputer(strategy='constant', fill_value='None')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  - Guarantees zero leakage by fitting strictly on training splits.

### 3.3 Training & Quality Gates (`mlops_factory.training`, `mlops_factory.evaluation`)
- **Training Orchestration (`train.py`)**: Orchestrates stratified splits (Train/Val/Test), trains both a balanced linear baseline (`LogisticRegression`) and a non-linear tree-based candidate (`HistGradientBoostingClassifier`), performs validation selection, and tests the winner on holdout test data.
- **Model Quality Gates (`gates.py`)**: Automated verification against predefined thresholds:
  - Minimum ROC-AUC: `0.82`
  - Minimum Recall (churners): `0.70`
  - Minimum Precision: `0.50`
  - Minimum F1-Score: `0.60`
  - Minimum Accuracy: `0.75`
  - Maximum Training Duration: `120.0s`

### 3.4 Serving Engine (`mlops_factory.serving`)
- **REST API (`app.py`)**: Built on FastAPI, offering sub-50ms inference latency.
- **Typed Contracts (`schemas.py`)**: Pydantic v2 schemas providing automatic validation, casting, and error responses.
- **Lifespan Management**: Validates model artifact checksum on startup; if artifact is missing or altered, gracefully degrades readiness (`503 Service Unavailable`) while keeping liveness (`200 OK`) active.

### 3.5 Monitoring & Telemetry (`mlops_factory.monitoring`)
- **Telemetry Tracker (`tracker.py`)**: Thread-safe tracker computing request counts, error counts, latency percentiles (`p50`, `p95`, `p99`), and prediction distribution. Supports Prometheus scraping.
- **Drift Engine (`drift.py`)**: Implements Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests to detect feature distribution shifts between production inference traffic and training reference distributions.

---

## 4. Security & Isolation Boundaries

1. **Non-Root Execution**: Container runs under dedicated unprivileged UID `10001` (`appuser`).
2. **Path Sanitization**: Error handlers catch internal exceptions and return standardized JSON without revealing stack traces, file paths, or system credentials.
3. **Model Integrity**: Serialized model files are hashed with SHA-256 during training; the serving layer verifies this hash on startup before unpickling.
4. **Data Isolation**: Raw datasets and local `.env` configuration files are strictly excluded from git tracking via `.gitignore` and `.dockerignore`.
