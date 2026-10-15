# Operational Runbook — MLOps Factory

## 1. System Overview

MLOps Factory runs as a stateless containerized REST API powered by FastAPI and Uvicorn, serving inferences from a pre-compiled Scikit-Learn Pipeline (`artifacts/model.joblib`) accompanied by a cryptographic governance manifest (`artifacts/metadata.json`).

| Component | Target Port | Endpoints | Health Probe |
| :--- | :---: | :--- | :--- |
| Inference API | `8000` | `/health`, `/ready`, `/metadata`, `/predict`, `/metrics` | `GET /health`, `GET /ready` |
| Metrics Exposition | `8000` | `/metrics`, `/metrics/prometheus` | Prometheus Scraper |

---

## 2. Standard Operating Procedures (SOP)

### SOP-01: Running the Automated Test Suite
Execute the full test suite locally before any deployment:
```bash
pytest -v --durations=10
```
Expected outcome: All 32 unit and integration tests PASS.

### SOP-02: Executing Model Retraining
Trigger the full reproducible training pipeline:
```bash
python scripts/run_pipeline.py
```
This updates:
- `artifacts/model.joblib`
- `artifacts/metadata.json`
- `artifacts/metrics.json`
- `artifacts/data_summary.json`
- `reports/model_evaluation_report.md`
- `reports/data_quality_report.md`

### SOP-03: Starting the Inference API Locally
```bash
python scripts/run_api.py --host 0.0.0.0 --port 8000
```
Interactive API docs are accessible at: `http://localhost:8000/docs`.

### SOP-04: Running with Docker Compose
```bash
# Build and run in detached mode
docker compose up --build -d

# Check container status
docker compose ps

# View live container logs
docker compose logs -f mlops-api

# Stop container
docker compose down
```

### SOP-05: Running Production Drift Simulation
```bash
python scripts/run_drift_simulation.py
```
Inspect generated outputs in `reports/drift_report.json`.

---

## 3. Incident Response Playbooks

### Playbook A: `/ready` Returns `503 Service Unavailable`
- **Symptom**: Load balancer removes instance from traffic rotation; `/ready` returns status 503 with `"Service unready"`.
- **Root Cause**:
  1. The model binary `artifacts/model.joblib` or metadata `artifacts/metadata.json` is missing.
  2. The model file was modified, triggering a SHA-256 checksum mismatch.
- **Remediation**:
  1. Verify the contents of the `artifacts/` directory:
     ```bash
     ls -l artifacts/
     ```
  2. Regenerate model artifacts via retraining:
     ```bash
     python scripts/run_pipeline.py
     ```
  3. Restart the API service.

### Playbook B: Statistical Drift Alert (PSI >= 0.25)
- **Symptom**: Drift report flags critical drift on `Monthly Charge`, `Contract`, or `Tenure in Months`.
- **Root Cause**: Macroeconomic changes, seasonal campaigns, or tariff restructuring shifted incoming subscriber distribution.
- **Remediation**:
  1. Extract latest production data snapshot and inspect PSI scores per feature.
  2. Confirm whether business pricing or promotions recently changed.
  3. Ingest fresh labeled data and trigger pipeline retraining.
  4. Compare candidate model metrics against baseline. If quality gates pass, promote new artifact.

### Playbook C: High Latency (> 250ms p95)
- **Symptom**: Telemetry `/metrics` reports `latency_ms.p95_ms > 250.0`.
- **Root Cause**: Excessive batch prediction sizes or CPU resource starvation.
- **Remediation**:
  1. Verify input request size; ensure batch requests respect the 1,000 item limit.
  2. Check host CPU utilization.
  3. Scale up replica count using Docker Compose or container orchestrator.
