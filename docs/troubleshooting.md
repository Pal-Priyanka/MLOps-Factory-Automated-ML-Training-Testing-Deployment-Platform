# Troubleshooting & Diagnostic Guide

This guide covers common issues, root cause diagnostics, and step-by-step recovery solutions for **MLOps Factory**.

---

## 1. Environment & Setup Issues

### Issue 1.1: `docker` Command Not Found
- **Symptom**: `docker : The term 'docker' is not recognized as the name of a cmdlet`
- **Cause**: Docker Desktop is not installed, not running, or not added to your system `PATH`.
- **Solution**:
  - If evaluating on a machine without Docker, the application can run natively via standard Python:
    ```bash
    pip install -r requirements.txt
    python scripts/run_api.py
    ```
  - To install Docker Desktop, visit [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/) and ensure WSL 2 backend integration is enabled.

---

### Issue 1.2: Port 8000 Already in Use
- **Symptom**: `[Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000): only one usage of each socket address`
- **Cause**: Another process or background server is currently bound to port 8000.
- **Solution**:
  - Identify and terminate the lingering process on port 8000:
    ```powershell
    Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
    ```
  - Alternatively, launch the API on an alternative port:
    ```bash
    python scripts/run_api.py --port 8080
    ```

---

## 2. Model Training & Artifact Diagnostics

### Issue 2.1: Missing Dataset Error
- **Symptom**: `FileNotFoundError: Dataset 'telco.csv' could not be located in ...`
- **Cause**: The raw dataset was moved, deleted, or placed into an excluded folder (e.g. `.venv/` or `artifacts/`).
- **Solution**:
  - Verify that `telco.csv` is present in the workspace root or pass its location explicitly:
    ```bash
    python scripts/run_pipeline.py --data-path path/to/your/telco.csv
    ```

---

### Issue 2.2: Model Checksum Mismatch
- **Symptom**: `ValueError: Model artifact checksum mismatch! File may be corrupted or modified.`
- **Cause**: `artifacts/model.joblib` was edited, partially overwritten, or replaced without updating `artifacts/metadata.json`.
- **Solution**:
  - Re-run the training pipeline to regenerate synchronized binary and metadata manifests:
    ```bash
    python scripts/run_pipeline.py
    ```

---

## 3. Inference API & HTTP Errors

### Issue 3.1: 422 Unprocessable Entity
- **Symptom**: API returns `422 Unprocessable Entity` with message: `"Supplied input payload does not match expected schema contract."`
- **Cause**: Missing required customer fields, invalid numeric ranges (e.g. negative age), or mismatched payload formatting.
- **Solution**:
  - Inspect the validation details in the error response body. Ensure fields match `CustomerFeatures` schema contract defined in `src/mlops_factory/serving/schemas.py`.

---

### Issue 3.2: 503 Service Unavailable on `/predict`
- **Symptom**: `503 Service Unavailable: "Model is not ready to serve predictions."`
- **Cause**: The API was launched before the model training pipeline was executed.
- **Solution**:
  - Run the training pipeline first:
    ```bash
    python scripts/run_pipeline.py
    ```
  - Check the logs to ensure `artifacts/model.joblib` and `artifacts/metadata.json` were created, then restart the API.
