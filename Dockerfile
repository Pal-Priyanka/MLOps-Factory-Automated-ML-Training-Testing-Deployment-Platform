# ==============================================================================
# MLOps Factory — Production Inference Container
# Multi-stage, Non-root, Lightweight runtime
# ==============================================================================

# Build Stage: Prepare wheels and dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Runtime Stage: Secure, minimal image
FROM python:3.12-slim AS runner

# System environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH"

# Create unprivileged service user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application source, scripts, and production model artifacts
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY artifacts/ /app/artifacts/

# Set ownership to unprivileged user
RUN chown -R appuser:appgroup /app /home/appuser

USER appuser

EXPOSE 8000

# Container Healthcheck using built-in standard python library
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["uvicorn", "mlops_factory.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
