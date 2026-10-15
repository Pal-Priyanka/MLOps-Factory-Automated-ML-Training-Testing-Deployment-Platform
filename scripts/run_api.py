"""CLI runner to start FastAPI uvicorn server."""

import argparse
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import uvicorn
from mlops_factory.config import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Start MLOps Factory Inference REST API")
    parser.add_argument("--host", type=str, default=DEFAULT_CONFIG.api_host, help="API server host")
    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG.api_port, help="API server port")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload for development")
    args = parser.parse_args()

    print(f"[*] Starting MLOps Factory Inference API on http://{args.host}:{args.port}")
    print(f"[*] Interactive Swagger docs available at http://{args.host}:{args.port}/docs")
    print(f"[*] Prometheus metrics at http://{args.host}:{args.port}/metrics/prometheus")

    uvicorn.run(
        "mlops_factory.serving.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
