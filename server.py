"""
Simple Flask server that initializes the DepthSplat encoder on startup and
exposes an endpoint to trigger encoding runs.
"""
from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

from inference import SetupResult, run_encoder, setup


app = Flask(__name__)

# Globals protected by a lock to avoid duplicate initialization.
_SETUP_LOCK = threading.Lock()
_SETUP_RESULT: Optional[SetupResult] = None
_SETUP_ERROR: Optional[str] = None


def _initialize_encoder() -> None:
    """Initialize the encoder once, caching the setup result for reuse."""
    global _SETUP_RESULT, _SETUP_ERROR

    with _SETUP_LOCK:
        # Already initialized or failed previously.
        if _SETUP_RESULT is not None or _SETUP_ERROR is not None:
            return

        try:
            app.logger.info("Initializing encoder via inference.setup()")
            _SETUP_RESULT = setup()
            app.logger.info("Encoder initialized successfully.")
        except Exception as exc:  # noqa: BLE001 - we want to surface any failure.
            _SETUP_ERROR = str(exc)
            app.logger.exception("Failed to initialize encoder: %s", exc)


@app.route("/healthz", methods=["GET"])
def healthcheck():
    """Basic health endpoint indicating encoder readiness."""
    status = "ready" if _SETUP_RESULT is not None else "initializing"
    message = "Encoder ready." if _SETUP_RESULT else "Encoder not yet ready."

    if _SETUP_ERROR is not None:
        status = "error"
        message = _SETUP_ERROR

    return jsonify({"status": status, "message": message})


@app.route("/process", methods=["POST"])
def process_route():
    """
    Trigger an encoder run.

    Optional JSON body fields:
        - num_runs: override number of encoder runs
        - output_dir: override output directory path
    """
    if _SETUP_ERROR is not None:
        return (
            jsonify({"status": "error", "message": "Encoder failed to initialize.", "details": _SETUP_ERROR}),
            500,
        )

    if _SETUP_RESULT is None:
        # Try lazy initialization in case the first request arrives before the
        # before_first_request hook fires (for example in some WSGI setups).
        _initialize_encoder()

    if _SETUP_RESULT is None:
        return (
            jsonify({"status": "error", "message": "Encoder is still initializing. Try again later."}),
            503,
        )

    payload: Dict[str, Any] = request.get_json(force=False, silent=True) or {}
    num_runs = payload.get("num_runs")
    output_dir = payload.get("output_dir")

    try:
        run_kwargs: Dict[str, Any] = {}
        if isinstance(num_runs, int) and num_runs > 0:
            run_kwargs["num_runs"] = num_runs
        if isinstance(output_dir, str) and output_dir:
            run_kwargs["output_dir"] = Path(output_dir)

        inference_result = run_encoder(_SETUP_RESULT, **run_kwargs)

        response_body = {
            "status": "success",
            "encoder_total_time": inference_result.get("encoder_total_time"),
            "encoder_average_time": inference_result.get("encoder_average_time"),
            "ply_export_time": inference_result.get("ply_export_time"),
            "device": inference_result.get("device"),
            "num_runs": inference_result.get("num_runs"),
            "ply_path": str(inference_result["ply_path"]) if inference_result.get("ply_path") else None,
        }

        return jsonify(response_body)
    except Exception as exc:  # noqa: BLE001 - surface inference failures.
        app.logger.exception("Encoder run failed: %s", exc)
        return jsonify({"status": "error", "message": "Encoder run failed.", "details": str(exc)}), 500


def create_app() -> Flask:
    """Factory for WSGI servers."""
    _initialize_encoder()
    return app


if __name__ == "__main__":
    # Allow overriding host/port via environment for flexibility.
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", "8081"))

    _initialize_encoder()
    app.run(host=host, port=port)

