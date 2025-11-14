"""
Simple Flask server that initializes the DepthSplat encoder on startup and
exposes an endpoint to trigger encoding runs.
"""
from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from flask import Flask, jsonify, request

from inference import SetupResult, run_encoder, setup_encoder


app = Flask(__name__)

# Globals protected by a lock to avoid duplicate initialization.
_SETUP_LOCK = threading.Lock()
_SETUP_RESULT: Optional[SetupResult] = None
_SETUP_ERROR: Optional[str] = None

_APP_ROOT = Path(__file__).resolve().parent


def _resolve_path(path_value: Union[str, Path]) -> Path:
    """Resolve incoming paths relative to the repo root if not absolute."""
    path = Path(path_value)
    if not path.is_absolute():
        path = (_APP_ROOT / path).resolve()
    return path


def _initialize_encoder() -> None:
    """Initialize the encoder exactly once for the process lifetime."""
    global _SETUP_RESULT, _SETUP_ERROR

    with _SETUP_LOCK:
        if _SETUP_RESULT is not None:
            return

        try:
            app.logger.info("Initializing encoder via inference.setup_encoder()")
            _SETUP_RESULT = setup_encoder()
            _SETUP_ERROR = None
            app.logger.info("Encoder initialized successfully.")
        except Exception as exc:  # noqa: BLE001 - we want to surface any failure.
            _SETUP_RESULT = None
            _SETUP_ERROR = str(exc)
            app.logger.exception("Failed to initialize encoder: %s", exc)
            raise


def _extract_payload_image_base(payload: Dict[str, Any]) -> Tuple[str, List[Tuple[Path, Path]]]:
    """Validate POST payload and extract capture ID with resolved image entries."""
    capture_id = payload.get("capture_id")
    if not capture_id or not isinstance(capture_id, str):
        raise ValueError("`capture_id` must be provided as a non-empty string.")

    images = payload.get("images")
    if not images or not isinstance(images, list):
        raise ValueError("`images` must be provided as a non-empty list.")

    if any(not isinstance(item, dict) for item in images):
        raise ValueError("`images` entries must be objects containing file paths.")

    metadata_dirs = set()
    missing_files: List[str] = []
    resolved_entries: List[Tuple[Path, Path]] = []

    for item in images:
        metadata_path = item.get("metadata_path")
        image_path = item.get("image_path")
        if not metadata_path or not image_path:
            raise ValueError("Each image entry must include `image_path` and `metadata_path`.")

        resolved_metadata = _resolve_path(metadata_path)
        resolved_image = _resolve_path(image_path)

        if not resolved_metadata.exists():
            missing_files.append(str(resolved_metadata))
        if not resolved_image.exists():
            missing_files.append(str(resolved_image))

        metadata_dirs.add(resolved_metadata.parent)
        resolved_entries.append((resolved_metadata, resolved_image))

    if missing_files:
        raise FileNotFoundError(f"Missing files for encoder run: {missing_files}")

    if len(metadata_dirs) != 1:
        raise ValueError("All metadata files must share a single parent directory.")

    return capture_id, resolved_entries


@app.route("/healthz", methods=["GET"])
def healthcheck():
    """Basic health endpoint indicating encoder readiness."""
    status = "ready" if _SETUP_RESULT is not None else "idle"
    message = "Encoder ready." if _SETUP_RESULT else "Encoder awaiting initialization."

    if _SETUP_ERROR is not None:
        status = "error"
        message = _SETUP_ERROR

    response: Dict[str, Any] = {"status": status, "message": message}
    return jsonify(response)


@app.route("/process", methods=["POST"])
def process_route():
    """
    Trigger an encoder run.

    Optional JSON body fields:
        - num_runs: override number of encoder runs
        - output_dir: override output directory path
    """
    payload: Dict[str, Any] = request.get_json(force=False, silent=True) or {}
    num_runs = payload.get("num_runs")
    output_dir = payload.get("output_dir")

    try:
        capture_id, image_entries = _extract_payload_image_base(payload)
    except (ValueError, FileNotFoundError) as payload_error:
        app.logger.warning("Invalid /process payload: %s", payload_error)
        return jsonify({"status": "error", "message": str(payload_error)}), 400


    try:
        _initialize_encoder()
        if _SETUP_RESULT is None:
            raise RuntimeError("Encoder is not initialized.")

        images = [str(image_path) for _, image_path in image_entries]
        run_kwargs: Dict[str, Any] = {"images": images}

        if num_runs is not None:
            run_kwargs["num_runs"] = num_runs
        if output_dir is not None:
            run_kwargs["output_dir"] = _resolve_path(output_dir)

        app.logger.info(
            "Starting encoder run for capture %s with %d images", capture_id, len(images)
        )
        inference_result = run_encoder(_SETUP_RESULT, **run_kwargs)

        response_body = {
            "status": "success",
            "capture_id": capture_id,
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
    try:
        _initialize_encoder()
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Deferred encoder initialization failed during app creation: %s", exc)
    return app


if __name__ == "__main__":
    # Allow overriding host/port via environment for flexibility.
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", "8081"))


    _initialize_encoder()
    app.run(host=host, port=port)

