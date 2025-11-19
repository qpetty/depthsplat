"""
Simple Flask server that initializes the DepthSplat encoder on startup and
exposes an endpoint to trigger encoding runs.
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from flask import Flask, jsonify, request

from inference import SetupResult, run_encoder, setup_encoder


app = Flask(__name__)

# Globals protected by a lock to avoid duplicate initialization.
_SETUP_LOCK = threading.Lock()
_SETUP_RESULT: Optional[SetupResult] = None
_SETUP_ERROR: Optional[str] = None

_APP_ROOT = Path(__file__).resolve().parent

# Configuration for PLY upload - hardcoded IP with environment variable override
# Default to a common local network IP (update with your actual destination IP)
_PLY_UPLOAD_URL = os.environ.get("PLY_UPLOAD_URL", "http://192.168.4.24:8080/api/upload")
_PLY_UPLOAD_ENABLED = os.environ.get("PLY_UPLOAD_ENABLED", "true").lower() == "true"


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


def _get_logger():
    """Get a logger that works in both Flask and standalone contexts."""
    try:
        return app.logger
    except RuntimeError:
        # Flask app context not available, use standard logger
        return logging.getLogger(__name__)


def _upload_ply_file(ply_path: Path, capture_id: str) -> None:
    """
    Upload PLY file to external service in the background.
    
    This function runs in a separate thread and won't block the main request.
    Failures are logged but not retried - another PLY will be sent shortly.
    """
    logger = _get_logger()
    
    if not _PLY_UPLOAD_ENABLED:
        logger.debug("PLY upload disabled, skipping upload for capture %s", capture_id)
        return
    
    if not ply_path or not ply_path.exists():
        logger.warning("PLY file does not exist, cannot upload: %s", ply_path)
        return
    
    # Record start time for upload duration measurement
    upload_start_time = time.time()
    
    try:
        logger.info("Starting PLY upload for capture %s: %s -> %s", capture_id, ply_path, _PLY_UPLOAD_URL)
        
        # Stream the file to avoid loading entire 66MB into memory
        with open(ply_path, "rb") as f:
            files = {"file": (ply_path.name, f, "application/octet-stream")}
            data = {"capture_id": capture_id}
            
            response = requests.post(
                _PLY_UPLOAD_URL,
                files=files,
                data=data,
                timeout=300,  # 5 minute timeout for large file
            )
            response.raise_for_status()
        
        # Calculate and log upload duration
        upload_duration = time.time() - upload_start_time
        file_size_mb = ply_path.stat().st_size / (1024 * 1024)
        
        logger.info(
            "Successfully uploaded PLY for capture %s: %s (status: %d, size: %.2f MB, duration: %.2f seconds)",
            capture_id,
            ply_path,
            response.status_code,
            file_size_mb,
            upload_duration,
        )
    except Exception as e:
        # Calculate duration even on failure
        upload_duration = time.time() - upload_start_time
        
        # Log and fail silently - another PLY will be sent shortly
        logger.warning(
            "PLY upload failed for capture %s: %s. Error: %s (duration: %.2f seconds, will retry with next PLY)",
            capture_id,
            ply_path,
            e,
            upload_duration,
        )


def _queue_ply_upload(ply_path: Optional[Path], capture_id: str) -> None:
    """Queue a PLY upload task to be processed in the background."""
    if ply_path and ply_path.exists():
        # Start upload in a daemon thread so it doesn't block server shutdown
        upload_thread = threading.Thread(
            target=_upload_ply_file,
            args=(ply_path, capture_id),
            daemon=True,
            name=f"ply-upload-{capture_id}",
        )
        upload_thread.start()
        app.logger.debug("Queued PLY upload thread for capture %s", capture_id)
    else:
        app.logger.debug("No PLY file to upload for capture %s", capture_id)


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

        ply_path = inference_result.get("ply_path")
        
        # Queue the upload in a background thread (non-blocking)
        if ply_path:
            resolved_ply_path = Path(ply_path) if isinstance(ply_path, str) else ply_path
            _queue_ply_upload(resolved_ply_path, capture_id)

        response_body = {
            "status": "success",
            "capture_id": capture_id,
            "encoder_total_time": inference_result.get("encoder_total_time"),
            "encoder_average_time": inference_result.get("encoder_average_time"),
            "ply_export_time": inference_result.get("ply_export_time"),
            "device": inference_result.get("device"),
            "num_runs": inference_result.get("num_runs"),
            "ply_path": str(ply_path) if ply_path else None,
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
    parser = argparse.ArgumentParser(
        description="DepthSplat encoder server with optional PLY upload capability"
    )
    parser.add_argument(
        "--upload-ply",
        type=str,
        metavar="PATH",
        help="Upload a PLY file to the configured destination and exit",
    )
    parser.add_argument(
        "--capture-id",
        type=str,
        metavar="ID",
        help="Capture ID to use when uploading PLY file (defaults to filename stem if not provided)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("FLASK_RUN_HOST", "0.0.0.0"),
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FLASK_RUN_PORT", "8081")),
        help="Port to bind the server to (default: 8081)",
    )

    args = parser.parse_args()

    # If --upload-ply is provided, upload the file and exit
    if args.upload_ply:
        ply_path = Path(args.upload_ply)
        if not ply_path.exists():
            print(f"Error: PLY file not found: {ply_path}", file=sys.stderr)
            sys.exit(1)

        capture_id = args.capture_id or ply_path.stem
        print(f"Uploading PLY file: {ply_path}")
        print(f"Capture ID: {capture_id}")
        print(f"Destination: {_PLY_UPLOAD_URL}")

        # Configure logging for standalone upload
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Upload the file (this will run synchronously since we're exiting anyway)
        _upload_ply_file(ply_path, capture_id)
        sys.exit(0)

    # Otherwise, run the Flask server
    host = args.host
    port = args.port

    _initialize_encoder()
    app.run(host=host, port=port)

