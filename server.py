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

from inference import SetupResult, run_encoder, setup


app = Flask(__name__)

# Globals protected by a lock to avoid duplicate initialization.
_SETUP_LOCK = threading.Lock()
_SETUP_RESULT: Optional[SetupResult] = None
_SETUP_ERROR: Optional[str] = None
_SETUP_CONTEXT_PATH: Optional[Path] = None
_SETUP_IMAGE_KEY: Optional[Tuple[str, ...]] = None

_APP_ROOT = Path(__file__).resolve().parent


def _resolve_path(path_value: Union[str, Path]) -> Path:
    """Resolve incoming paths relative to the repo root if not absolute."""
    path = Path(path_value)
    if not path.is_absolute():
        path = (_APP_ROOT / path).resolve()
    return path


def _build_manifest_key(image_entries: Optional[List[Tuple[Path, Path]]]) -> Optional[Tuple[str, ...]]:
    """Create a stable cache key for a collection of image/metadata path pairs."""
    if not image_entries:
        return None

    normalized = [
        f"{metadata_path.resolve()}::{image_path.resolve()}"
        for metadata_path, image_path in image_entries
    ]
    normalized.sort()
    return tuple(normalized)


def _initialize_encoder(
    image_base_path: Optional[Union[str, Path]] = None,
    *,
    force: bool = False,
    image_entries: Optional[List[Tuple[Path, Path]]] = None,
) -> None:
    """Initialize the encoder for a specific image base path."""
    global _SETUP_RESULT, _SETUP_ERROR, _SETUP_CONTEXT_PATH, _SETUP_IMAGE_KEY

    with _SETUP_LOCK:
        requested_path: Optional[Path] = None
        if image_base_path is not None:
            requested_path = _resolve_path(image_base_path)

        manifest_key = _build_manifest_key(image_entries)

        # Already initialized for this path.
        cache_match = (
            _SETUP_RESULT is not None
            and _SETUP_CONTEXT_PATH == requested_path
            and _SETUP_IMAGE_KEY == manifest_key
        )

        if not force and cache_match:
            return

        # If we have a cached result for a different path and the caller did not force re-init,
        # reinitialize automatically so the cached context matches the requested path.
        if _SETUP_RESULT is not None and not cache_match:
            force = True

        if not force and _SETUP_RESULT is not None and requested_path is None:
            return

        try:
            setup_kwargs: Dict[str, Any] = {}
            if requested_path is not None:
                setup_kwargs["image_base_path"] = requested_path
            if image_entries:
                setup_kwargs["image_manifest"] = [
                    (metadata_path, image_path) for metadata_path, image_path in image_entries
                ]
            app.logger.info(
                "Initializing encoder via inference.setup() for path: %s (images: %s)",
                str(requested_path) if requested_path is not None else "<default>",
                "payload-provided" if image_entries else "directory-default",
            )
            _SETUP_RESULT = setup(**setup_kwargs)
            _SETUP_CONTEXT_PATH = requested_path
            _SETUP_IMAGE_KEY = manifest_key
            _SETUP_ERROR = None
            app.logger.info("Encoder initialized successfully for path: %s", _SETUP_CONTEXT_PATH or "<default>")
        except Exception as exc:  # noqa: BLE001 - we want to surface any failure.
            _SETUP_RESULT = None
            _SETUP_CONTEXT_PATH = None
            _SETUP_IMAGE_KEY = None
            _SETUP_ERROR = str(exc)
            app.logger.exception("Failed to initialize encoder: %s", exc)


def _extract_payload_image_base(payload: Dict[str, Any]) -> Tuple[str, Path, List[Tuple[Path, Path]]]:
    """Validate POST payload and extract capture ID, base path, and image entries."""
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

    image_base_path = metadata_dirs.pop()
    return capture_id, image_base_path, resolved_entries


@app.route("/healthz", methods=["GET"])
def healthcheck():
    """Basic health endpoint indicating encoder readiness."""
    status = "ready" if _SETUP_RESULT is not None else "idle"
    message = "Encoder ready." if _SETUP_RESULT else "Encoder awaiting initialization."

    if _SETUP_ERROR is not None:
        status = "error"
        message = _SETUP_ERROR

    response: Dict[str, Any] = {"status": status, "message": message}
    if _SETUP_CONTEXT_PATH is not None:
        response["image_base_path"] = str(_SETUP_CONTEXT_PATH)
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
        capture_id, image_base_path, image_entries = _extract_payload_image_base(payload)
    except (ValueError, FileNotFoundError) as payload_error:
        app.logger.warning("Invalid /process payload: %s", payload_error)
        return jsonify({"status": "error", "message": str(payload_error)}), 400

    try:
        _initialize_encoder(image_base_path, image_entries=image_entries)
    except Exception as exc:  # noqa: BLE001
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Encoder failed to initialize for requested path.",
                    "details": str(exc),
                }
            ),
            500,
        )

    try:
        run_kwargs: Dict[str, Any] = {}
        if isinstance(num_runs, int) and num_runs > 0:
            run_kwargs["num_runs"] = num_runs

        if isinstance(output_dir, str) and output_dir:
            resolved_output_dir = _resolve_path(output_dir)
            run_kwargs["output_dir"] = resolved_output_dir
            app.logger.info("Overriding output directory via payload: %s", resolved_output_dir)
        else:
            resolved_output_dir = _resolve_path(os.environ.get("DEPTHSPLAT_OUTPUT_ROOT", "run-output"))
            run_kwargs["output_dir"] = resolved_output_dir
            app.logger.info("Using shared output directory: %s", resolved_output_dir)

        if _SETUP_RESULT is None:
            raise RuntimeError("Encoder initialization missing after setup.")

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
    default_image_base = os.environ.get("DEPTHSPLAT_IMAGE_BASE_PATH")
    if default_image_base:
        try:
            _initialize_encoder(default_image_base)
        except Exception as exc:  # noqa: BLE001
            app.logger.warning("Deferred encoder initialization failed during app creation: %s", exc)
    return app


if __name__ == "__main__":
    # Allow overriding host/port via environment for flexibility.
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", "8081"))

    default_image_base = os.environ.get("DEPTHSPLAT_IMAGE_BASE_PATH")
    if default_image_base:
        _initialize_encoder(default_image_base)
    app.run(host=host, port=port)

