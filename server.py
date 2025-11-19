"""
Simple Flask server that initializes the DepthSplat encoder on startup and
exposes an endpoint to trigger encoding runs.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, jsonify, request
import spz

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


class WorkflowCoordinator:
    """
    Manages the workflow of:
    1. Requesting frames from capture server.
    2. Waiting for encoder processing (triggered via /process).
    3. Uploading the resulting PLY file.
    4. Looping back to step 1.
    """
    def __init__(self):
        # Configuration
        self.ply_upload_url = os.environ.get("PLY_UPLOAD_URL", "http://192.168.4.24:8080/api/upload")
        self.ply_upload_enabled = os.environ.get("PLY_UPLOAD_ENABLED", "true").lower() == "true"
        self.capture_server_url = os.environ.get("CAPTURE_SERVER_URL", "http://localhost:8080/trigger_capture")
        self.frame_request_enabled = os.environ.get("FRAME_REQUEST_ENABLED", "true").lower() == "true"
        self.retry_delay = 5  # Seconds

        # State
        self._session: Optional[requests.Session] = None
        self._session_lock = threading.Lock()
        self._completion_event = threading.Event()
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self.local_port = 8081

    def _get_session(self) -> requests.Session:
        """Get or create a persistent HTTP session with connection pooling and keep-alive."""
        with self._session_lock:
            if self._session is None:
                session = requests.Session()
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=0.1,
                    status_forcelist=[500, 502, 503, 504],
                )
                adapter = HTTPAdapter(
                    pool_connections=1,
                    pool_maxsize=1,
                    max_retries=retry_strategy,
                    pool_block=False,
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                session.headers.update({"Connection": "keep-alive"})
                self._session = session
                app.logger.info("Initialized persistent HTTP session for PLY uploads")
            return self._session

    def _ensure_connection(self):
        """Establish connection to upload server if not already connected."""
        session = self._get_session()
        try:
            parsed = urlparse(self.ply_upload_url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            session.head(base_url, timeout=5)
        except Exception:
            pass

    def start_continuous_requests(self, capture_server_url: Optional[str] = None, local_port: int = 8081, run_once: bool = False):
        """Start the background loop to request frames."""
        if not self.frame_request_enabled:
            app.logger.info("Frame requests disabled.")
            return

        if self._active:
            app.logger.info("Continuous requests already active.")
            return
        
        if capture_server_url:
            self.capture_server_url = capture_server_url
        
        self.local_port = local_port
        self._run_once = run_once

        self._active = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="workflow-coordinator"
        )
        self._thread.start()
        app.logger.info(f"Started continuous frame request loop targeting: {self.capture_server_url} (Run Once: {run_once})")

    def notify_processing_complete(self, capture_id: str, file_path: Optional[Union[str, Path]]):
        """
        Called when encoder finishes. 
        Handles file upload (if enabled) and signals the loop to continue.
        """
        # If we have a file and upload is enabled, do it in background
        if self.ply_upload_enabled and file_path:
            threading.Thread(
                target=self._upload_and_signal,
                args=(Path(file_path), capture_id),
                daemon=True,
                name=f"upload-{capture_id}"
            ).start()
        else:
            # No upload needed, just signal completion immediately
            if not file_path:
                 app.logger.info(f"No file generated for {capture_id}, skipping upload.")
            elif not self.ply_upload_enabled:
                 app.logger.info(f"Upload disabled, skipping upload for {capture_id}.")
            
            self._signal_completion()

    def _upload_and_signal(self, file_path: Path, capture_id: str):
        """Upload file then signal completion."""
        try:
            if not file_path.exists():
                app.logger.error(f"File missing: {file_path}")
                return

            session = self._get_session()
            self._ensure_connection()

            app.logger.info(f"Uploading file for {capture_id}: {file_path}")
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/octet-stream")}
                data = {"capture_id": capture_id}
                
                start_time = time.time()
                response = session.post(
                    self.ply_upload_url,
                    files=files,
                    data=data,
                    timeout=300
                )
                duration = time.time() - start_time
                response.raise_for_status()

            app.logger.info(
                f"Uploaded file for {capture_id} (Size: {file_size_mb:.2f}MB, Time: {duration:.2f}s)"
            )

        except Exception as e:
            app.logger.exception(f"Failed to upload file for {capture_id}: {e}")
        finally:
            # Always signal completion so the loop doesn't hang
            self._signal_completion()

    def _signal_completion(self):
        """Set the event to wake up the request loop."""
        app.logger.info("Processing cycle complete. Signaling next request.")
        self._completion_event.set()

    def _request_frames(self) -> bool:
        """
        Send trigger request to capture server. 
        Returns True if capture was triggered successfully.
        """
        # Clear event before requesting to ensure we catch the *new* completion
        self._completion_event.clear()
        
        try:
            app.logger.info(f"Requesting frames from {self.capture_server_url}")
            response = requests.post(self.capture_server_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            status = data.get("status")
            if status == "success":
                app.logger.info(f"Capture triggered: {data.get('capture_id')} (Clients: {data.get('connected_clients')})")
                return True
            elif status == "error":
                msg = data.get("message", "Unknown error")
                clients = data.get("connected_clients", 0)
                app.logger.warning(f"Capture trigger refused: {msg} (Clients: {clients})")
                return False
            else:
                app.logger.warning(f"Unexpected status: {status}")
                return False

        except Exception as e:
            app.logger.warning(f"Failed to request frames: {e}")
            return False

    def _wait_for_server_ready(self):
        """Poll /healthz to ensure local Flask server is up before triggering remote."""
        url = f"http://localhost:{self.local_port}/healthz"
        app.logger.info(f"Coordinator waiting for local server at {url}...")
        
        while self._active:
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    app.logger.info("Local server is ready. Starting workflow.")
                    return
            except requests.RequestException:
                pass
            
            time.sleep(1)

    def _loop(self):
        """Main loop: Request -> Wait for Processing -> Repeat."""
        # 1. Wait for local server to be ready
        self._wait_for_server_ready()
        
        while self._active:
            # 2. Trigger Capture
            if self._request_frames():
                # 3. Wait for completion
                if not self._completion_event.wait(timeout=600):
                    app.logger.error("Timeout waiting for processing completion. Resetting loop.")
                elif getattr(self, "_run_once", False):
                     app.logger.info("Run once enabled and processing complete. Exiting.")
                     os._exit(0)
            else:
                # Request failed, wait and retry
                time.sleep(self.retry_delay)


# Initialize coordinator
coordinator = WorkflowCoordinator()


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
        
        # Compress PLY to SPZ before sending
        if ply_path and Path(ply_path).exists():
            try:
                app.logger.info(f"Compressing PLY to SPZ: {ply_path}")
                spz_path = Path(ply_path).with_suffix(".spz")
                
                unpack_options = spz.UnpackOptions()
                cloud = spz.load_splat_from_ply(str(ply_path), unpack_options)
                
                pack_options = spz.PackOptions()
                spz.save_spz(cloud, pack_options, str(spz_path))
                
                app.logger.info(f"Compression complete: {spz_path}")
                ply_path = spz_path
            except Exception as e:
                app.logger.error(f"Failed to compress PLY to SPZ: {e}")
                # Proceed with original PLY if compression fails? 
                # For now, let's assume we want to fail or continue with PLY.
                # Given the requirement "compress... before sending", maybe we should just log error and send PLY.
        
        # Notify coordinator to handle upload and next request
        coordinator.notify_processing_complete(capture_id, ply_path)

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
        description="DepthSplat encoder server"
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
        help="Capture ID to use when uploading PLY file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("FLASK_RUN_HOST", "0.0.0.0"),
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FLASK_RUN_PORT", "8081")),
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--request-frames",
        action="store_true",
        help="Start loop to continuously request frames from capture server",
    )
    parser.add_argument(
        "--capture-server-url",
        type=str,
        default=None,
        help="URL of the capture server endpoint",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Exit after successfully processing one capture",
    )

    args = parser.parse_args()

    # Always configure logging so output is visible
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # CLI Mode: Upload PLY
    if args.upload_ply:
        ply_path = Path(args.upload_ply)
        capture_id = args.capture_id or ply_path.stem
        coordinator._upload_and_signal(ply_path, capture_id) # Reusing this method
        sys.exit(0)

    # Server Mode
    _initialize_encoder()
    
    if args.request_frames:
        coordinator.start_continuous_requests(args.capture_server_url, local_port=args.port, run_once=args.run_once)

    app.run(host=args.host, port=args.port)
