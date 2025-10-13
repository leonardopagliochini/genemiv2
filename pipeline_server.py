#!/usr/bin/env python3
"""
Lightweight web server exposing the pipeline workflow through a HTML/JS GUI.

It provides:
  - POST /api/run to start a new pipeline execution with user parameters.
  - GET  /api/stream/<run_id> to stream step-by-step logs and progress.
  - Static assets served from ./web for the browser interface.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    request,
    send_from_directory,
    stream_with_context,
)

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_SCRIPT = REPO_ROOT / "pipeline.py"
STATIC_DIR = REPO_ROOT / "web"

STAGE_NAMES = [
    "Compiling sources",
    "Running simulation",
    "Extracting mesh subsets",
    "Extracting surfaces",
]


@dataclass
class RunState:
    queue: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    thread: Optional[threading.Thread] = None
    process: Optional[subprocess.Popen] = None
    done: bool = False
    returncode: Optional[int] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


runs: Dict[str, RunState] = {}

app = Flask(
    __name__,
    static_folder=STATIC_DIR.as_posix(),
    static_url_path="",
)


def publish_event(run_id: str, payload: dict) -> None:
    state = runs.get(run_id)
    if state is None:
        return
    state.queue.put(payload)


def build_pipeline_command(params: dict) -> list[str]:
    cmd = ["python3", str(PIPELINE_SCRIPT), "--extract-time", str(params["extract_time"])]
    if params.get("simulation_time") is not None:
        cmd.extend(["--simulation-time", str(params["simulation_time"])])
    if params.get("mesh"):
        cmd.extend(["--mesh", str(params["mesh"])])
    if params.get("procs") is not None:
        cmd.extend(["--procs", str(params["procs"])])
    return cmd


def monitor_pipeline(run_id: str, params: dict) -> None:
    state = runs[run_id]
    stage_index = {name: idx for idx, name in enumerate(STAGE_NAMES)}
    current_stage_idx: Optional[int] = None
    stages_status = {name: "pending" for name in STAGE_NAMES}

    def set_stage(stage: str, status: str) -> None:
        stages_status[stage] = status
        publish_event(run_id, {"type": "stage", "stage": stage, "status": status})

    def complete_previous() -> None:
        nonlocal current_stage_idx
        if current_stage_idx is None:
            return
        stage = STAGE_NAMES[current_stage_idx]
        if stages_status[stage] == "running":
            set_stage(stage, "completed")

    cmd = build_pipeline_command(params)
    publish_event(
        run_id,
        {
            "type": "init",
            "stages": [{"name": name, "status": "pending"} for name in STAGE_NAMES],
            "command": cmd,
        },
    )

    try:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        publish_event(
            run_id,
            {"type": "error", "message": f"Unable to start pipeline: {exc}"},
        )
        set_stage(STAGE_NAMES[0], "error")
        state.returncode = -1
        publish_event(run_id, {"type": "done", "returncode": -1})
        state.done = True
        return

    state.process = process

    try:
        assert process.stdout is not None  # For type checkers
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            publish_event(run_id, {"type": "log", "message": line})

            if line.startswith("[step]"):
                description = line[len("[step]") :].strip()
                if description in stage_index:
                    complete_previous()
                    current_stage_idx = stage_index[description]
                    set_stage(description, "running")
                continue

            if line.startswith("[skip]"):
                # Treat any skip as completion of the simulation stage.
                sim_stage = "Running simulation"
                if stages_status.get(sim_stage) in {"pending", "running"}:
                    current_stage_idx = stage_index[sim_stage]
                    set_stage(sim_stage, "skipped")
                continue

            if line.startswith("[warn]") and "pvpython" in line:
                stage = "Extracting surfaces"
                if stages_status.get(stage) in {"pending", "running"}:
                    set_stage(stage, "skipped")
                continue

            if line.startswith("[error]"):
                if current_stage_idx is not None:
                    stage = STAGE_NAMES[current_stage_idx]
                    set_stage(stage, "error")

    except Exception as exc:  # pragma: no cover - defensive
        publish_event(
            run_id, {"type": "error", "message": f"Server error while streaming logs: {exc}"}
        )
    finally:
        process.wait()
        state.returncode = process.returncode

        # Close stdout explicitly to avoid resource warnings.
        if process.stdout is not None:
            process.stdout.close()

        # Finalise stage status if we ended cleanly.
        if process.returncode == 0:
            complete_previous()
            for name, status in list(stages_status.items()):
                if status == "running":
                    set_stage(name, "completed")
        else:
            for name, status in list(stages_status.items()):
                if status == "running":
                    set_stage(name, "error")

        publish_event(run_id, {"type": "done", "returncode": process.returncode})
        state.done = True


@app.route("/")
def index() -> Response:
    return send_from_directory(STATIC_DIR, "index.html")


@app.post("/api/run")
def start_run() -> Response:
    if not PIPELINE_SCRIPT.exists():
        return jsonify({"error": "pipeline.py not found. Please create it first."}), 500

    payload = request.get_json(silent=True) or {}
    errors = {}

    def parse_int(name: str, required: bool = False) -> Optional[int]:
        value = payload.get(name)
        if value in (None, ""):
            if required:
                errors[name] = "This field is required."
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            errors[name] = "Must be an integer."
            return None

    extract_time = parse_int("extractTime", required=True)
    procs = parse_int("procs")
    simulation_time = payload.get("simulationTime")
    mesh = payload.get("mesh") or "mesh/MNI_with_phys.msh"

    if simulation_time not in (None, ""):
        try:
            simulation_time = float(simulation_time)
        except (TypeError, ValueError):
            errors["simulationTime"] = "Must be a number."
    else:
        simulation_time = None

    if extract_time is not None and extract_time < 0:
        errors["extractTime"] = "Must be >= 0."
    if procs is not None and procs <= 0:
        errors["procs"] = "Must be > 0."

    mesh_path = Path(mesh)
    if not mesh_path.is_absolute():
        mesh_path = REPO_ROOT / mesh_path
    if not mesh_path.exists():
        errors["mesh"] = f"Mesh file '{mesh_path}' not found."

    if errors:
        return jsonify({"errors": errors}), 400

    run_id = uuid.uuid4().hex
    params = {
        "extract_time": extract_time,
        "simulation_time": simulation_time,
        "mesh": str(mesh_path),
        "procs": procs if procs is not None else 4,
    }

    state = RunState()
    runs[run_id] = state

    thread = threading.Thread(
        target=monitor_pipeline,
        args=(run_id, params),
        name=f"pipeline-{run_id[:8]}",
        daemon=True,
    )
    state.thread = thread
    thread.start()

    return jsonify({"runId": run_id})


@app.get("/api/stream/<run_id>")
def stream(run_id: str) -> Response:
    state = runs.get(run_id)
    if state is None:
        abort(404)

    def event_stream():
        while True:
            try:
                event = state.queue.get(timeout=0.5)
            except queue.Empty:
                if state.done:
                    break
                continue
            yield f"data: {json.dumps(event)}\n\n"

        # Allow clients to clean up gracefully.
        yield f"data: {json.dumps({'type': 'close'})}\n\n"
        runs.pop(run_id, None)

    response = Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
    )
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
