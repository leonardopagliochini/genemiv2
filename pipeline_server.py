#!/usr/bin/env python3
"""
Lightweight web server exposing the pipeline workflow through a HTML/JS GUI.

It provides:
  - POST /api/run to start a new pipeline execution with user parameters.
  - GET  /api/stream/<run_id> to stream step-by-step logs and progress.
  - Static assets served from ./web for the browser interface.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import re
import signal
import subprocess
import threading
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    request,
    send_file,
    send_from_directory,
    stream_with_context,
)

from case_config import (
    DEFAULT_CASE_KEY,
    get_case_config,
    get_case_key,
    list_cases,
)
from paths import REPO_ROOT, extracted_dir, surfaces_dir
from time_units import months_to_years, years_to_months

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
    params: dict[str, Any] = field(default_factory=dict)
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    target_time: Optional[float] = None
    delta_t: Optional[float] = None
    cancelled: bool = False


runs: Dict[str, RunState] = {}
run_metadata: Dict[str, dict[str, Any]] = {}

app = Flask(
    __name__,
    static_folder=STATIC_DIR.as_posix(),
    static_url_path="",
)


SIM_PROGRESS_PATTERN = re.compile(r"^n\s*=\s*(?P<step>\d+),\s*t\s*=\s*(?P<time>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def format_years_label(years: float) -> str:
    formatted = f"{years:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


def normalise_extract_years(value: Any) -> tuple[list[float], list[int]]:
    if value in (None, "", []):
        raise ValueError("Please provide at least one year value.")

    tokens: list[float] = []

    def extend_from_token(token_value: Any) -> None:
        if token_value in (None, ""):
            return
        if isinstance(token_value, (int, float)):
            tokens.append(float(token_value))
            return
        if isinstance(token_value, str):
            chunks = token_value.replace(",", " ").split()
            if not chunks:
                return
            for chunk in chunks:
                if not chunk:
                    continue
                try:
                    tokens.append(float(chunk))
                except ValueError as exc:
                    raise ValueError(f"Invalid year value '{chunk}'.") from exc
            return
        raise ValueError(f"Unsupported value '{token_value}' in extractYears.")

    if isinstance(value, (list, tuple)):
        for item in value:
            extend_from_token(item)
    else:
        extend_from_token(value)

    if not tokens:
        raise ValueError("Please provide at least one year value.")

    seen_months: set[int] = set()
    years_list: list[float] = []
    months_list: list[int] = []

    for numeric in tokens:
        try:
            months = years_to_months(numeric)
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        if months in seen_months:
            continue
        seen_months.add(months)
        years_list.append(months_to_years(months))
        months_list.append(months)

    if not years_list:
        raise ValueError("Please provide at least one valid year value.")

    return years_list, months_list


def publish_event(run_id: str, payload: dict) -> None:
    state = runs.get(run_id)
    if state is None:
        return
    state.queue.put(payload)


def build_pipeline_command(params: dict) -> list[str]:
    cmd = ["python3", str(PIPELINE_SCRIPT)]
    case_key = params.get("case")
    if case_key:
        cmd.extend(["--case", case_key])
    extract_years = params.get("extract_years") or []
    for years in extract_years:
        cmd.extend(["--extract-years", format_years_label(float(years))])
    if params.get("simulation_years") is not None:
        cmd.extend(["--simulation-years", str(params["simulation_years"])])
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
        popen_kwargs: dict[str, Any] = {
            "cwd": REPO_ROOT,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if os.name != "nt":
            popen_kwargs["preexec_fn"] = os.setsid  # Run in a new process group for clean termination.
        else:
            creation_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if creation_flag:
                popen_kwargs["creationflags"] = creation_flag

        process = subprocess.Popen(cmd, **popen_kwargs)
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

            progress_match = SIM_PROGRESS_PATTERN.match(line)
            if progress_match:
                try:
                    step = int(progress_match.group("step"))
                    current_time = float(progress_match.group("time"))
                except (TypeError, ValueError):
                    step = None
                    current_time = None

                if step is not None:
                    state.current_step = step
                fraction = None
                if current_time is not None and current_time > 0:
                    if state.delta_t is None:
                        state.delta_t = current_time
                    if state.target_time is None:
                        target_candidate = state.params.get("simulation_years")
                        if target_candidate is None:
                            extract_list = state.params.get("extract_years") or []
                            if extract_list:
                                target_candidate = max(extract_list)
                        if target_candidate is not None:
                            with contextlib.suppress(TypeError, ValueError):
                                state.target_time = float(target_candidate)
                    if state.total_steps is None and state.target_time:
                        with contextlib.suppress(ZeroDivisionError):
                            estimated = state.target_time / current_time
                            if estimated:
                                state.total_steps = max(step or 0, int(round(estimated)))
                    if state.target_time and state.target_time > 0:
                        fraction = current_time / state.target_time
                    elif state.total_steps:
                        fraction = (step or 0) / state.total_steps

                publish_event(
                    run_id,
                    {
                        "type": "progress",
                        "stage": "Running simulation",
                        "currentStep": state.current_step,
                        "totalSteps": state.total_steps,
                        "currentTime": current_time,
                        "targetTime": state.target_time,
                        "fraction": max(0.0, min(fraction, 1.0)) if fraction is not None else None,
                    },
                )
                continue

            if line.startswith("[step]"):
                description = line[len("[step]") :].strip()
                if description in stage_index:
                    next_idx = stage_index[description]
                    if current_stage_idx is None or next_idx != current_stage_idx:
                        complete_previous()
                    current_stage_idx = next_idx
                    set_stage(description, "running")
                continue

            if line.startswith("[skip]"):
                content = line[len("[skip]") :].strip()
                matched_stage: Optional[str] = None
                for name in STAGE_NAMES:
                    if content.startswith(name):
                        matched_stage = name
                        break

                if matched_stage is None:
                    # Backwards compatibility: treat generic skip as simulation skip.
                    matched_stage = "Running simulation"

                if stages_status.get(matched_stage) in {"pending", "running"}:
                    current_stage_idx = stage_index[matched_stage]
                    status = "completed" if matched_stage in STAGE_NAMES[:2] else "skipped"
                    set_stage(matched_stage, status)
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
                    set_stage(name, "cancelled" if state.cancelled else "error")

        publish_event(run_id, {"type": "done", "returncode": process.returncode})
        state.done = True


@app.route("/")
def index() -> Response:
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/cases")
def list_cases_api() -> Response:
    cases_payload = [
        {
            "key": cfg.key,
            "label": cfg.label,
            "defaultMesh": cfg.default_mesh,
            "defaultProcs": cfg.default_procs,
        }
        for cfg in list_cases()
    ]
    return jsonify({"cases": cases_payload, "default": DEFAULT_CASE_KEY})


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

    def parse_float(name: str, required: bool = False) -> Optional[float]:
        value = payload.get(name)
        if value in (None, ""):
            if required:
                errors[name] = "This field is required."
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            errors[name] = "Must be a number."
            return None

    extract_years_raw = payload.get("extractYears")
    try:
        extract_years_list, extract_timesteps = normalise_extract_years(extract_years_raw)
    except ValueError as exc:
        extract_years_list = []
        extract_timesteps = []
        errors["extractYears"] = str(exc)

    raw_case = payload.get("case") or DEFAULT_CASE_KEY
    try:
        case_key = get_case_key(raw_case)
    except ValueError as exc:
        errors["case"] = str(exc)
        case_key = DEFAULT_CASE_KEY
    case_config = get_case_config(case_key)

    simulation_years = parse_float("simulationYears")
    if simulation_years is not None and simulation_years < 0:
        errors["simulationYears"] = "Must be >= 0."

    procs = parse_int("procs")
    if procs is not None and procs <= 0:
        errors["procs"] = "Must be > 0."

    mesh = payload.get("mesh") or case_config.default_mesh
    mesh_path = Path(mesh)
    if not mesh_path.is_absolute():
        mesh_path = REPO_ROOT / mesh_path
    if not mesh_path.exists():
        errors["mesh"] = f"Mesh file '{mesh_path}' not found."

    if errors:
        return jsonify({"errors": errors}), 400

    if not extract_timesteps:
        return jsonify({"errors": {"extractYears": "Please provide at least one valid extraction time in years."}}), 400

    run_id = uuid.uuid4().hex
    params = {
        "case": case_key,
        "extract_years": extract_years_list,
        "extract_timesteps": extract_timesteps,
        "simulation_years": simulation_years,
        "mesh": str(mesh_path),
        "procs": procs if procs is not None else case_config.default_procs,
        "output_dir": str(case_config.sim_output_dir()),
    }

    target_time = None
    if simulation_years is not None:
        with contextlib.suppress(TypeError, ValueError):
            target_time = float(simulation_years)
    elif extract_years_list:
        target_time = max(extract_years_list)

    extracted_paths = [extracted_dir(timestep, case_key) for timestep in extract_timesteps]
    surfaces_paths = [surfaces_dir(timestep, case_key) for timestep in extract_timesteps]

    state = RunState(params=params, target_time=target_time)
    runs[run_id] = state
    run_metadata[run_id] = {
        "case": case_key,
        "params": params,
        "target_time": target_time,
        "surfaces_dirs": [str(path) for path in surfaces_paths],
        "simulation_output_dir": str(case_config.sim_output_dir()),
        "extracted_dirs": [str(path) for path in extracted_paths],
        "extract_years": extract_years_list,
        "extract_timesteps": extract_timesteps,
    }

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


def resolve_run_metadata(run_id: str) -> dict[str, Any]:
    metadata = run_metadata.get(run_id)
    if metadata is None:
        abort(404)
    return metadata


def secure_child_path(base: Path, candidate: Path) -> Path:
    base_resolved = base.resolve(strict=False)
    candidate_resolved = candidate.resolve(strict=False)
    if base_resolved not in candidate_resolved.parents and candidate_resolved != base_resolved:
        abort(400)
    return candidate_resolved


@app.get("/api/run/<run_id>/outputs")
def list_outputs(run_id: str) -> Response:
    metadata = resolve_run_metadata(run_id)
    surfaces_dirs = [Path(path) for path in metadata.get("surfaces_dirs", [])]
    extract_years = metadata.get("extract_years") or []
    extract_timesteps = metadata.get("extract_timesteps") or []
    files = []
    for idx, surfaces_dir in enumerate(surfaces_dirs):
        if not surfaces_dir.exists():
            continue
        year_value = extract_years[idx] if idx < len(extract_years) else None
        timestep_value = extract_timesteps[idx] if idx < len(extract_timesteps) else None
        for path in sorted(surfaces_dir.glob("*.stl")):
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            entry: dict[str, Any] = {
                "name": path.name,
                "size": size,
                "url": f"/api/run/{run_id}/download/{path.name}",
            }
            if year_value is not None:
                entry["year"] = year_value
            if timestep_value is not None:
                entry["timestep"] = timestep_value
            files.append(entry)

    return jsonify({
        "files": files,
        "downloadAllUrl": f"/api/run/{run_id}/download-all" if files else None,
    })


@app.get("/api/run/<run_id>/download/<path:filename>")
def download_single(run_id: str, filename: str) -> Response:
    metadata = resolve_run_metadata(run_id)
    surfaces_dirs = [Path(path) for path in metadata.get("surfaces_dirs", [])]
    for surfaces_dir in surfaces_dirs:
        if not surfaces_dir.exists():
            continue
        requested = secure_child_path(surfaces_dir, surfaces_dir / filename)
        if requested.exists() and requested.is_file():
            return send_from_directory(surfaces_dir, requested.name, as_attachment=True)
    abort(404)


@app.get("/api/run/<run_id>/download-all")
def download_all(run_id: str) -> Response:
    metadata = resolve_run_metadata(run_id)
    surfaces_dirs = [Path(path) for path in metadata.get("surfaces_dirs", [])]

    files: list[Path] = []
    for surfaces_dir in surfaces_dirs:
        if surfaces_dir.exists():
            files.extend(path for path in sorted(surfaces_dir.glob("*.stl")) if path.is_file())
    if not files:
        abort(404)

    case_key = metadata.get("case")
    try:
        case_config = get_case_config(case_key)
        case_part = case_config.key
    except ValueError:
        case_part = (case_key or DEFAULT_CASE_KEY) or "case"
    year_values = metadata.get("extract_years") or []
    if year_values:
        years_part = "-".join(format_years_label(float(year)) for year in year_values)
        base_name = f"{case_part}_{years_part}"
    else:
        base_name = case_part
    zip_name = f"{base_name}_{run_id[:8]}_outputs.zip"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            try:
                zf.write(path, arcname=path.name)
            except OSError:
                continue
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=zip_name,
    )


@app.post("/api/run/<run_id>/cancel")
def cancel_run(run_id: str) -> Response:
    state = runs.get(run_id)
    if state is None:
        abort(404)

    with state.lock:
        process = state.process
        if process is None:
            if state.done:
                return jsonify({"status": "finished"})
            state.cancelled = True
            publish_event(run_id, {"type": "cancelled"})
            return jsonify({"status": "pending"})

        if process.poll() is not None:
            return jsonify({"status": "finished"})

        if not state.cancelled:
            state.cancelled = True
            publish_event(run_id, {"type": "cancelled"})

        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                except (ValueError, AttributeError):
                    process.terminate()
                else:
                    process.wait(timeout=2)
                    if process.poll() is None:
                        process.terminate()
        except ProcessLookupError:
            pass

    return jsonify({"status": "terminating"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)