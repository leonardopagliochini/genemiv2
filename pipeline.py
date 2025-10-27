#!/usr/bin/env python3
"""
High-level driver for the Fisher-Kolmogorov workflow.

The script runs (in order):
  1. Project compilation via compile.py.
  2. The MPI simulation binary.
  3. Mesh extraction for a selected timestep.
  4. Surface extraction for the filtered meshes (if pvpython is available).

You can customise the main parameters through the CLI; see --help for details.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile, run the simulation, extract meshes, and export surfaces."
    )
    parser.add_argument(
        "--simulation-time",
        type=float,
        default=None,
        help=(
            "Final simulation time passed to the solver. "
            "Defaults to the value provided for --extract-time."
        ),
    )
    parser.add_argument(
        "--extract-time",
        type=int,
        required=True,
        help=(
            "Timestep index to post-process (e.g. 108 selects files named output_108*.vtu)."
        ),
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="mesh/MNI_with_phys.msh",
        help="Mesh file to feed to the solver.",
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=4,
        help="Number of MPI processes to launch with mpirun.",
    )
    return parser.parse_args()


def run_command(label: str, cmd: list[str], cwd: Path) -> None:
    print(f"[step] {label}")
    print(f"       {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_binary(build_dir: Path) -> Path:
    binary = build_dir / "main"
    if not binary.exists():
        raise FileNotFoundError(
            f"Binary '{binary}' not found. Ensure the project compiles correctly."
        )
    return binary


def resolve_mesh(mesh_arg: str, repo_root: Path) -> Path:
    mesh_path = Path(mesh_arg)
    if not mesh_path.is_absolute():
        mesh_path = (repo_root / mesh_path).resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file '{mesh_path}' not found.")
    return mesh_path


def timestep_outputs_exist(output_dir: Path, timestep: int) -> bool:
    if not output_dir.exists():
        return False
    prefix = f"output_{timestep:03d}"
    for suffix in (".vtu", ".pvtu"):
        if any(output_dir.glob(f"{prefix}*{suffix}")):
            return True
    return False


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    build_dir = repo_root / "build"
    build_dir.mkdir(exist_ok=True)

    mesh_path = resolve_mesh(args.mesh, repo_root)
    sim_time = args.simulation_time if args.simulation_time is not None else args.extract_time
    if sim_time <= 0:
        raise ValueError("Simulation time must be positive.")

    # Step 1: compile the project.
    run_command("Compiling sources", ["python3", "compile.py"], repo_root)

    # Step 2: run the simulation.
    binary = ensure_binary(build_dir)
    output_dir = build_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        raise FileNotFoundError("Neither 'mpirun' nor 'mpiexec' was found in PATH.")

    if timestep_outputs_exist(output_dir, args.extract_time):
        print(
            f"[skip] Found existing output for timestep {args.extract_time:03d}; "
            "skipping simulation."
        )
    else:
        run_command(
            "Running simulation",
            [
                mpirun,
                "-np",
                str(args.procs),
                str(binary),
                "--mesh",
                str(mesh_path),
                "--T",
                str(sim_time),
                "--output",
                str(output_dir),
            ],
            repo_root,
        )

    # Step 3: extract filtered meshes for the requested timestep.
    extracted_dir = build_dir / f"extracted_{args.extract_time:03d}"
    run_command(
        "Extracting mesh subsets",
        [
            "python3",
            "extract_mesh.py",
            "--years",
            str(args.extract_time),
            "--output-dir",
            str(extracted_dir),
        ],
        repo_root,
    )

    # Step 4: extract surfaces using ParaView's pvpython, if available.
    pvpython = shutil.which("pvpython")
    if pvpython is None:
        print(
            "[warn] 'pvpython' not found in PATH. Skipping surface extraction step."
        )
        return 0

    run_command(
        "Extracting surfaces",
        [
            pvpython,
            "extract_surfaces.py",
            "--input",
            str(extracted_dir),
        ],
        repo_root,
    )

    print("[done] Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"[error] Step failed with exit code {exc.returncode}")
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:
        print(f"[error] {exc}")
        raise SystemExit(1) from exc
