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

from APP.utils.case_config import CaseConfig, DEFAULT_CASE_KEY, get_case_config, list_cases
from APP.utils.paths import REPO_ROOT
from APP.utils.time_units import months_to_years, years_to_months


CASE_CHOICES = tuple(cfg.key for cfg in list_cases())

def parse_extract_years(raw_values: list[str]) -> tuple[list[float], list[int]]:
    tokens: list[str] = []
    for raw in raw_values or []:
        if raw is None:
            continue
        chunks = str(raw).replace(",", " ").split()
        tokens.extend(chunk for chunk in chunks if chunk)

    if not tokens:
        raise ValueError("--extract-years is required.")

    seen_months: set[int] = set()
    years_list: list[float] = []
    months_list: list[int] = []

    for token in tokens:
        try:
            numeric = float(token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid --extract-years value '{token}'.") from exc
        try:
            months = years_to_months(numeric)
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        if months in seen_months:
            continue
        seen_months.add(months)
        years_list.append(months_to_years(months))
        months_list.append(months)

    return years_list, months_list


def format_years_label(years: float) -> str:
    formatted = f"{years:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile, run the simulation, extract meshes, and export surfaces."
    )
    parser.add_argument(
        "--case",
        choices=CASE_CHOICES,
        default=DEFAULT_CASE_KEY,
        help=f"Select the simulation case to run (default: {DEFAULT_CASE_KEY}).",
    )
    parser.add_argument(
        "--extract-years",
        dest="extract_years",
        type=str,
        action="append",
        help=(
            "Simulation time expressed in years to post-process "
            "(e.g. 9 or 9,10 selects timesteps corresponding to these years)."
        ),
        required=True,
    )
    parser.add_argument(
        "--extract-time",
        dest="extract_years",
        type=str,
        action="append",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--simulation-years",
        dest="simulation_years",
        type=float,
        default=None,
        help=(
            "Final simulation time, expressed in years, passed to the solver. "
            "Defaults to the maximum value provided for --extract-years."
        ),
    )
    parser.add_argument(
        "--simulation-time",
        dest="simulation_years",
        type=float,
        help=argparse.SUPPRESS,
    )
    extract_group = parser.add_argument_group("Simulation parameters")
    extract_group.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Mesh file to feed to the solver.",
    )
    extract_group.add_argument(
        "--procs",
        type=int,
        default=None,
        help="Number of MPI processes to launch with mpirun.",
    )
    args = parser.parse_args()
    try:
        (
            args.extract_years,
            args.extract_timesteps,
        ) = parse_extract_years(args.extract_years)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def run_command(label: str, cmd: list[str], cwd: Path) -> None:
    print(f"[step] {label}")
    subprocess.run(cmd, cwd=cwd, check=True)


def run_compile_stage(case_config: CaseConfig, cwd: Path) -> None:
    commands = [list(cmd) for cmd in case_config.compile_commands]
    if not commands:
        print("[skip] Compiling sources (no compile commands configured).")
        return

    first, *rest = commands
    run_command("Compiling sources", first, cwd)
    for command in rest:
        print(f"[info] ({case_config.key}) running compile helper: {' '.join(command)}")
        subprocess.run(command, cwd=cwd, check=True)


def ensure_binary(binary: Path) -> Path:
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
    repo_root = REPO_ROOT
    case_config = get_case_config(args.case)

    extract_years = args.extract_years
    extract_timesteps = args.extract_timesteps
    if not extract_years or not extract_timesteps:
        raise ValueError("No valid extraction times were provided.")

    extract_targets = list(zip(extract_timesteps, extract_years))
    max_extract_year = max(extract_years)

    mesh_argument = args.mesh if args.mesh else case_config.default_mesh
    mesh_path = resolve_mesh(mesh_argument, repo_root)

    procs = args.procs if args.procs is not None else case_config.default_procs
    sim_years = (
        args.simulation_years if args.simulation_years is not None else max_extract_year
    )
    if sim_years < 0:
        raise ValueError("Simulation time must be positive.")

    sim_output_dir = case_config.sim_output_dir()
    build_dir = case_config.build_dir
    binary_path = case_config.binary_path()

    outputs_available = all(
        timestep_outputs_exist(sim_output_dir, timestep)
        for timestep in extract_timesteps
    )

    sim_output_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    if outputs_available:
        targets_summary = ", ".join(
            f"{format_years_label(year)}y (timestep {timestep:03d})"
            for timestep, year in extract_targets
        )
        print(
            "[skip] Compiling sources (existing results found for the requested timesteps)."
        )
        print(f"[skip] Running simulation (found existing outputs for {targets_summary}).")
    else:
        if sim_years < max_extract_year:
            raise ValueError(
                f"Simulation time ({sim_years:g} years) must cover the requested extraction "
                f"times (maximum {max_extract_year:g} years)."
            )
        # Step 1: compile the project.
        run_compile_stage(case_config, repo_root)

        # Step 2: run the simulation.
        binary = ensure_binary(binary_path)

        mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
        if mpirun is None:
            raise FileNotFoundError("Neither 'mpirun' nor 'mpiexec' was found in PATH.")

        run_command(
            "Running simulation",
            [
                mpirun,
                "-np",
                str(procs),
                str(binary),
                "--mesh",
                str(mesh_path),
                "--T",
                str(sim_years),
                "--output",
                str(sim_output_dir),
            ],
            repo_root,
        )

    # Step 3: extract filtered meshes for the requested timesteps.
    for timestep, years in extract_targets:
        extracted_path = case_config.extracted_dir(timestep)
        run_command(
            "Extracting mesh subsets",
            [
                "python3",
                str(repo_root / "post_process" / "extract_mesh.py"),
                "--case",
                case_config.key,
                "--timestep",
                str(timestep),
                "--years",
                format_years_label(years),
                "--input-dir",
                str(sim_output_dir),
                "--output-dir",
                str(extracted_path),
            ],
            repo_root,
        )

    # Step 4: extract surfaces using ParaView's pvpython, if available.
    pvpython = shutil.which("pvpython")
    if pvpython is None:
        print("[warn] 'pvpython' not found in PATH. Skipping surface extraction step.")
        return 0

    for timestep, _years in extract_targets:
        extracted_path = case_config.extracted_dir(timestep)
        run_command(
            "Extracting surfaces",
            [
                pvpython,
                str(repo_root / "post_process" / "extract_surfaces.py"),
                "--input",
                str(extracted_path),
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
