#!/usr/bin/env python3
"""
Generate multiple breast simulations with random tumour starting points.

This helper mirrors the standard pipeline workflow, but it:
  * draws tumour centres from the lobule vertex catalogue produced by
    create_internal_volumes.py;
  * launches the solver once per sampled point, overriding --starting;
  * stores intermediate and post-processed outputs in per-simulation folders.

All simulations reuse the same mesh, extraction schedule, and runtime parameters.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

from case_config import CaseConfig, get_case_config


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
        months = int(round(numeric * 12))
        if not np.isclose(months / 12.0, numeric, atol=1e-6):
            raise ValueError("Extraction times must align to monthly increments (multiples of 1/12 year).")
        if months in seen_months:
            continue
        seen_months.add(months)
        years_list.append(months / 12.0)
        months_list.append(months)

    return years_list, months_list


def format_years_label(years: float) -> str:
    formatted = f"{years:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


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


def load_lobule_points(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Lobule coordinates file '{csv_path}' not found. "
            "Generate it with create_internal_volumes.py --lobule-coords-output."
        )
    try:
        data = np.loadtxt(csv_path, delimiter=",")
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unable to read lobule coordinates from '{csv_path}': {exc}") from exc

    if data.size == 0:
        raise ValueError(f"Lobule coordinates file '{csv_path}' is empty.")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(
            f"Lobule coordinates file '{csv_path}' must contain at least three columns per row."
        )
    return data[:, :3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple breast simulations with random tumour starting points."
    )
    parser.add_argument(
        "--extract-years",
        dest="extract_years",
        type=str,
        action="append",
        required=True,
        help="Simulation time expressed in years to post-process (e.g. 9 or 9,10).",
    )
    parser.add_argument(
        "--simulation-years",
        dest="simulation_years",
        type=float,
        default=None,
        help="Final simulation time in years passed to the solver.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Mesh file to feed to the solver (defaults to the breast case configuration).",
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=None,
        help="Number of MPI processes to launch with mpirun.",
    )
    parser.add_argument(
        "--lobule-points",
        type=str,
        required=True,
        help="Path to the CSV file containing lobule vertex coordinates.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        required=True,
        help="Number of clinical scenarios to generate.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        required=True,
        help="Directory where per-simulation results will be stored.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run identifier used to namespace result directories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random number generator.",
    )
    return parser.parse_args()


def pick_random_points(points: np.ndarray, samples: int, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(points), size=samples, endpoint=False)
    return points[idx]


def main() -> int:
    args = parse_args()

    if args.num_simulations <= 0:
        raise ValueError("--num-simulations must be positive.")

    extract_years, extract_timesteps = parse_extract_years(args.extract_years)
    if not extract_years or not extract_timesteps:
        raise ValueError("No valid extraction times were provided.")

    case_config = get_case_config("breast")
    repo_root = Path(__file__).resolve().parent

    mesh_argument = args.mesh if args.mesh else case_config.default_mesh
    mesh_path = resolve_mesh(mesh_argument, repo_root)

    procs = args.procs if args.procs is not None else case_config.default_procs
    sim_years = (
        args.simulation_years if args.simulation_years is not None else max(extract_years)
    )
    if sim_years < 0:
        raise ValueError("Simulation time must be positive.")

    lobule_points = load_lobule_points(Path(args.lobule_points))
    samples = pick_random_points(lobule_points, args.num_simulations, args.seed)

    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] Run ID: {args.run_id}")
    print(f"[info] Using mesh: {mesh_path}")
    print(f"[info] Lobule coordinates: {Path(args.lobule_points).resolve()}")
    print(f"[info] Results root: {results_root}")
    print(f"[info] Requested extraction years: {', '.join(format_years_label(y) for y in extract_years)}")
    print(f"[info] Generating {args.num_simulations} simulations.")

    run_compile_stage(case_config, repo_root)

    binary = ensure_binary(case_config.binary_path())

    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        raise FileNotFoundError("Neither 'mpirun' nor 'mpiexec' was found in PATH.")

    pvpython = shutil.which("pvpython")
    if pvpython is None:
        print("[warn] 'pvpython' not found in PATH. Surface extraction will be skipped.")

    for index, sample_point in enumerate(samples, start=1):
        sim_label = f"simulation_{index:03d}"
        sim_base = results_root / sim_label
        sim_output_dir = sim_base / "output"
        sim_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] [{sim_label}] Selected tumour centre: ({sample_point[0]:.6f}, {sample_point[1]:.6f}, {sample_point[2]:.6f})")

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
                "--starting",
                f"{sample_point[0]}",
                f"{sample_point[1]}",
                f"{sample_point[2]}",
            ],
            repo_root,
        )

        for timestep, years in zip(extract_timesteps, extract_years):
            extracted_path = sim_base / f"extracted_{timestep:03d}"
            extracted_path.mkdir(parents=True, exist_ok=True)
            run_command(
                "Extracting mesh subsets",
                [
                    "python3",
                    "extract_mesh.py",
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

            if pvpython is not None:
                run_command(
                    "Extracting surfaces",
                    [
                        pvpython,
                        "extract_surfaces.py",
                        "--input",
                        str(extracted_path),
                    ],
                    repo_root,
                )

        print(f"[info] [{sim_label}] Completed.")

    if pvpython is None:
        print("[warn] Surfaces were skipped for all simulations because 'pvpython' is unavailable.")

    print("[done] Random breast pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
