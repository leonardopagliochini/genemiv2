#!/usr/bin/env python3
"""
Secondary pipeline that post-processes existing simulation outputs.

The script discovers all timesteps that have VTU files inside the selected
output directory, then runs:
  1. extract_mesh.py for each timestep to generate filtered meshes.
  2. extract_surfaces.py (via pvpython) to export STL surfaces, if available.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from APP.utils.paths import REPO_ROOT, SIM_OUTPUT_DIR, extracted_dir, surfaces_dir
from APP.utils.time_units import months_to_years, years_to_months

VTU_TIMESTEP_PATTERN = re.compile(r"output_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run mesh and surface extraction for every timestep that has VTU files in the input directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SIM_OUTPUT_DIR,
        help="Directory containing the raw VTU outputs (default: build/output).",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit list of timesteps (in months) to process (default: auto-detect).",
    )
    selector.add_argument(
        "--years",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit list of simulation times in years to process (default: auto-detect).",
    )
    parser.add_argument(
        "--field",
        type=str,
        default=None,
        help="Optional override for the field passed to extract_mesh.py.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for the threshold passed to extract_mesh.py.",
    )
    parser.add_argument(
        "--skip-surfaces",
        action="store_true",
        help="Skip the surface extraction stage even if pvpython is available.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip steps that already produced VTU/STL files for a timestep.",
    )
    parser.add_argument(
        "--pvpython",
        type=str,
        default=None,
        help="Path to the pvpython binary (default: detected via PATH).",
    )
    return parser.parse_args()


def run_command(label: str, cmd: Sequence[str], cwd: Path) -> None:
    print(f"[step] {label}")
    print(f"       {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def discover_timesteps(input_dir: Path) -> list[int]:
    timesteps: set[int] = set()
    for path in input_dir.glob("output_*.vtu"):
        match = VTU_TIMESTEP_PATTERN.search(path.name)
        if match:
            timesteps.add(int(match.group(1)))
    return sorted(timesteps)


def has_generated_files(directory: Path, pattern: str) -> bool:
    if not directory.exists():
        return False
    return any(directory.glob(pattern))


def format_timesteps(timesteps: Iterable[int]) -> str:
    return ", ".join(f"{months_to_years(timestep):g}y (#{timestep:03d})" for timestep in timesteps)


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT

    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = (repo_root / input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' not found.")

    if args.timesteps:
        invalid = [value for value in args.timesteps if value < 0]
        if invalid:
            raise ValueError(f"Timestep values must be non-negative: {invalid}")
        timesteps = sorted(set(args.timesteps))
    elif args.years:
        converted: set[int] = set()
        for value in args.years:
            try:
                timestep = years_to_months(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid --years entry '{value}': {exc}") from exc
            converted.add(timestep)
        timesteps = sorted(converted)
    else:
        timesteps = discover_timesteps(input_dir)
        if not timesteps:
            raise FileNotFoundError(
                f"No VTU files matching 'output_*.vtu' found inside {input_dir}."
            )

    print(f"[info] Processing timesteps: {format_timesteps(timesteps)}")

    pvpython_bin = None
    if not args.skip_surfaces:
        pvpython_bin = args.pvpython or shutil.which("pvpython")
        if pvpython_bin is None:
            print("[warn] 'pvpython' not found. Surface extraction will be skipped.")

    for timestep in timesteps:
        years_value = months_to_years(timestep)
        print(f"[info] Time {years_value:g} years (timestep {timestep:03d})")
        extracted_path = extracted_dir(timestep)
        mesh_already_done = has_generated_files(extracted_path, "output_*.vtu")

        if mesh_already_done and args.only_missing:
            print("[skip] Mesh extraction already present.")
        else:
            mesh_cmd = [
                "python3",
                str(repo_root / "post_process" / "extract_mesh.py"),
                "--timestep",
                str(timestep),
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(extracted_path),
            ]
            if args.field:
                mesh_cmd.extend(["--field", args.field])
            if args.threshold is not None:
                mesh_cmd.extend(["--threshold", str(args.threshold)])
            run_command("Extracting mesh subsets", mesh_cmd, repo_root)

        if pvpython_bin is None:
            continue

        surface_target = surfaces_dir(timestep)
        surfaces_done = has_generated_files(surface_target, "*.stl")
        if surfaces_done and args.only_missing:
            print("[skip] Surface extraction already present.")
            continue

        surface_cmd = [
            pvpython_bin,
            str(repo_root / "post_process" / "extract_surfaces.py"),
            "--input",
            str(extracted_path),
        ]
        run_command("Extracting surfaces", surface_cmd, repo_root)

    print("[done] Post-processing completed successfully.")
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
