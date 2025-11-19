#!/usr/bin/env python3
"""One-click pipeline to generate the full augmented breast mesh."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
UTILS_DIR = THIS_DIR / "utils"
DEFAULT_SURFACE_DIR = UTILS_DIR / "segmentation_surfaces"


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the full breast mesh by running create_mesh_refinements.py and "
            "create_internal_volumes.py in sequence. "
            "Any extra arguments provided will be forwarded to create_internal_volumes.py."
        )
    )
    parser.add_argument(
        "-N",
        "--resolution",
        type=int,
        default=128,
        help="Resolution parameter passed to the SVMTK mesher (default: 128).",
    )
    parser.add_argument(
        "--surfaces-dir",
        type=Path,
        default=DEFAULT_SURFACE_DIR,
        help=f"Directory containing the segmentation STL files (default: {DEFAULT_SURFACE_DIR}).",
    )
    parser.add_argument(
        "--skin-surface",
        default="skin.stl",
        help="Filename of the outer skin STL surface (default: skin.stl).",
    )
    parser.add_argument(
        "--fat-surface",
        default="fat.stl",
        help="Filename of the inner fat STL surface (default: fat.stl).",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=THIS_DIR,
        help="Directory where intermediate and final meshes are stored (default: mesh/breast).",
    )
    parser.add_argument(
        "--base-mesh",
        type=Path,
        default=None,
        help="Optional override for the raw SVMTK mesh output path.",
    )
    parser.add_argument(
        "--final-mesh",
        type=Path,
        default=None,
        help="Optional override for the augmented mesh output path.",
    )
    parser.add_argument(
        "--lobule-csv",
        type=Path,
        default=None,
        help="Optional destination for the lobule coordinate CSV (default follows create_internal_volumes).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser.parse_known_args(argv)


def build_command_summary(cmd: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(token)) for token in cmd)


def run_step(label: str, cmd: list[str | Path], dry_run: bool) -> None:
    summary = build_command_summary(cmd)
    print(f"[run] {label}\n       {summary}")
    if dry_run:
        return
    subprocess.run([str(token) for token in cmd], check=True)


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    args, extra_internal_args = parse_args(argv)

    working_dir = args.working_dir.resolve()
    working_dir.mkdir(parents=True, exist_ok=True)

    surfaces_dir = args.surfaces_dir.resolve()
    skin_surface = surfaces_dir / args.skin_surface
    fat_surface = surfaces_dir / args.fat_surface
    for surface in (skin_surface, fat_surface):
        if not surface.exists():
            raise FileNotFoundError(f"Required surface '{surface}' does not exist.")

    base_mesh = (
        Path(args.base_mesh).resolve()
        if args.base_mesh is not None
        else working_dir / f"breast_{args.resolution}.mesh"
    )
    final_mesh = (
        Path(args.final_mesh).resolve()
        if args.final_mesh is not None
        else working_dir / f"breast_{args.resolution}_augmented.msh"
    )
    lobule_csv = Path(args.lobule_csv).resolve() if args.lobule_csv is not None else None

    base_mesh = ensure_parent(base_mesh)
    final_mesh = ensure_parent(final_mesh)
    if lobule_csv is not None:
        ensure_parent(lobule_csv)

    mesh_ref_script = UTILS_DIR / "create_mesh_refinements.py"
    if not mesh_ref_script.exists():
        raise FileNotFoundError(f"Mesh refinement script not found at {mesh_ref_script}.")
    internal_script = UTILS_DIR / "create_internal_volumes.py"
    if not internal_script.exists():
        raise FileNotFoundError(f"Internal volume script not found at {internal_script}.")

    mesh_cmd: list[str | Path] = [
        sys.executable,
        mesh_ref_script,
        "--resolution",
        str(args.resolution),
        "--surfaces-dir",
        surfaces_dir,
        "--skin-surface",
        args.skin_surface,
        "--fat-surface",
        args.fat_surface,
        "--output",
        base_mesh,
    ]
    run_step("SVMTK meshing", mesh_cmd, args.dry_run)

    internal_cmd: list[str | Path] = [
        sys.executable,
        internal_script,
        "--input",
        base_mesh,
        "--output",
        final_mesh,
    ]
    if lobule_csv is not None:
        internal_cmd += ["--lobule-coords-output", lobule_csv]
    internal_cmd.extend(extra_internal_args)
    run_step("Internal volume augmentation", internal_cmd, args.dry_run)

    print(f"[done] Base mesh:  {base_mesh}")
    print(f"[done] Final mesh: {final_mesh}")
    if lobule_csv is not None:
        print(f"[done] Lobule CSV: {lobule_csv}")
    elif extra_internal_args:
        # create_internal_volumes already echoes where it wrote the CSV.
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
