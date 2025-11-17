#!/usr/bin/env python3
"""Generate a tetrahedral breast mesh from segmented surface shells."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import SVMTK as svmtk

DEFAULT_SURFACE_DIR = Path(__file__).resolve().parent / "segmentation_surfaces"


def load_surface(surface_path: Path) -> svmtk.Surface:
    if not surface_path.exists():
        raise FileNotFoundError(f"Surface file '{surface_path}' not found.")
    return svmtk.Surface(str(surface_path))


def build_domain(skin_surface: Path, fat_surface: Path) -> svmtk.Domain:
    skin = load_surface(skin_surface)
    fat = load_surface(fat_surface)

    surfaces = [skin, fat]
    subdomain_map = svmtk.SubdomainMap()
    subdomain_map.add("10", 1)
    subdomain_map.add("11", 2)
    return svmtk.Domain(surfaces, subdomain_map)


def create_mesh(resolution: int, output_path: Path, skin_surface: Path, fat_surface: Path) -> Path:
    if resolution <= 0:
        raise ValueError("Resolution parameter must be positive.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    domain = build_domain(skin_surface, fat_surface)

    print(f"[SVMTK] Creating mesh (N={resolution})...")
    start = time.time()
    domain.create_mesh(resolution)
    elapsed = time.time() - start

    domain.save(str(output_path))
    print(f"[SVMTK] Saved {output_path} in {elapsed:.2f} s.")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tetrahedral breast mesh from segmented skin/fat STL surfaces."
    )
    parser.add_argument(
        "-N",
        "--resolution",
        type=int,
        default=128,
        help="Meshing resolution passed to SVMTK.Domain.create_mesh (default: 128).",
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
        help="Filename of the fat STL surface (default: fat.stl).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("breast_128.mesh"),
        help="Destination MEDIT mesh path (default: breast_128.mesh).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    surfaces_dir = args.surfaces_dir.resolve()
    skin_path = surfaces_dir / args.skin_surface
    fat_path = surfaces_dir / args.fat_surface
    output_path = Path(args.output).resolve()

    create_mesh(
        resolution=args.resolution,
        output_path=output_path,
        skin_surface=skin_path,
        fat_surface=fat_path,
    )


if __name__ == "__main__":
    main()
