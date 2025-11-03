#!/usr/bin/env python3
"""Extract per-material submeshes from VTU output and export STL surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import meshio
import numpy as np

# material_names (id -> name)
MATERIALS: Dict[int, str] = {
    1: "skin",
    2: "fat",
    3: "ductal",
    4: "lobulus",
    5: "stroma",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a parallel VTU output into per-material VTU and STL files for a "
            "single time step."
        )
    )
    parser.add_argument(
        "time",
        help="Time identifier (e.g. 0, 6, 012). This selects files named output_XXX.*.vtu.",
    )
    parser.add_argument(
        "--input-dir",
        default="build/output",
        help="Directory containing the VTU/PVTU files (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Destination directory for extracted meshes. "
            "Defaults to <input-dir>/materials."
        ),
    )
    parser.add_argument(
        "--materials",
        nargs="*",
        type=int,
        default=sorted(MATERIALS.keys()),
        help=(
            "Optional subset of material ids to export. "
            "Defaults to all known ids."
        ),
    )
    return parser.parse_args()


def normalize_time_code(raw: str) -> str:
    return f"{int(raw):03d}" if raw.isdigit() else raw


def find_piece_files(input_dir: Path, time_code: str) -> List[Path]:
    pieces = sorted(
        path
        for path in input_dir.glob(f"output_{time_code}.*.vtu")
        if path.suffix == ".vtu"
    )
    if not pieces:
        raise FileNotFoundError(
            f"No VTU pieces matching 'output_{time_code}.*.vtu' found in {input_dir}"
        )
    return pieces


def ensure_material_ids(requested_ids: Iterable[int]) -> List[int]:
    unknown = sorted(set(requested_ids).difference(MATERIALS.keys()))
    if unknown:
        raise ValueError(f"Unknown material ids requested: {unknown}")
    return sorted(set(requested_ids))


def extract_material_cells(
    mesh: meshio.Mesh, material_id: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]] | None:
    if "material_id" not in mesh.point_data:
        raise KeyError("Point data 'material_id' not found in VTU piece.")

    tetra_cells = mesh.cells_dict.get("tetra")
    if tetra_cells is None or tetra_cells.size == 0:
        return None

    material_values = mesh.point_data["material_id"]
    cell_mask = np.all(material_values[tetra_cells] == material_id, axis=1)
    if not np.any(cell_mask):
        return None

    selected_cells = tetra_cells[cell_mask]
    unique_point_ids, inverse_indices = np.unique(
        selected_cells, return_inverse=True
    )
    remapped_cells = inverse_indices.reshape(selected_cells.shape)
    selected_points = mesh.points[unique_point_ids]

    point_data_subset = {
        name: data[unique_point_ids] for name, data in mesh.point_data.items()
    }
    # Preserve material ids as integers where possible.
    if "material_id" in point_data_subset:
        point_data_subset["material_id"] = point_data_subset["material_id"].astype(
            np.int32
        )

    return selected_points, remapped_cells, point_data_subset


def add_piece_to_material(
    mesh: meshio.Mesh,
    material_id: int,
    aggregators: Dict[int, Dict[str, object]],
) -> None:
    subset = extract_material_cells(mesh, material_id)
    if subset is None:
        return

    points, cells, point_data = subset
    material_store = aggregators[material_id]

    if not material_store["point_data"]:
        material_store["point_data"] = {name: [] for name in point_data.keys()}

    point_offset = material_store["point_offset"]
    material_store["points"].append(points)
    material_store["cells"].append(cells + point_offset)

    for name, chunk in point_data.items():
        material_store["point_data"][name].append(chunk)

    material_store["point_offset"] += points.shape[0]


def concatenate_chunks(chunks: List[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty((0,), dtype=float)
    first = chunks[0]
    if first.ndim == 1:
        return np.concatenate(chunks, axis=0)
    return np.concatenate(chunks, axis=0)


def build_material_mesh(
    material_store: Dict[str, object]
) -> meshio.Mesh | None:
    if material_store["point_offset"] == 0:
        return None

    points = concatenate_chunks(material_store["points"])
    cells = concatenate_chunks(material_store["cells"])
    point_data = {
        name: concatenate_chunks(chunks)
        for name, chunks in material_store["point_data"].items()
    }
    return meshio.Mesh(
        points=points,
        cells=[("tetra", cells.astype(np.int64, copy=False))],
        point_data=point_data,
    )


def extract_surface_triangles(cells: np.ndarray) -> np.ndarray:
    face_map: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    for cell in cells:
        i0, i1, i2, i3 = cell
        faces = [
            (i0, i1, i2),
            (i0, i1, i3),
            (i0, i2, i3),
            (i1, i2, i3),
        ]
        for face in faces:
            key = tuple(sorted(face))
            if key in face_map:
                del face_map[key]
            else:
                face_map[key] = face
    if not face_map:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(list(face_map.values()), dtype=np.int64)


def write_outputs(
    material_id: int,
    mesh: meshio.Mesh,
    base_output_dir: Path,
    time_code: str,
) -> None:
    material_name = MATERIALS[material_id]
    prefix = f"{time_code}_{material_id}_{material_name}"

    vtu_path = base_output_dir / f"{prefix}.vtu"
    meshio.write(vtu_path, mesh)

    surface_faces = extract_surface_triangles(mesh.cells_dict["tetra"])
    if surface_faces.size == 0:
        print(f"[{material_name}] No boundary faces detected; STL skipped.")
        return

    stl_mesh = meshio.Mesh(points=mesh.points, cells=[("triangle", surface_faces)])
    stl_path = base_output_dir / f"{prefix}.stl"
    meshio.write(stl_path, stl_mesh)


def main() -> None:
    args = parse_args()
    time_code = normalize_time_code(args.time)

    input_dir = Path(args.input_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir / "materials"
    )
    output_dir = output_dir / f"time_{time_code}"
    output_dir.mkdir(parents=True, exist_ok=True)

    material_ids = ensure_material_ids(args.materials)
    piece_files = find_piece_files(input_dir, time_code)

    print(f"Processing time step {time_code} with {len(piece_files)} pieces...")

    aggregators: Dict[int, Dict[str, object]] = {
        material_id: {
            "points": [],
            "cells": [],
            "point_data": {},
            "point_offset": 0,
        }
        for material_id in material_ids
    }

    for piece_path in piece_files:
        mesh = meshio.read(piece_path)
        for material_id in material_ids:
            add_piece_to_material(mesh, material_id, aggregators)

    for material_id in material_ids:
        material_mesh = build_material_mesh(aggregators[material_id])
        material_name = MATERIALS[material_id]
        if material_mesh is None:
            print(f"[{material_name}] No elements found in time step {time_code}.")
            continue

        write_outputs(material_id, material_mesh, output_dir, time_code)
        print(
            f"[{material_name}] Wrote VTU and STL to {output_dir} "
            f"(cells={material_mesh.cells_dict['tetra'].shape[0]})."
        )


if __name__ == "__main__":
    main()
