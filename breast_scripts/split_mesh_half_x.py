#!/usr/bin/env python3
"""
split_mesh_half_x.py

Divide una mesh in formato Medit (.mesh) in due metà lungo il piano ortogonale
all'asse X passante per il punto medio del bounding box (o per una coordinata
specificata).

Il programma produce due nuovi file .mesh contenenti i soli vertici e gli
elementi (triangoli e tetraedri) che ricadono rispettivamente nella metà
sinistra e destra. Gli elementi che attraversano il piano vengono assegnati in
base alla posizione del baricentro.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Vertex:
    x: float
    y: float
    z: float
    tag: int


@dataclass
class Element:
    node_ids: list[int]
    tag: int


def find_section(lines: list[str], name: str):
    """Restituisce (header_idx, count, start_idx, end_idx) per la sezione."""
    target = name.lower()
    for idx, line in enumerate(lines):
        if line.strip().lower() == target:
            # Salta eventuali righe vuote fino al contatore
            j = idx + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                raise ValueError(f"Sezione '{name}' senza contatore.")
            count = int(lines[j].strip())
            data_start = j + 1
            data_end = data_start + count
            return idx, count, data_start, data_end
    raise ValueError(f"Sezione '{name}' non trovata nella mesh.")


def read_vertices(lines: list[str]) -> dict[int, Vertex]:
    _, count, start, end = find_section(lines, "Vertices")
    vertices: dict[int, Vertex] = {}
    current_id = 1
    for raw_line in lines[start:end]:
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) < 3:
            raise ValueError(f"Linea Vertices non valida: '{raw_line}'")
        x, y, z = map(float, parts[:3])
        tag = int(parts[3]) if len(parts) > 3 else 0
        vertices[current_id] = Vertex(x, y, z, tag)
        current_id += 1
    if current_id - 1 != count:
        raise ValueError("Numero di vertici letto incompatibile con il contatore.")
    return vertices


def read_elements(lines: list[str], name: str, nodes_per_element: int) -> list[Element]:
    try:
        _, count, start, end = find_section(lines, name)
    except ValueError:
        return []
    elements: list[Element] = []
    for raw_line in lines[start:end]:
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) < nodes_per_element + 1:
            raise ValueError(f"Linea {name} non valida: '{raw_line}'")
        node_ids = list(map(int, parts[:nodes_per_element]))
        tag = int(parts[nodes_per_element])
        elements.append(Element(node_ids=node_ids, tag=tag))
    if len(elements) != count:
        raise ValueError(f"Numero di elementi {name} letto incompatibile con il contatore.")
    return elements


def compute_plane(vertices: dict[int, Vertex], override: float | None) -> float:
    if override is not None:
        return override
    xs = [v.x for v in vertices.values()]
    if not xs:
        raise ValueError("Mesh priva di vertici; impossibile calcolare il piano di split.")
    return (min(xs) + max(xs)) * 0.5


def centroid_x(node_ids: Iterable[int], vertices: dict[int, Vertex]) -> float:
    values = [vertices[nid].x for nid in node_ids]
    return sum(values) / len(values)


def classify_element(
    element: Element,
    vertices: dict[int, Vertex],
    plane_x: float,
    tolerance: float,
) -> str:
    xs = [vertices[nid].x for nid in element.node_ids]
    min_x = min(xs)
    max_x = max(xs)
    if max_x < plane_x - tolerance:
        return "left"
    if min_x > plane_x + tolerance:
        return "right"
    return "left" if centroid_x(element.node_ids, vertices) <= plane_x else "right"


def build_subset(
    vertices: dict[int, Vertex],
    triangles: list[Element],
    tetrahedra: list[Element],
) -> tuple[list[Vertex], list[Element], list[Element], dict[int, int]]:
    used_vertex_ids: set[int] = set()
    for elem in triangles + tetrahedra:
        used_vertex_ids.update(elem.node_ids)

    ordered_ids = sorted(used_vertex_ids)
    mapping = {old_id: new_idx + 1 for new_idx, old_id in enumerate(ordered_ids)}

    new_vertices = [vertices[old_id] for old_id in ordered_ids]

    def remap(elements: list[Element]) -> list[Element]:
        remapped: list[Element] = []
        for elem in elements:
            remapped_ids = [mapping[nid] for nid in elem.node_ids]
            remapped.append(Element(remapped_ids, elem.tag))
        return remapped

    return new_vertices, remap(triangles), remap(tetrahedra), mapping


def write_mesh(
    path: Path,
    version: str,
    dimension: str,
    vertices: list[Vertex],
    triangles: list[Element],
    tetrahedra: list[Element],
):
    with path.open("w", encoding="utf8") as fh:
        fh.write(f"MeshVersionFormatted {version}\n\n")
        fh.write(f"Dimension {dimension}\n\n")

        fh.write("Vertices\n")
        fh.write(f"{len(vertices)}\n")
        for v in vertices:
            fh.write(f"{v.x:.16g} {v.y:.16g} {v.z:.16g} {v.tag}\n")
        fh.write("\n")

        if triangles:
            fh.write("Triangles\n")
            fh.write(f"{len(triangles)}\n")
            for tri in triangles:
                nodes = " ".join(str(nid) for nid in tri.node_ids)
                fh.write(f"{nodes} {tri.tag}\n")
            fh.write("\n")

        if tetrahedra:
            fh.write("Tetrahedra\n")
            fh.write(f"{len(tetrahedra)}\n")
            for tet in tetrahedra:
                nodes = " ".join(str(nid) for nid in tet.node_ids)
                fh.write(f"{nodes} {tet.tag}\n")
            fh.write("\n")

        fh.write("End\n")


def extract_header(lines: list[str]) -> tuple[str, str]:
    version = "1"
    dimension = "3"
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("meshversionformatted"):
            parts = stripped.split()
            if len(parts) >= 2:
                version = parts[-1]
        elif stripped.lower().startswith("dimension"):
            parts = stripped.split()
            if len(parts) >= 2:
                dimension = parts[-1]
        if stripped.lower() == "vertices":
            break
    return version, dimension


def split_mesh(
    input_path: Path,
    plane_x: float | None,
    left_path: Path,
    right_path: Path,
    tolerance: float,
) -> None:
    lines = input_path.read_text(encoding="utf8").splitlines()
    version, dimension = extract_header(lines)
    vertices = read_vertices(lines)
    triangles = read_elements(lines, "Triangles", 3)
    tetrahedra = read_elements(lines, "Tetrahedra", 4)

    if not tetrahedra and not triangles:
        raise ValueError("La mesh non contiene né Triangles né Tetrahedra da dividere.")

    split_plane = compute_plane(vertices, plane_x)

    left_triangles: list[Element] = []
    right_triangles: list[Element] = []
    for tri in triangles:
        side = classify_element(tri, vertices, split_plane, tolerance)
        if side == "left":
            left_triangles.append(tri)
        else:
            right_triangles.append(tri)

    left_tetra: list[Element] = []
    right_tetra: list[Element] = []
    for tet in tetrahedra:
        side = classify_element(tet, vertices, split_plane, tolerance)
        if side == "left":
            left_tetra.append(tet)
        else:
            right_tetra.append(tet)

    left_vertices, left_triangles, left_tetra, _ = build_subset(vertices, left_triangles, left_tetra)
    right_vertices, right_triangles, right_tetra, _ = build_subset(vertices, right_triangles, right_tetra)

    write_mesh(left_path, version, dimension, left_vertices, left_triangles, left_tetra)
    write_mesh(right_path, version, dimension, right_vertices, right_triangles, right_tetra)

    print(f"Piano di split X = {split_plane:.6f}")
    print(f"Mesh sinistra -> {left_path} (Tetraedri: {len(left_tetra)}, Triangoli: {len(left_triangles)})")
    print(f"Mesh destra   -> {right_path} (Tetraedri: {len(right_tetra)}, Triangoli: {len(right_triangles)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dividi una mesh .mesh in due metà lungo l'asse X.")
    parser.add_argument("input", type=Path, help="File .mesh di input (formato Medit).")
    parser.add_argument(
        "--plane-x",
        type=float,
        default=None,
        help="Coordinata X del piano di split. Default: punto medio del bounding box.",
    )
    parser.add_argument(
        "--left-output",
        type=Path,
        default=None,
        help="File di output per la metà sinistra. Default: <input>_left.mesh",
    )
    parser.add_argument(
        "--right-output",
        type=Path,
        default=None,
        help="File di output per la metà destra. Default: <input>_right.mesh",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolleranza per determinare se un elemento è da un lato del piano.",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"File di input '{input_path}' non trovato.")

    left_path = args.left_output or input_path.with_name(f"{input_path.stem}_left.mesh")
    right_path = args.right_output or input_path.with_name(f"{input_path.stem}_right.mesh")

    split_mesh(
        input_path=input_path,
        plane_x=args.plane_x,
        left_path=left_path,
        right_path=right_path,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
