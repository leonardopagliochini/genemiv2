#!/usr/bin/env python3
"""Augment a breast mesh with duct-like volumetric segments and terminal ellipsoids.

The script loads an input MEDIT `.mesh` file that contains two breast volumes
(skin and fat), identifies the nipple as the vertex with the highest Y value,
and carves 15 cylindrical segments (radius 1 mm) that start at the nipple and
extend inwards. The segment directions stay within a user-configurable angular
window around the inward axis defined from the nipple towards the mesh centroid,
and their lengths span 20–40% of the overall mesh extent in Y.

At the end of each segment, an oblate ellipsoid (20 mm length along the segment,
10 mm x 10 mm cross-section) is created. Cells belonging to configurable source
regions (adipose by default) that fall within a configurable 5 mm shell
surrounding the ducts and ellipsoids are reassigned to a dedicated periductal
region. Ducts share one region identifier, lobules share another, and all other
volumes remain untouched apart from the optional periduct reassignment.

The default geometry matches the request, but CLI flags are provided so the
workflow can be tuned without editing the code.

Additional behavior per request:
- Remove surface elements (e.g., triangles/lines), keeping only tetrahedra.
- Name volumes with Gmsh physical groups: 1 Skin, 2 Fat, 3 Ductal, 4 Lobulus, 5 Stroma.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import meshio
import numpy as np


@dataclass(frozen=True)
class SegmentSpec:
    """Encapsulate geometry and region data for an inward segment."""

    origin: np.ndarray
    direction: np.ndarray
    length: float
    radius: float
    segment_region_id: int
    ellipse_region_id: int
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray] = field(init=False)
    end_point: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        direction = self.direction / np.linalg.norm(self.direction)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "end_point", self.origin + direction * self.length)
        object.__setattr__(self, "frame", _build_frame(direction))


def _build_frame(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an orthonormal frame with the third axis aligned with `axis`."""
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0])
    if np.isclose(abs(helper @ axis), 1.0):
        helper = np.array([0.0, 0.0, 1.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)
    return u, v, axis


def _load_mesh(path: str) -> meshio.Mesh:
    mesh = meshio.read(path)
    if "tetra" not in mesh.cells_dict:
        raise ValueError("Input mesh must contain tetrahedral volume elements.")
    return mesh


def _find_nipple(points: np.ndarray) -> Tuple[int, np.ndarray]:
    idx = int(np.argmax(points[:, 1]))
    return idx, points[idx]


def _generate_directions(
    count: int, min_angle_rad: float, max_angle_rad: float, base_axis: np.ndarray
) -> List[np.ndarray]:
    """Spread directions inside a cone around `base_axis` using a spiral pattern."""
    directions: List[np.ndarray] = []
    base_axis = base_axis / np.linalg.norm(base_axis)
    u, v, w = _build_frame(base_axis)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(count):
        frac = (i + 0.5) / count
        polar = min_angle_rad + frac * (max_angle_rad - min_angle_rad)
        azim = (i * golden_angle) % (2.0 * np.pi)
        sin_p = np.sin(polar)
        dir_vec = (
            np.cos(polar) * w
            + sin_p * np.cos(azim) * u
            + sin_p * np.sin(azim) * v
        )
        directions.append(dir_vec / np.linalg.norm(dir_vec))
    return directions


def _build_segments(
    nipple: np.ndarray,
    base_axis: np.ndarray,
    y_extent: float,
    segment_count: int,
    segment_radius: float,
    max_angle_deg: float,
    min_length_ratio: float,
    max_length_ratio: float,
    starting_region_id: int,
    min_angle_deg: float,
) -> List[SegmentSpec]:
    min_angle_rad = np.deg2rad(min_angle_deg)
    max_angle_rad = np.deg2rad(max_angle_deg)
    if max_angle_rad < min_angle_rad:
        raise ValueError("max_angle must be greater than or equal to min_angle.")
    inward_axis = base_axis / np.linalg.norm(base_axis)
    directions = _generate_directions(
        segment_count, min_angle_rad, max_angle_rad, inward_axis
    )
    lengths = np.linspace(min_length_ratio, max_length_ratio, segment_count) * y_extent
    duct_region_id = starting_region_id
    lobule_region_id = starting_region_id + 1
    segments: List[SegmentSpec] = []
    for direction, length in zip(directions, lengths):
        segments.append(
            SegmentSpec(
                origin=nipple,
                direction=direction,
                length=float(length),
                radius=segment_radius,
                segment_region_id=duct_region_id,
                ellipse_region_id=lobule_region_id,
            )
        )
    return segments


def _compute_tetra_centroids(points: np.ndarray, tetras: np.ndarray) -> np.ndarray:
    return points[tetras].mean(axis=1)


def _select_cylinder(
    centroids: np.ndarray,
    segment: SegmentSpec,
    radius: float | None = None,
) -> np.ndarray:
    radius = segment.radius if radius is None else radius
    rel = centroids - segment.origin
    axial = rel @ segment.direction
    within_length = (axial >= 0.0) & (axial <= segment.length)
    mask = np.zeros_like(axial, dtype=bool)
    if not np.any(within_length):
        return mask
    rel_in = rel[within_length] - np.outer(axial[within_length], segment.direction)
    radial_sq = np.einsum("ij,ij->i", rel_in, rel_in)
    inside = radial_sq <= radius**2
    mask[within_length] = inside
    return mask


def _select_ellipsoid(
    centroids: np.ndarray,
    segment: SegmentSpec,
    ellipse_axes: Sequence[float],
) -> np.ndarray:
    u_vec, v_vec, w_vec = segment.frame
    rel = centroids - segment.end_point
    coords = np.stack((rel @ u_vec, rel @ v_vec, rel @ w_vec), axis=1)
    rx, ry, rz = ellipse_axes
    norm_coords = np.empty_like(coords)
    norm_coords[:, 0] = coords[:, 0] / rx
    norm_coords[:, 1] = coords[:, 1] / ry
    norm_coords[:, 2] = coords[:, 2] / rz
    radius_sq = np.einsum("ij,ij->i", norm_coords, norm_coords)
    return radius_sq <= 1.0


def _update_regions(
    mesh: meshio.Mesh,
    segments: Iterable[SegmentSpec],
    ellipse_axes: Sequence[float],
    periduct_region_id: int | None,
    periduct_shell: float,
    base_region_ids: Sequence[int],
) -> meshio.Mesh:
    tetra_block_idx = next(
        idx for idx, block in enumerate(mesh.cells) if block.type == "tetra"
    )
    tetra = mesh.cells[tetra_block_idx].data
    original_refs = mesh.cell_data["medit:ref"][tetra_block_idx].copy()
    refs = original_refs.copy()
    centroids = _compute_tetra_centroids(mesh.points, tetra)
    occupied_mask = np.zeros_like(refs, dtype=bool)
    for segment in segments:
        seg_mask = _select_cylinder(centroids, segment)
        refs[seg_mask] = segment.segment_region_id
        occupied_mask |= seg_mask
    for segment in segments:
        ellipsoid_mask = _select_ellipsoid(centroids, segment, ellipse_axes)
        refs[ellipsoid_mask] = segment.ellipse_region_id
        occupied_mask |= ellipsoid_mask
    if (
        periduct_region_id is not None
        and periduct_shell > 0.0
        and len(base_region_ids) > 0
    ):
        periduct_mask = np.zeros_like(refs, dtype=bool)
        buffer_axes = tuple(ax + periduct_shell for ax in ellipse_axes)
        for segment in segments:
            thick_seg = _select_cylinder(
                centroids, segment, radius=segment.radius + periduct_shell
            )
            periduct_mask |= thick_seg
            thick_ellipsoid = _select_ellipsoid(centroids, segment, buffer_axes)
            periduct_mask |= thick_ellipsoid
        periduct_mask &= ~occupied_mask
        periduct_mask &= np.isin(original_refs, base_region_ids)
        refs[periduct_mask] = periduct_region_id
    mesh.cell_data["medit:ref"][tetra_block_idx] = refs
    return mesh


def _strip_surface_elements(mesh: meshio.Mesh) -> meshio.Mesh:
    """Return a mesh containing only tetrahedra, dropping triangles/lines etc.

    Cell data arrays are filtered to remain aligned with the kept cell blocks.
    """
    keep_types = {"tetra"}
    new_cells = []
    keep_indices = []
    for i, block in enumerate(mesh.cells):
        if block.type in keep_types:
            new_cells.append(block)
            keep_indices.append(i)
    if not new_cells:
        raise ValueError("No tetrahedra found while stripping surface elements.")
    # Filter cell data keys to the kept blocks
    new_cell_data = {}
    for key, data_list in mesh.cell_data.items():
        new_cell_data[key] = [data_list[i] for i in keep_indices]
    # Preserve field_data if present
    return meshio.Mesh(points=mesh.points, cells=new_cells, cell_data=new_cell_data, field_data=getattr(mesh, "field_data", None))


def _ensure_gmsh_physical_labels(
    mesh: meshio.Mesh,
    skin_old_id: int,
    fat_old_id: int,
    duct_old_id: int,
    lobule_old_id: int,
    stroma_old_id: int,
    renumber_to_standard: bool = True,
) -> meshio.Mesh:
    """Ensure that tetrahedral cells carry gmsh:physical ids and set field_data names.

    The mapping requested is:
    1 -> Skin, 2 -> Fat, 3 -> Ductal, 4 -> Lobulus, 5 -> Stroma

    If medit:ref is present, it is mirrored into gmsh:physical for tetrahedra.
    Field data are set unconditionally so that writing to Gmsh propagates names.
    """
    # Identify tetra block index in the (possibly stripped) mesh
    try:
        tetra_block_idx = next(idx for idx, block in enumerate(mesh.cells) if block.type == "tetra")
    except StopIteration:
        raise ValueError("Mesh lacks tetrahedra for labeling.")

    # Derive refs from either medit or gmsh data (prefer medit when present after augmentation)
    refs = None
    if "medit:ref" in mesh.cell_data:
        refs = mesh.cell_data["medit:ref"][tetra_block_idx]
    elif "gmsh:physical" in mesh.cell_data:
        refs = mesh.cell_data["gmsh:physical"][tetra_block_idx]
    else:
        raise ValueError("No region references found to assign gmsh physical groups.")

    # Optionally renumber to the standard IDs 1..5
    if renumber_to_standard:
        mapping = {
            int(skin_old_id): 1,
            int(fat_old_id): 2,
            int(duct_old_id): 3,
            int(lobule_old_id): 4,
            int(stroma_old_id): 5,
        }
        new_refs = refs.copy()
        # Map only known ids; keep others untouched
        for old, new in mapping.items():
            new_refs[refs == old] = new
        refs = new_refs
        # Keep medit references in sync if present
        if "medit:ref" in mesh.cell_data:
            medit_refs = mesh.cell_data["medit:ref"][tetra_block_idx]
            new_medit = medit_refs.copy()
            for old, new in mapping.items():
                new_medit[medit_refs == old] = new
            mesh.cell_data["medit:ref"][tetra_block_idx] = new_medit

    # Mirror refs into gmsh:physical for tetrahedra
    gmsh_phys = list(mesh.cell_data.get("gmsh:physical", [None] * len(mesh.cells)))
    gmsh_phys = [gmsh_phys[i] if i != tetra_block_idx else refs.copy() for i in range(len(mesh.cells))]
    mesh.cell_data["gmsh:physical"] = gmsh_phys

    # Set desired field_data names (dimension 3 for volumes)
    mesh.field_data = {
        "Skin": np.array([1, 3], dtype=int),
        "Fat": np.array([2, 3], dtype=int),
        "Ductal": np.array([3, 3], dtype=int),
        "Lobulus": np.array([4, 3], dtype=int),
        "Stroma": np.array([5, 3], dtype=int),
    }

    return mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        default="breast_128.mesh",
        help="Path to the source MEDIT mesh (default: breast_128.mesh)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="breast_128_augmented.msh",
        help="Output path for the augmented mesh (default: breast_128_augmented.msh)",
    )
    parser.add_argument(
        "--segment-count",
        type=int,
        default=15,
        help="Number of tubular segments to generate (default: 15)",
    )
    parser.add_argument(
        "--segment-radius",
        type=float,
        default=1.0,
        help="Radius of each cylindrical segment in mm (default: 1.0)",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=24.0,
        help="Maximum angular deviation from the nipple-to-centroid inward axis in degrees (default: 24)",
    )
    parser.add_argument(
        "--min-angle",
        type=float,
        default=0.0,
        help="Minimum angular deviation from the nipple-to-centroid inward axis in degrees (default: 0)",
    )
    parser.add_argument(
        "--min-length-ratio",
        type=float,
        default=0.20,
        help="Lower bound for segment length as a fraction of total Y span (default: 0.20)",
    )
    parser.add_argument(
        "--max-length-ratio",
        type=float,
        default=0.40,
        help="Upper bound for segment length as a fraction of total Y span (default: 0.40)",
    )
    parser.add_argument(
        "--ellipse-long",
        type=float,
        default=20.0,
        help="Major axis (mm) of ellipsoids along the segment direction (default: 20)",
    )
    parser.add_argument(
        "--ellipse-short",
        type=float,
        default=10.0,
        help="First transverse axis (mm) of ellipsoids perpendicular to the segment (default: 10)",
    )
    parser.add_argument(
        "--ellipse-thickness",
        type=float,
        default=10.0,
        help="Second transverse axis (mm) of ellipsoids perpendicular to the segment (default: 10)",
    )
    parser.add_argument(
        "--periduct-thickness",
        type=float,
        default=5.0,
        help="Shell thickness in mm to capture periductal adipose tissue (default: 5)",
    )
    parser.add_argument(
        "--starting-region-id",
        type=int,
        default=None,
        help="Base region ID for ducts (lobules use base+1, periduct defaults to base+2); defaults to max existing ID + 1",
    )
    parser.add_argument(
        "--periduct-region-id",
        type=int,
        default=None,
        help="Region ID to assign to periductal tissue (default: auto after ducts and ellipsoids)",
    )
    parser.add_argument(
        "--periduct-base-regions",
        type=int,
        nargs="+",
        default=[2],
        metavar="REGION",
        help="Existing region IDs eligible for periduct reassignment (default: 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh = _load_mesh(args.input)
    nipple_idx, nipple_point = _find_nipple(mesh.points)
    interior_target = mesh.points.mean(axis=0)
    inward_axis = interior_target - nipple_point
    if np.linalg.norm(inward_axis) < 1e-8:
        raise ValueError("Failed to derive inward axis from nipple towards mesh interior.")
    y_extent = mesh.points[:, 1].max() - mesh.points[:, 1].min()
    tetra_block_idx = next(
        idx for idx, block in enumerate(mesh.cells) if block.type == "tetra"
    )
    tetra_refs = mesh.cell_data["medit:ref"][tetra_block_idx]
    if args.periduct_thickness < 0.0:
        raise ValueError("periduct-thickness must be non-negative.")
    start_id = (
        args.starting_region_id
        if args.starting_region_id is not None
        else int(np.max(tetra_refs)) + 1
    )
    periduct_region_id = (
        args.periduct_region_id
        if args.periduct_region_id is not None
        else start_id + 2
    )
    segments = _build_segments(
        nipple=nipple_point,
        base_axis=inward_axis,
        y_extent=y_extent,
        segment_count=args.segment_count,
        segment_radius=args.segment_radius,
        max_angle_deg=args.max_angle,
        min_length_ratio=args.min_length_ratio,
        max_length_ratio=args.max_length_ratio,
        starting_region_id=start_id,
        min_angle_deg=args.min_angle,
    )
    ellipse_axes = (
        args.ellipse_short * 0.5,
        args.ellipse_thickness * 0.5,
        args.ellipse_long * 0.5,
    )
    updated = _update_regions(
        mesh,
        segments,
        ellipse_axes,
        periduct_region_id=periduct_region_id,
        periduct_shell=args.periduct_thickness,
        base_region_ids=tuple(args.periduct_base_regions),
    )
    # Drop surface elements and assign gmsh physical names for volumes
    cleaned = _strip_surface_elements(updated)
    # Determine original base (skin/fat) ids from the unmodified mesh tetra refs
    base_ids = np.unique(tetra_refs)
    if len(base_ids) < 2:
        raise ValueError("Expected at least two base regions for skin and fat.")
    base_ids_sorted = np.sort(base_ids)
    skin_old_id = int(base_ids_sorted[0])
    fat_old_id = int(base_ids_sorted[1])

    labeled = _ensure_gmsh_physical_labels(
        cleaned,
        skin_old_id=skin_old_id,
        fat_old_id=fat_old_id,
        duct_old_id=start_id,
        lobule_old_id=start_id + 1,
        stroma_old_id=periduct_region_id,
        renumber_to_standard=True,
    )

    # Choose output format from extension; default to gmsh22 if .msh
    out_fmt = None
    out_lower = str(args.output).lower()
    if out_lower.endswith(".msh"):
        out_fmt = "gmsh22"
    elif out_lower.endswith(".mesh"):
        out_fmt = "medit"

    if out_fmt is None:
        meshio.write(args.output, labeled)
    else:
        meshio.write(args.output, labeled, file_format=out_fmt)


if __name__ == "__main__":
    main()
