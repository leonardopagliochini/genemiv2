#!/usr/bin/env python3
"""Augment a breast mesh with duct-like volumetric segments and terminal ellipsoids.

[... testo docstring invariato ...]
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
    cone_axis: np.ndarray
    min_angle_rad: float
    max_angle_x_rad: float
    max_angle_z_rad: float
    length: float
    radius: float
    segment_region_id: int
    ellipse_region_id: int
    cone_basis_x: np.ndarray
    cone_basis_z: np.ndarray
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray] = field(init=False)
    end_point: np.ndarray = field(init=False)
    min_slope: float = field(init=False)
    max_slope_x: float = field(init=False)
    max_slope_z: float = field(init=False)

    def __post_init__(self) -> None:
        direction = self.direction / np.linalg.norm(self.direction)
        cone_axis = self.cone_axis / np.linalg.norm(self.cone_axis)
        basis_x = np.asarray(self.cone_basis_x, dtype=float)
        basis_z = np.asarray(self.cone_basis_z, dtype=float)

        basis_x = _project_onto_plane(basis_x, cone_axis)
        x_norm = np.linalg.norm(basis_x)
        if x_norm < 1e-8:
            raise ValueError("Cone basis X must not be parallel to the cone axis.")
        basis_x /= x_norm

        basis_z = _project_onto_plane(basis_z, cone_axis)
        basis_z -= (basis_z @ basis_x) * basis_x
        z_norm = np.linalg.norm(basis_z)
        if z_norm < 1e-8:
            basis_z = np.cross(cone_axis, basis_x)
            z_norm = np.linalg.norm(basis_z)
        if z_norm < 1e-8:
            raise ValueError("Cone basis Z must not be parallel to the cone axis.")
        basis_z /= z_norm

        min_slope = np.tan(self.min_angle_rad)
        max_slope_x = np.tan(self.max_angle_x_rad)
        max_slope_z = np.tan(self.max_angle_z_rad)

        if not _direction_within_elliptical_cone(
            direction,
            cone_axis,
            basis_x,
            basis_z,
            min_slope,
            max_slope_x,
            max_slope_z,
        ):
            raise ValueError(
                "Segment direction lies outside the permitted elliptical cone."
            )

        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "cone_axis", cone_axis)
        object.__setattr__(self, "cone_basis_x", basis_x)
        object.__setattr__(self, "cone_basis_z", basis_z)
        object.__setattr__(self, "min_slope", min_slope)
        object.__setattr__(self, "max_slope_x", max_slope_x)
        object.__setattr__(self, "max_slope_z", max_slope_z)
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


def _project_onto_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project `vec` onto the plane orthogonal to `normal`."""
    normal = normal / np.linalg.norm(normal)
    return vec - (vec @ normal) * normal


def _build_cone_basis(cone_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return orthonormal directions aligned with global X and Z within the cone plane."""
    axis = cone_axis / np.linalg.norm(cone_axis)
    global_x = np.array([1.0, 0.0, 0.0])
    global_z = np.array([0.0, 0.0, 1.0])

    basis_x = _project_onto_plane(global_x, axis)
    if np.linalg.norm(basis_x) < 1e-8:
        basis_x = _project_onto_plane(global_z, axis)
    basis_x_norm = np.linalg.norm(basis_x)
    if basis_x_norm < 1e-8:
        raise ValueError("Failed to build X-aligned basis for the cone.")
    basis_x /= basis_x_norm
    if basis_x @ global_x < 0.0:
        basis_x = -basis_x

    basis_z = _project_onto_plane(global_z, axis)
    if np.linalg.norm(basis_z) < 1e-8:
        basis_z = np.cross(axis, basis_x)
    else:
        basis_z -= (basis_z @ basis_x) * basis_x
    basis_z_norm = np.linalg.norm(basis_z)
    if basis_z_norm < 1e-8:
        raise ValueError("Failed to build Z-aligned basis for the cone.")
    basis_z /= basis_z_norm
    if basis_z @ global_z < 0.0:
        basis_z = -basis_z

    return basis_x, basis_z


def _direction_within_elliptical_cone(
    direction: np.ndarray,
    axis: np.ndarray,
    basis_x: np.ndarray,
    basis_z: np.ndarray,
    min_slope: float,
    max_slope_x: float,
    max_slope_z: float,
    tol: float = 1e-6,
) -> bool:
    """Check whether `direction` lies inside the elliptical cone defined by the bounds."""
    direction = direction / np.linalg.norm(direction)
    axis = axis / np.linalg.norm(axis)
    basis_x = basis_x / np.linalg.norm(basis_x)
    basis_z = basis_z / np.linalg.norm(basis_z)

    axial = direction @ axis
    if axial <= tol:
        return False

    comp_x = direction @ basis_x
    comp_z = direction @ basis_z
    slope_x = abs(comp_x / axial)
    slope_z = abs(comp_z / axial)
    slope = np.hypot(slope_x, slope_z)

    if slope < max(min_slope - tol, 0.0):
        return False

    max_slope_x = max(max_slope_x, tol)
    max_slope_z = max(max_slope_z, tol)
    value = (slope_x / max_slope_x) ** 2 + (slope_z / max_slope_z) ** 2
    return value <= 1.0 + tol


def _load_mesh(path: str) -> meshio.Mesh:
    mesh = meshio.read(path)
    if "tetra" not in mesh.cells_dict:
        raise ValueError("Input mesh must contain tetrahedral volume elements.")
    return mesh


def _find_nipple(points: np.ndarray) -> Tuple[int, np.ndarray]:
    idx = int(np.argmax(points[:, 1]))
    return idx, points[idx]


def _generate_directions(
    count: int,
    min_angle_rad: float,
    max_angle_x_rad: float,
    max_angle_z_rad: float,
    base_axis: np.ndarray,
    basis_x: np.ndarray,
    basis_z: np.ndarray,
) -> List[np.ndarray]:
    """Spread directions inside an elliptical cone using a spiral pattern."""
    directions: List[np.ndarray] = []
    axis = base_axis / np.linalg.norm(base_axis)
    x_dir = basis_x / np.linalg.norm(basis_x)
    z_dir = basis_z / np.linalg.norm(basis_z)

    tan_min = np.tan(min_angle_rad)
    tan_x = np.tan(max_angle_x_rad)
    tan_z = np.tan(max_angle_z_rad)
    eps = 1e-9

    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(count):
        frac = (i + 0.5) / max(count, 1)
        azim = (i * golden_angle) % (2.0 * np.pi)
        cos_phi = np.cos(azim)
        sin_phi = np.sin(azim)

        denom = 0.0
        if tan_x > eps:
            denom += (cos_phi**2) / (tan_x**2)
        elif abs(cos_phi) > eps:
            denom = np.inf

        if tan_z > eps:
            denom += (sin_phi**2) / (tan_z**2)
        elif abs(sin_phi) > eps:
            denom = np.inf

        if not np.isfinite(denom) or denom <= 0.0:
            r_cap = 0.0
        else:
            r_cap = 1.0 / np.sqrt(denom)

        if tan_min > r_cap:
            r_min = r_cap
        else:
            r_min = tan_min

        if r_cap > r_min:
            r = r_min + frac * (r_cap - r_min)
        else:
            r = r_cap

        cos_theta = 1.0 / np.sqrt(1.0 + r * r)
        sin_theta = r * cos_theta
        radial_dir = cos_phi * x_dir + sin_phi * z_dir
        radial_norm = np.linalg.norm(radial_dir)
        if radial_norm < eps:
            radial_dir = x_dir
        else:
            radial_dir /= radial_norm

        dir_vec = cos_theta * axis + sin_theta * radial_dir
        dir_vec /= np.linalg.norm(dir_vec)

        if not _direction_within_elliptical_cone(
            dir_vec, axis, x_dir, z_dir, tan_min, tan_x, tan_z
        ):
            raise RuntimeError(
                "Generated direction violated elliptical cone constraints."
            )
        directions.append(dir_vec)
    return directions


def _build_segments(
    nipple: np.ndarray,
    base_axis: np.ndarray,
    y_extent: float,
    segment_count: int,
    segment_radius: float,
    max_angle_x_deg: float,
    max_angle_z_deg: float,
    min_length_ratio: float,
    max_length_ratio: float,
    starting_region_id: int,
    min_angle_deg: float,
) -> List[SegmentSpec]:
    min_angle_rad = np.deg2rad(min_angle_deg)
    max_angle_x_rad = np.deg2rad(max_angle_x_deg)
    max_angle_z_rad = np.deg2rad(max_angle_z_deg)
    if max(max_angle_x_rad, max_angle_z_rad) < min_angle_rad:
        raise ValueError(
            "Maximum cone angles must not be smaller than the minimum angle."
        )
    inward_axis = base_axis / np.linalg.norm(base_axis)
    basis_x, basis_z = _build_cone_basis(inward_axis)
    directions = _generate_directions(
        segment_count,
        min_angle_rad,
        max_angle_x_rad,
        max_angle_z_rad,
        inward_axis,
        basis_x,
        basis_z,
    )
    if len(directions) != segment_count:
        raise RuntimeError(
            "Failed to generate the requested number of segment directions."
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
                cone_axis=inward_axis,
                min_angle_rad=min_angle_rad,
                max_angle_x_rad=max_angle_x_rad,
                max_angle_z_rad=max_angle_z_rad,
                length=float(length),
                radius=segment_radius,
                segment_region_id=duct_region_id,
                ellipse_region_id=lobule_region_id,
                cone_basis_x=basis_x,
                cone_basis_z=basis_z,
            )
        )
    return segments


def _compute_tetra_centroids(points: np.ndarray, tetras: np.ndarray) -> np.ndarray:
    return points[tetras].mean(axis=1)


def _build_tetra_adjacency(tetras: np.ndarray) -> Tuple[Tuple[int, ...], ...]:
    """Return face-adjacency lists for each tetrahedron."""
    face_owner: dict[Tuple[int, int, int], int] = {}
    adjacency: List[List[int]] = [[] for _ in range(len(tetras))]
    # Each tetra has four faces opposite one vertex.
    face_indices = (
        (1, 2, 3),
        (0, 2, 3),
        (0, 1, 3),
        (0, 1, 2),
    )
    for idx, tetra in enumerate(tetras):
        for face in face_indices:
            key = tuple(sorted(tetra[list(face)]))
            owner = face_owner.get(key)
            if owner is None:
                face_owner[key] = idx
            else:
                adjacency[idx].append(owner)
                adjacency[owner].append(idx)
    return tuple(tuple(neigh) for neigh in adjacency)


def _dilate_mask(
    mask: np.ndarray, adjacency: Sequence[Sequence[int]]
) -> np.ndarray:
    """Expand `mask` by one adjacency layer."""
    if not np.any(mask):
        return mask.copy()
    dilated = mask.copy()
    active = np.flatnonzero(mask)
    for idx in active:
        for neighbor in adjacency[idx]:
            dilated[neighbor] = True
    return dilated


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
    center: np.ndarray | None = None,
) -> np.ndarray:
    u_vec, v_vec, w_vec = segment.frame
    center = segment.end_point if center is None else center
    rel = centroids - center
    coords = np.stack((rel @ u_vec, rel @ v_vec, rel @ w_vec), axis=1)
    rx, ry, rz = ellipse_axes
    norm_coords = np.empty_like(coords)
    norm_coords[:, 0] = coords[:, 0] / rx
    norm_coords[:, 1] = coords[:, 1] / ry
    norm_coords[:, 2] = coords[:, 2] / rz
    radius_sq = np.einsum("ij,ij->i", norm_coords, norm_coords)
    return radius_sq <= 1.0


def _orient_segment(
    centroids: np.ndarray,
    segment: SegmentSpec,
    ellipse_axes: Sequence[float],
    lobule_mask: np.ndarray,
    adjacency: Sequence[Sequence[int]],
) -> Tuple[SegmentSpec, np.ndarray, np.ndarray]:
    """Rotate segment direction to place lobule without colliding with existing ones."""
    forbidden = _dilate_mask(lobule_mask, adjacency)
    tilt_step_deg = 5.0
    max_tilt_deg = 45.0
    tilt_step_rad = np.deg2rad(tilt_step_deg)
    max_tilt_rad = np.deg2rad(max_tilt_deg)
    levels = int(np.ceil(max_tilt_rad / max(tilt_step_rad, 1e-6)))
    tilt_values = [0.0] + [min(max_tilt_rad, k * tilt_step_rad) for k in range(1, levels + 1)]
    base_u, base_v, base_w = segment.frame

    for tilt in tilt_values:
        if np.isclose(tilt, 0.0):
            directions = [segment.direction]
        else:
            azim_samples = 12
            directions = []
            sin_t = np.sin(tilt)
            cos_t = np.cos(tilt)
            for j in range(azim_samples):
                azim = (2.0 * np.pi * j) / azim_samples
                dir_vec = (
                    cos_t * base_w
                    + sin_t * (np.cos(azim) * base_u + np.sin(azim) * base_v)
                )
                directions.append(dir_vec / np.linalg.norm(dir_vec))
        for direction in directions:
            if not _direction_within_elliptical_cone(
                direction,
                segment.cone_axis,
                segment.cone_basis_x,
                segment.cone_basis_z,
                segment.min_slope,
                segment.max_slope_x,
                segment.max_slope_z,
            ):
                continue
            candidate = SegmentSpec(
                origin=segment.origin,
                direction=direction,
                cone_axis=segment.cone_axis,
                min_angle_rad=segment.min_angle_rad,
                max_angle_x_rad=segment.max_angle_x_rad,
                max_angle_z_rad=segment.max_angle_z_rad,
                length=segment.length,
                radius=segment.radius,
                segment_region_id=segment.segment_region_id,
                ellipse_region_id=segment.ellipse_region_id,
                cone_basis_x=segment.cone_basis_x,
                cone_basis_z=segment.cone_basis_z,
            )
            cyl_mask = _select_cylinder(centroids, candidate)
            if np.any(cyl_mask & lobule_mask):
                continue
            cyl_mask &= ~lobule_mask
            ellipsoid_mask = _select_ellipsoid(centroids, candidate, ellipse_axes)
            if not np.any(ellipsoid_mask):
                continue
            if np.any(ellipsoid_mask & forbidden):
                continue
            return candidate, cyl_mask, ellipsoid_mask

    fallback_cyl = _select_cylinder(centroids, segment)
    fallback_cyl &= ~lobule_mask
    fallback_ellipsoid = _select_ellipsoid(centroids, segment, ellipse_axes)
    fallback_ellipsoid &= ~forbidden
    return segment, fallback_cyl, fallback_ellipsoid


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
    adjacency = _build_tetra_adjacency(tetra)
    occupied_mask = np.zeros_like(refs, dtype=bool)
    lobule_mask = np.zeros_like(refs, dtype=bool)
    adjusted_segments: List[SegmentSpec] = []
    lobule_successes = 0
    for segment in segments:
        oriented_segment, seg_mask, ellipsoid_mask = _orient_segment(
            centroids, segment, ellipse_axes, lobule_mask, adjacency
        )
        if np.any(seg_mask):
            refs[seg_mask] = oriented_segment.segment_region_id
            occupied_mask |= seg_mask
        if np.any(ellipsoid_mask):
            refs[ellipsoid_mask] = oriented_segment.ellipse_region_id
            lobule_mask |= ellipsoid_mask
            occupied_mask |= ellipsoid_mask
            lobule_successes += 1
        adjusted_segments.append(oriented_segment)
    if (
        periduct_region_id is not None
        and periduct_shell > 0.0
        and len(base_region_ids) > 0
    ):
        periduct_mask = np.zeros_like(refs, dtype=bool)
        buffer_axes = tuple(ax + periduct_shell for ax in ellipse_axes)
        for segment in adjusted_segments:
            # Only capture a shell surrounding the lobule volume.
            thick_ellipsoid = _select_ellipsoid(centroids, segment, buffer_axes)
            periduct_mask |= thick_ellipsoid
        periduct_mask &= ~occupied_mask
        periduct_mask &= np.isin(original_refs, base_region_ids)
        refs[periduct_mask] = periduct_region_id
    if lobule_successes < len(segments):
        print(
            f"Warning: placed {lobule_successes} lobules out of {len(segments)} "
            "due to overlap and cone constraints."
        )
    mesh.cell_data["medit:ref"][tetra_block_idx] = refs
    return mesh


def _strip_surface_elements(mesh: meshio.Mesh) -> meshio.Mesh:
    """Return a mesh containing only tetrahedra, dropping triangles/lines etc."""
    keep_types = {"tetra"}
    new_cells = []
    keep_indices = []
    for i, block in enumerate(mesh.cells):
        if block.type in keep_types:
            new_cells.append(block)
            keep_indices.append(i)
    if not new_cells:
        raise ValueError("No tetrahedra found while stripping surface elements.")
    new_cell_data = {}
    for key, data_list in mesh.cell_data.items():
        new_cell_data[key] = [data_list[i] for i in keep_indices]
    return meshio.Mesh(
        points=mesh.points,
        cells=new_cells,
        cell_data=new_cell_data,
        field_data=getattr(mesh, "field_data", None),
    )


def _ensure_gmsh_physical_labels(
    mesh: meshio.Mesh,
    skin_old_id: int,
    fat_old_id: int,
    duct_old_id: int,
    lobule_old_id: int,
    stroma_old_id: int,
    renumber_to_standard: bool = True,
) -> meshio.Mesh:
    """Ensure that tetrahedral cells carry gmsh:physical ids and set field_data names."""
    try:
        tetra_block_idx = next(
            idx for idx, block in enumerate(mesh.cells) if block.type == "tetra"
        )
    except StopIteration:
        raise ValueError("Mesh lacks tetrahedra for labeling.")

    refs = None
    if "medit:ref" in mesh.cell_data:
        refs = mesh.cell_data["medit:ref"][tetra_block_idx]
    elif "gmsh:physical" in mesh.cell_data:
        refs = mesh.cell_data["gmsh:physical"][tetra_block_idx]
    else:
        raise ValueError("No region references found to assign gmsh physical groups.")

    if renumber_to_standard:
        mapping = {
            int(skin_old_id): 1,
            int(fat_old_id): 2,
            int(duct_old_id): 3,
            int(lobule_old_id): 4,
            int(stroma_old_id): 5,
        }
        new_refs = refs.copy()
        for old, new in mapping.items():
            new_refs[refs == old] = new
        refs = new_refs
        if "medit:ref" in mesh.cell_data:
            medit_refs = mesh.cell_data["medit:ref"][tetra_block_idx]
            new_medit = medit_refs.copy()
            for old, new in mapping.items():
                new_medit[medit_refs == old] = new
            mesh.cell_data["medit:ref"][tetra_block_idx] = new_medit

    gmsh_phys = list(mesh.cell_data.get("gmsh:physical", [None] * len(mesh.cells)))
    gmsh_phys = [
        gmsh_phys[i] if i != tetra_block_idx else refs.copy()
        for i in range(len(mesh.cells))
    ]
    mesh.cell_data["gmsh:physical"] = gmsh_phys

    mesh.field_data = {
        "Skin":   np.array([1, 3], dtype=int),
        "Fat":    np.array([2, 3], dtype=int),
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
        default=0.55,
        help="Radius of each cylindrical segment in mm (default: 1.0)",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=None,
        help=(
            "Legacy shortcut to set both X and Z angular bounds (degrees). "
            "If omitted, defaults are 20° (X) and 27° (Z)."
        ),
    )
    parser.add_argument(
        "--max-angle-x",
        type=float,
        default=18,
        help=(
            "Maximum angular deviation along the global X direction in degrees "
            "(default: 20 if unspecified)."
        ),
    )
    parser.add_argument(
        "--max-angle-z",
        type=float,
        default=27,
        help=(
            "Maximum angular deviation along the global Z direction in degrees "
            "(default: 27 if unspecified)."
        ),
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
        default=0.35,
        help="Lower bound for segment length as a fraction of total Y span (default: 0.20)",
    )
    parser.add_argument(
        "--max-length-ratio",
        type=float,
        default=0.45,
        help="Upper bound for segment length as a fraction of total Y span (default: 0.40)",
    )
    parser.add_argument(
        "--ellipse-long",
        type=float,
        default=16.0,
        help="Major axis (mm) of ellipsoids along the segment direction (default: 20)",
    )
    parser.add_argument(
        "--ellipse-short",
        type=float,
        default=8.0,
        help="First transverse axis (mm) of ellipsoids perpendicular to the segment (default: 10)",
    )
    parser.add_argument(
        "--ellipse-thickness",
        type=float,
        default=8.0,
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
    inward_axis = np.array([0.0, -1.0, 0.0])

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

    max_angle_x = 20.0
    max_angle_z = 27.0
    if args.max_angle is not None:
        max_angle_x = max_angle_z = args.max_angle
    if args.max_angle_x is not None:
        max_angle_x = args.max_angle_x
    if args.max_angle_z is not None:
        max_angle_z = args.max_angle_z

    segments = _build_segments(
        nipple=nipple_point,
        base_axis=inward_axis,
        y_extent=y_extent,
        segment_count=args.segment_count,
        segment_radius=args.segment_radius,
        max_angle_x_deg=max_angle_x,
        max_angle_z_deg=max_angle_z,
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

    cleaned = _strip_surface_elements(updated)

    # --- OLD LOGIC ---
    # base_ids = np.unique(tetra_refs)
    # if len(base_ids) < 2:
    #     raise ValueError("Expected at least two base regions for skin and fat.")
    # base_ids_sorted = np.sort(base_ids)
    # skin_old_id = int(base_ids_sorted[0])
    # fat_old_id  = int(base_ids_sorted[1])

    # --- NEW, MORE ROBUST LOGIC ---  # <<< CHANGED >>>
    base_ids = np.sort(np.unique(tetra_refs))

    if len(base_ids) == 0:
        # This should basically never happen if the mesh had tets,
        # but let's just guard.
        raise ValueError("Mesh has tetrahedra but no region IDs at all.")

    elif len(base_ids) == 1:
        # Only one material in the original mesh (e.g. only fat).
        # We'll call that 'fat', and create a dummy skin id that won't match anything.
        fat_old_id = int(base_ids[0])
        skin_old_id = -999999  # dummy that will not match any cell
    else:
        # We have at least two distinct regions. Assume the first is skin, second is fat.
        skin_old_id = int(base_ids[0])
        fat_old_id = int(base_ids[1])

    labeled = _ensure_gmsh_physical_labels(
        cleaned,
        skin_old_id=skin_old_id,
        fat_old_id=fat_old_id,
        duct_old_id=start_id,
        lobule_old_id=start_id + 1,
        stroma_old_id=periduct_region_id,
        renumber_to_standard=True,
    )

    out_fmt = None
    out_lower = str(args.output).lower()
    if out_lower.endswith(".msh"):
        out_fmt = "gmsh22"
    elif out_lower.endswith(".mesh"):
        out_fmt = "medit"

    if out_fmt is None:
        meshio.write(args.output, labeled)
    else:
        meshio.write(args.output, labeled, file_format=out_fmt, binary=False)


if __name__ == "__main__":
    main()
