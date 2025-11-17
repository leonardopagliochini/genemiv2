import argparse
import contextlib
import glob
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import meshio
import numpy as np
from APP.utils.case_config import DEFAULT_CASE_KEY, get_case_config, list_cases
from APP.utils.time_units import months_to_years, years_to_months

DEFAULT_YEARS = 9
DEFAULT_TIMESTEP = DEFAULT_YEARS * 12
DEFAULT_FIELD = "c"
DEFAULT_THRESHOLD = None
EXPORT_DIR_TEMPLATE = "{label}y_extracted"
MATERIAL_FIELD = "material_id"

CASE_CHOICES = tuple(cfg.key for cfg in list_cases())


def format_years_label(years: float) -> str:
    text = f"{years:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate meshes filtered by a field threshold by reading all VTU pieces "
            "belonging to the same timestep."
        )
    )
    parser.add_argument(
        "--case",
        choices=CASE_CHOICES,
        default=DEFAULT_CASE_KEY,
        help=f"Pipeline case to process (default: {DEFAULT_CASE_KEY}).",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=None,
        help=(
            "Numero di anni del dato; imposta il timestep a years*12. "
            f"Default: {DEFAULT_YEARS} anni."
        ),
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=None,
        help=(
            "Timestep number to inspect (e.g. 6 for files named output_006.xx.vtu). "
            f"Default: {DEFAULT_TIMESTEP}."
        ),
    )
    parser.add_argument(
        "--field",
        type=str,
        default=DEFAULT_FIELD,
        help="Point-data field to analyze.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Numeric threshold used to filter points (defaults to the case configuration).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where filtered meshes will be written. Default: '<years>_extracted'.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing the raw simulation VTU files. Defaults to the case output directory.",
    )
    args = parser.parse_args()

    if args.years is not None:
        try:
            args.timestep = years_to_months(args.years)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
    elif args.timestep is None:
        args.years = DEFAULT_YEARS
        args.timestep = DEFAULT_TIMESTEP

    if args.timestep is None:
        parser.error("Unable to determine the timestep to process.")

    canonical_years = months_to_years(args.timestep)
    args.years = canonical_years
    year_label = format_years_label(canonical_years)

    if args.output_dir is None:
        args.output_dir = Path(EXPORT_DIR_TEMPLATE.format(label=year_label))
    if not args.output_dir.is_absolute():
        args.output_dir = args.output_dir.resolve()
    if args.input_dir is not None and not args.input_dir.is_absolute():
        args.input_dir = args.input_dir.resolve()

    args.year_label = year_label
    return args


def find_vtu_files(timestep: int, input_dir: Path) -> list[Path]:
    timestep_str = f"{timestep:03d}"
    pattern = input_dir / f"output_{timestep_str}*.vtu"
    return sorted(Path(path) for path in glob.glob(pattern.as_posix()))


def collapse_point_materials(
    per_cell_values: np.ndarray, priority_ids: tuple[int, ...]
) -> np.ndarray:
    """Derive one material id per cell starting from per-vertex material ids."""
    array = np.asarray(per_cell_values, dtype=int)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(
            f"Il campo '{MATERIAL_FIELD}' sui punti deve risultare in un array bidimensionale per cella."
        )

    num_cells = array.shape[0]
    result = np.empty(num_cells, dtype=int)

    remaining_mask = np.ones(num_cells, dtype=bool)

    for material_id in priority_ids:
        mask = np.any(array == material_id, axis=1) & remaining_mask
        result[mask] = material_id
        remaining_mask &= ~mask

    if np.any(remaining_mask):
        remaining_values = array[remaining_mask]
        dominant_labels = np.array(
            [np.bincount(row).argmax() for row in remaining_values],
            dtype=int,
        )
        result[remaining_mask] = dominant_labels

    return result


def load_combined_mesh(
    vtu_files: list[Path],
    field_name: str,
    material_field: str,
    priority_ids: tuple[int, ...],
) -> tuple[np.ndarray, list[meshio.CellBlock], np.ndarray, list[np.ndarray]]:
    points_list: list[np.ndarray] = []
    values_list: list[np.ndarray] = []
    cells_map: dict[str, list[np.ndarray]] = defaultdict(list)
    material_map: dict[str, list[np.ndarray]] = defaultdict(list)
    point_offset = 0

    def normalize_material_values(
        raw: np.ndarray, *, context: str
    ) -> np.ndarray:
        arr = np.asarray(raw)
        if arr.ndim == 2:
            if arr.shape[1] != 1:
                raise ValueError(
                    f"Il campo '{material_field}' in {context} deve avere una sola componente."
                )
            arr = arr[:, 0]
        elif arr.ndim != 1:
            raise ValueError(
                f"Forma non supportata per il campo '{material_field}' in {context}."
            )

        float_arr = arr.astype(float, copy=False)
        rounded = np.rint(float_arr)
        if not np.allclose(float_arr, rounded, atol=1e-6):
            raise ValueError(
                f"Il campo '{material_field}' deve contenere valori interi, trovati valori non interi in {context}."
            )
        return rounded.astype(int)

    for path in vtu_files:
        mesh = meshio.read(path.as_posix())
        if field_name not in mesh.point_data:
            available = ", ".join(mesh.point_data.keys())
            raise KeyError(
                f"Il campo '{field_name}' non è presente in {path.name}. "
                f"Campi disponibili: {available or 'nessuno'}"
            )

        points_list.append(mesh.points)
        values_list.append(mesh.point_data[field_name])

        material_source: Optional[str] = None
        cell_material_blocks: Optional[list[np.ndarray]] = None
        point_material_values: Optional[np.ndarray] = None

        if material_field in mesh.cell_data:
            material_source = "cell"
            material_blocks = mesh.cell_data[material_field]
            if len(material_blocks) != len(mesh.cells):
                raise ValueError(
                    f"Numero di blocchi di celle per '{material_field}' inaspettato in {path.name}."
                )
            cell_material_blocks = [
                normalize_material_values(
                    block_values,
                    context=f"cell data '{material_field}' ({block.type}) in {path.name}",
                )
                for block_values, block in zip(material_blocks, mesh.cells)
            ]
        elif material_field in mesh.point_data:
            material_source = "point"
            point_material_values = normalize_material_values(
                mesh.point_data[material_field],
                context=f"point data '{material_field}' in {path.name}",
            )
            expected_points = mesh.points.shape[0]
            if point_material_values.shape[0] != expected_points:
                raise ValueError(
                    f"Numero di valori per '{material_field}' ({point_material_values.shape[0]}) "
                    f"diverso dal numero di punti ({expected_points}) in {path.name}."
                )
        else:
            available_point = ", ".join(mesh.point_data.keys())
            available_cell = ", ".join(mesh.cell_data.keys())
            raise KeyError(
                f"Il campo '{material_field}' non è presente in {path.name}. "
                f"Punti disponibili: {available_point or 'nessuno'}. "
                f"Celle disponibili: {available_cell or 'nessuna'}."
            )

        for block_idx, block in enumerate(mesh.cells):
            cells_map[block.type].append(block.data + point_offset)

            if material_source == "cell" and cell_material_blocks is not None:
                material_values = cell_material_blocks[block_idx]
                if material_values.shape[0] != block.data.shape[0]:
                    raise ValueError(
                        f"Dimensione del campo '{material_field}' non coerente con il blocco "
                        f"'{block.type}' in {path.name}."
                    )
                material_map[block.type].append(np.asarray(material_values, dtype=int))
            elif material_source == "point" and point_material_values is not None:
                per_cell_values = point_material_values[block.data]
                material_ids = collapse_point_materials(per_cell_values, priority_ids)
                material_map[block.type].append(material_ids)
            else:
                raise RuntimeError(
                    f"Sorgente del campo '{material_field}' non riconosciuta per {path.name}."
                )

        point_offset += mesh.points.shape[0]

    points = np.vstack(points_list)
    values = np.concatenate(values_list)
    cells: list[meshio.CellBlock] = []
    cell_materials: list[np.ndarray] = []
    for cell_type, blocks in cells_map.items():
        merged_cells = np.vstack(blocks)
        cells.append(meshio.CellBlock(cell_type, merged_cells))
        materials = material_map.get(cell_type)
        if not materials:
            raise KeyError(
                f"Non sono stati trovati valori per '{material_field}' per celle di tipo '{cell_type}'."
            )
        cell_materials.append(np.concatenate(materials))
    return points, cells, values, cell_materials


def build_subset_mesh(
    points: np.ndarray,
    cells: list[meshio.CellBlock],
    values: np.ndarray,
    predicate: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    field_name: str,
    cell_materials: Optional[list[np.ndarray]] = None,
    material_field: Optional[str] = None,
) -> Optional[meshio.Mesh]:
    if cell_materials is not None and material_field is None:
        raise ValueError(
            "È necessario specificare 'material_field' quando si passano 'cell_materials'."
        )

    selected_blocks: list[meshio.CellBlock] = []
    selected_materials: Optional[list[np.ndarray]] = (
        [] if cell_materials is not None else None
    )
    used_points = np.zeros(points.shape[0], dtype=bool)

    for idx, block in enumerate(cells):
        cell_values = values[block.data]
        block_materials = cell_materials[idx] if cell_materials is not None else None
        mask = predicate(cell_values, block_materials)
        if not np.any(mask):
            continue

        filtered = block.data[mask]
        selected_blocks.append(meshio.CellBlock(block.type, filtered))
        used_points[np.unique(filtered)] = True
        if block_materials is not None and selected_materials is not None:
            selected_materials.append(block_materials[mask])

    if not selected_blocks:
        return None

    kept_indices = np.where(used_points)[0]
    index_map = -np.ones(points.shape[0], dtype=int)
    index_map[kept_indices] = np.arange(kept_indices.size)

    remapped_blocks = [
        meshio.CellBlock(block.type, index_map[block.data])
        for block in selected_blocks
    ]

    new_points = points[kept_indices]
    new_values = values[kept_indices]
    cell_data: dict[str, list[np.ndarray]] = {}
    if selected_materials:
        if material_field is None:
            raise ValueError("material_field non può essere None quando si esportano i dati delle celle.")
        cell_data[material_field] = selected_materials

    return meshio.Mesh(
        points=new_points,
        cells=remapped_blocks,
        point_data={field_name: new_values},
        cell_data=cell_data if cell_data else None,
    )

def main() -> None:
    args = parse_args()
    case_config = get_case_config(args.case)

    input_dir = args.input_dir if args.input_dir is not None else case_config.sim_output_dir()
    if not input_dir.is_absolute():
        input_dir = input_dir.resolve()

    vtu_files = find_vtu_files(args.timestep, input_dir)
    if not vtu_files:
        pattern = input_dir / f"output_{args.timestep:03d}*.vtu"
        raise FileNotFoundError(
            f"Nessun file trovato per il timestep {args.timestep} usando il pattern {pattern}"
        )

    field_name = args.field
    threshold = args.threshold if args.threshold is not None else case_config.default_threshold
    if threshold is None:
        raise ValueError("È necessario specificare un valore di soglia numerico per l'estrazione.")

    print(
        f"Timestep {args.timestep:03d} ({format_years_label(args.years)} anni): "
        f"trovati {len(vtu_files)} file VTU."
    )
    points, cell_blocks, values, cell_materials = load_combined_mesh(
        vtu_files,
        field_name,
        MATERIAL_FIELD,
        tuple(case_config.priority_material_ids),
    )
    total_points = values.size

    sticky_ids = tuple(case_config.sticky_material_ids)

    def special_mask(materials: Optional[np.ndarray], reference: np.ndarray) -> np.ndarray:
        if materials is None or not sticky_ids:
            return np.zeros_like(reference, dtype=bool)
        return np.isin(materials, sticky_ids)

    def below_predicate(
        cell_vals: np.ndarray, materials: Optional[np.ndarray]
    ) -> np.ndarray:
        cell_min = np.min(cell_vals, axis=1)
        cell_max = np.max(cell_vals, axis=1)
        mean_vals = np.mean(cell_vals, axis=1)

        mask = cell_max < threshold
        special = special_mask(materials, mean_vals)
        mask |= special

        ambiguous = ~(mask | (cell_min >= threshold))
        if np.any(ambiguous):
            mask |= ambiguous & (mean_vals <= threshold)

        return mask

    def above_predicate(
        cell_vals: np.ndarray, materials: Optional[np.ndarray]
    ) -> np.ndarray:
        cell_min = np.min(cell_vals, axis=1)
        cell_max = np.max(cell_vals, axis=1)
        mean_vals = np.mean(cell_vals, axis=1)

        mask = cell_min >= threshold
        special = special_mask(materials, mean_vals)
        mask &= ~special

        ambiguous = ~(mask | (cell_max < threshold) | special)
        if np.any(ambiguous):
            mask |= ambiguous & (mean_vals > threshold)

        mask &= ~special
        return mask

    below_mesh = build_subset_mesh(
        points,
        cell_blocks,
        values,
        predicate=below_predicate,
        field_name=field_name,
        cell_materials=cell_materials,
        material_field=MATERIAL_FIELD,
    )
    above_mesh = build_subset_mesh(
        points,
        cell_blocks,
        values,
        predicate=above_predicate,
        field_name=field_name,
        cell_materials=cell_materials,
        material_field=MATERIAL_FIELD,
    )

    if below_mesh is None and above_mesh is None:
        raise ValueError(
            "Nessuna cella soddisfa i criteri sotto o sopra la soglia. Controlla il valore del threshold."
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale VTU files and previously extracted surfaces to avoid duplicate exports.
    for stale in output_dir.glob("*.vtu"):
        with contextlib.suppress(OSError):
            stale.unlink()
    surfaces_root = output_dir / "surfaces"
    if surfaces_root.exists():
        shutil.rmtree(surfaces_root, ignore_errors=True)
    year_label = args.year_label

    material_labels = case_config.material_labels
    single_surface_labels = case_config.single_surface_labels
    single_surface_ids = set(single_surface_labels)

    def save_mesh(tag_label: str, mesh: meshio.Mesh) -> None:
        base_name = f"output_{year_label}y_{tag_label}"
        vtu_path = output_dir / f"{base_name}.vtu"

        meshio.write(vtu_path.as_posix(), mesh)
        print(f"Scritto {vtu_path}")

    def write_meshes(tag: str, mesh: Optional[meshio.Mesh]) -> None:
        if mesh is None:
            print(f"Nessuna cella per '{tag}', salto l'esportazione.")
            return

        if MATERIAL_FIELD not in mesh.cell_data:
            raise KeyError(
                f"Il campo '{MATERIAL_FIELD}' non è presente nella mesh filtrata '{tag}'."
            )
        if field_name not in mesh.point_data:
            raise KeyError(
                f"Il campo '{field_name}' non è presente nella mesh filtrata '{tag}'."
            )

        cell_blocks_local = list(mesh.cells)
        material_blocks_local = [
            np.asarray(arr).reshape(-1) for arr in mesh.cell_data[MATERIAL_FIELD]
        ]
        values_array = np.asarray(mesh.point_data[field_name])

        for material_id, label in material_labels.items():
            if material_id in single_surface_ids:
                continue

            def material_predicate(
                _cell_vals: np.ndarray,
                block_materials: Optional[np.ndarray],
                *,
                target=material_id,
            ) -> np.ndarray:
                if block_materials is None:
                    raise ValueError(
                        "Dati di materiale mancanti durante la suddivisione per materiale."
                    )
                return block_materials == target

            submesh = build_subset_mesh(
                mesh.points,
                cell_blocks_local,
                values_array,
                predicate=material_predicate,
                field_name=field_name,
                cell_materials=material_blocks_local,
                material_field=MATERIAL_FIELD,
            )

            tag_label = f"{tag}_{label}"
            if submesh is None:
                print(f"Nessuna cella per '{tag_label}', salto l'esportazione.")
                continue

            save_mesh(tag_label, submesh)

    write_meshes("no_cancer", below_mesh)
    write_meshes("cancer", above_mesh)

    for material_id, label in single_surface_labels.items():
        def only_material(
            _cell_vals: np.ndarray,
            block_materials: Optional[np.ndarray],
            *,
            target=material_id,
        ) -> np.ndarray:
            if block_materials is None:
                raise ValueError(
                    "Dati di materiale mancanti durante la suddivisione per materiale."
                )
            return block_materials == target

        single_mesh = build_subset_mesh(
            points,
            cell_blocks,
            values,
            predicate=only_material,
            field_name=field_name,
            cell_materials=cell_materials,
            material_field=MATERIAL_FIELD,
        )
        if single_mesh is None:
            print(f"Nessuna cella per '{label}', salto l'esportazione.")
        else:
            save_mesh(label, single_mesh)

    print(f"Punti totali elaborati: {total_points}")


if __name__ == "__main__":
    main()
