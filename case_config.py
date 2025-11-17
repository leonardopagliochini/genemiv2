from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple

from paths import REPO_ROOT


@dataclass(frozen=True)
class CaseConfig:
    """Declarative configuration for a supported pipeline case."""

    key: str
    label: str
    build_dir: Path
    binary_name: str
    default_mesh: str
    compile_commands: Tuple[Tuple[str, ...], ...]
    material_labels: Dict[int, str]
    priority_material_ids: Tuple[int, ...] = ()
    sticky_material_ids: Tuple[int, ...] = ()
    single_surface_labels: Dict[int, str] = field(default_factory=dict)
    default_threshold: float = 0.5
    default_procs: int = 4

    def binary_path(self) -> Path:
        return self.build_dir / self.binary_name

    def sim_output_dir(self) -> Path:
        return self.build_dir / "output"

    def extracted_dir(self, timestep: int) -> Path:
        return self.build_dir / f"extracted_{timestep:03d}"

    def surfaces_dir(self, timestep: int) -> Path:
        return self.extracted_dir(timestep) / "surfaces"

    def normalise_mesh_path(self, mesh: str | None) -> str:
        return mesh or self.default_mesh

    def material_ids(self) -> Iterable[int]:
        return self.material_labels.keys()


DEFAULT_CASE_KEY = "brain"


CASE_CONFIGS: Dict[str, CaseConfig] = {
    "brain": CaseConfig(
        key="brain",
        label="Brain",
        build_dir=REPO_ROOT / "build",
        binary_name="main",
        default_mesh="mesh/MNI_with_phys.msh",
        compile_commands=(("python3", "compile.py"),),
        material_labels={
            0: "gray_matter",
            1: "CSF",
            2: "white_matter",
            3: "ventricles",
        },
        priority_material_ids=(3, 1),
        sticky_material_ids=(1, 3),
        single_surface_labels={1: "CSF"},
        default_threshold=0.5,
        default_procs=14,
    ),
    "breast": CaseConfig(
        key="breast",
        label="Breast",
        build_dir=REPO_ROOT / "breast_scripts" / "build",
        binary_name="main",
        default_mesh="breast_scripts/mesh/breast_128_augmented.msh",
        compile_commands=(
            ("cmake", "-S", "breast_scripts", "-B", "breast_scripts/build"),
            ("cmake", "--build", "breast_scripts/build", "--target", "main"),
        ),
        material_labels={
            1: "skin",
            2: "fat",
            3: "ductal",
            4: "lobulus",
            5: "stroma",
        },
        priority_material_ids=(1,),
        sticky_material_ids=(1,),
        single_surface_labels={1: "skin"},
        default_threshold=0.2,
        default_procs=14,
    ),
}


def get_case_key(case: str | None) -> str:
    if not case:
        return DEFAULT_CASE_KEY
    normalized = case.strip().lower()
    if normalized not in CASE_CONFIGS:
        available = ", ".join(sorted(CASE_CONFIGS))
        raise ValueError(f"Unknown case '{case}'. Available options: {available}.")
    return normalized


def get_case_config(case: str | None) -> CaseConfig:
    return CASE_CONFIGS[get_case_key(case)]


def list_cases() -> Tuple[CaseConfig, ...]:
    return tuple(CASE_CONFIGS[key] for key in sorted(CASE_CONFIGS))
