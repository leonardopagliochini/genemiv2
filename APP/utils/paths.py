from __future__ import annotations

from pathlib import Path


# Repository-level paths used across the workflow.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Default build/output locations for the standalone run_simulation helper (brain case).
BUILD_DIR = REPO_ROOT / "build" / "brain"
SIM_OUTPUT_DIR = BUILD_DIR / "output"


def case_build_dir(case: str | None = None) -> Path:
    from APP.utils.case_config import get_case_config

    return get_case_config(case).build_dir


def sim_output_dir(case: str | None = None) -> Path:
    from APP.utils.case_config import get_case_config

    return get_case_config(case).sim_output_dir()


def extracted_dir(timestep: int, case: str | None = None) -> Path:
    """Return the directory that stores filtered meshes for a timestep."""
    from APP.utils.case_config import get_case_config

    return get_case_config(case).extracted_dir(timestep)


def surfaces_dir(timestep: int, case: str | None = None) -> Path:
    """Return the directory that stores STL surfaces for a timestep."""
    from APP.utils.case_config import get_case_config

    return get_case_config(case).surfaces_dir(timestep)
