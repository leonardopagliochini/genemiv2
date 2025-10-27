#!/usr/bin/env python3
"""
Utility script to launch the Fisher-Kolmogorov simulation with user-defined parameters.

Edit the variables below to customise the run without touching the C++ sources.
"""

import subprocess
from pathlib import Path

from paths import BUILD_DIR, REPO_ROOT, SIM_OUTPUT_DIR

# ---------------------------------------------------------------------------
# User-configurable parameters
T = 9.0  # Final simulation time
mesh_path = "mesh/MNI_with_phys.msh"  # Mesh preset or path to a supported mesh file
output_folder = SIM_OUTPUT_DIR  # Folder where VTU results will be written
num_processors = 4  # Number of MPI processes to launch

# Optional overrides (uncomment / adjust as needed)
# deltat = 1.0 / 12.0
# output_period = 6
# ---------------------------------------------------------------------------


def main() -> int:
    repo_root = REPO_ROOT
    binary = BUILD_DIR / "main"

    if not binary.exists():
        print(f"[error] Binary '{binary}' not found. Compile the project first (e.g. python3 compile.py).")
        return 1

    output_path = Path(output_folder)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    print("T =", T)

    cmd = [
        "mpirun",
        "-np",
        str(num_processors),
        str(binary),
        "--mesh",
        mesh_path,
        "--T",
        str(T),
        "--output",
        str(output_path),
    ]

    # Optional arguments (only pass if defined)
    if "deltat" in globals():
        cmd.extend(["--deltat", str(deltat)])
    if "output_period" in globals():
        cmd.extend(["--output-period", str(output_period)])

    print("[info] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"[error] Simulation failed with exit code {exc.returncode}")
        raise SystemExit(exc.returncode) from exc
