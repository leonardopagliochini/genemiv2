#!/usr/bin/env pvpython
import os
import argparse
from paraview.simple import *

def process_vtu_file(vtu_path, output_dir):
    """Apply CleanToGrid + ExtractSurface and save as STL"""
    print(f"Processing: {vtu_path}")

    # 1. Read VTU
    reader = XMLUnstructuredGridReader(FileName=[vtu_path])

    # 2. Clean to Grid
    clean = CleantoGrid(Input=reader)

    # 3. Extract Surface
    surface = ExtractSurface(Input=clean)

    # 4. Prepare output path
    base_name = os.path.splitext(os.path.basename(vtu_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.stl")

    # 5. Save to STL
    SaveData(output_path, proxy=surface)
    print(f" → Saved: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract surfaces from VTU files using ParaView filters")
    parser.add_argument("--input", "-i", required=True, help="Input folder containing .vtu files")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.join(input_dir, "surfaces")
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(output_dir):
        if not name.endswith(".stl"):
            continue
        try:
            os.remove(os.path.join(output_dir, name))
        except OSError:
            pass

    # Find all VTU files in the folder
    vtu_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".vtu")]
    if not vtu_files:
        print(f"No .vtu files found in {input_dir}")
        return

    for vtu_file in sorted(vtu_files):
        process_vtu_file(vtu_file, output_dir)

    print("✅ All files processed successfully.")


if __name__ == "__main__":
    main()
