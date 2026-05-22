__author__ = "Pinkas Matěj"
__maintainer__ = "Pinkas Matěj"
__email__ = "pinkas.matej@gmail.com"
__created__ = "22/05/2026"
__date__ = "22/05/2026"
__status__ = "Prototype"
__version__ = "0.1.0"
__credits__ = []

"""
Project: BBAP20
Filename: 01_prepare_data.py
Directory: scripts/
"""

import pathlib
from pathlib import Path

from src.preprocessor import tif_to_satellite_map, laz_to_heightmap
from src.shared.utils import project_directory



def save_imgs(laz_filepath, tif_filepath, output_folder, base_name):
    heightmap_size_px = 500
    heightmap_bin_count = 256
    heightmap_size_m = 100  # keep
    satellite_map_size_px = 256
    satellite_map_size_m = 500  # keep

    # could be like [5,10,15,20,25,30,35,40,45,float("inf")]:
    for z_height in [float("inf")]:
        tif_to_satellite_map.split_tif_to_pngs(tif_filepath=tif_filepath, output_dir=output_folder, base_name=base_name, tile_size=satellite_map_size_m, output_size=satellite_map_size_px, z_height=z_height)
        laz_to_heightmap.split_lidar_into_blocks(laz_filepath=laz_filepath, output_folder=output_folder, block_size=heightmap_size_m, base_name=base_name, bin_count=heightmap_bin_count, z_height=z_height, mode="all")


def process_directory(input_folder, output_folder):
    num = 0
    for item in input_folder.iterdir():
        if item.is_file() and item.name.endswith(".laz"):
            num += 1

    i = 0

    for item in input_folder.iterdir():
        if item.is_file() and item.name.endswith(".laz"):
            base_name = item.name.split(".")[0]

            print(f"Processing {base_name} {i}/{num}")
            i += 1

            existing_files = list(output_folder.glob(f"*{base_name}*"))

            if len(existing_files) >= 200:
                print(f"[{i}/{num}] Skipping '{base_name}' - completely processed files.")
                continue
            elif len(existing_files) > 0:
                print(
                    f"[{i}/{num}] Found uncompleted data for '{base_name}' ({len(existing_files)}/200). Overwriting...")
            else:
                print(f"[{i}/{num}] Processing '{base_name}'...")

            laz_filepath = input_folder / pathlib.Path(base_name + ".laz")
            tif_filepath = input_folder / pathlib.Path("32" + base_name + ".tif")

            if laz_filepath.is_file() and tif_filepath.is_file():
                save_imgs(laz_filepath, tif_filepath, output_folder, base_name)
            else:
                print(f"[ERROR] For file '{base_name}' missing .laz or .tif pair!")


def main():
    directory = project_directory(Path('data/dataset'))

    print("Preparing data for training...")
    output_folder = directory / "processed" / "train"
    input_folder = directory / "raw" / "train"
    process_directory(input_folder, output_folder)

    print("Preparing data for validation...")
    output_folder = directory / "processed" / "validation"
    input_folder = directory / "raw" / "validation"
    process_directory(input_folder, output_folder)

    print("Preparing data for testing...")
    output_folder = directory / "processed" / "test"
    input_folder = directory / "raw" / "test"
    process_directory(input_folder, output_folder)


if __name__ == '__main__':
    main()

