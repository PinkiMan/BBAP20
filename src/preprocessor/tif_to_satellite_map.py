__author__ = "Pinkas Matěj"
__maintainer__ = "Pinkas Matěj"
__email__ = "pinkas.matej@gmail.com"
__created__ = "22/05/2026"
__date__ = "22/05/2026"
__status__ = "Prototype"
__version__ = "0.1.0"
__copyright__ = ""
__license__ = ""
__credits__ = []

"""
Project: BBAP20
Filename: tif_to_satellite_map.py
Directory: src/preprocessor/
"""

import os
from PIL import Image

def split_tif_to_pngs(tif_filepath: str, output_dir: str, base_name:str = "", tile_size: int = 500, output_size: int = 500, z_height=float("inf")):
    print(f"Opening file: {tif_filepath} ...")

    try:
        img = Image.open(tif_filepath)
    except Exception as e:
        raise Exception(f"Error opening {tif_filepath}: {e}")

    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')

    img_width, img_height = img.size
    print(f"Input image size: {img_width} x {img_height} pixels")

    os.makedirs(output_dir, exist_ok=True)
    #base_name = os.path.splitext(os.path.basename(tif_filepath))[0]

    saved_tiles = 0

    for y in range(0, img_height - tile_size + 1, tile_size):
        for x in range(0, img_width - tile_size + 1, tile_size):
            box = (x, y, x + tile_size, y + tile_size)
            cropped_img = img.crop(box)

            if tile_size != output_size:
                cropped_img = cropped_img.resize((output_size, output_size))

            filename = f"{base_name}_{z_height}_{saved_tiles}-satellite_map.png"
            filepath = os.path.join(output_dir, filename)

            cropped_img.save(filepath, format="PNG")
            saved_tiles += 1

    print(f"Finished. Saved to {saved_tiles} with size ({output_size}x{output_size}) to folder '{output_dir}'.")

if __name__ == '__main__':
    tif_img_filepath = "tif_to_satmap.png"
    img_output_dir = "img"
    #split_tif_to_pngs()
