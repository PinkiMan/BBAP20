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
Filename: laz_to_heightmap.py
Directory: src/preprocessor/
"""

import laspy
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt


def load_laz(laz_filepath: pathlib.Path) -> laspy.LasData:
    """ loads laz data and returns as las object"""
    las = laspy.read(laz_filepath)

    orig_x = np.copy(las.x)
    orig_y = np.copy(las.y)
    las.header.x_offset = np.min(-orig_y)
    las.header.y_offset = np.min(orig_x)
    las.x = -orig_y
    las.y = orig_x

    return las

def pointcloud_to_heightmap(point_cloud, bin_count=256, norm=True, move_to_ground=True, z_height=float("inf"), mode=None):
    pts_xyz = point_cloud[:, :3]

    x = pts_xyz[:, 0]
    y = pts_xyz[:, 1]
    z = pts_xyz[:, 2]

    z_mask = z <= (np.min(z) + z_height)

    x = x[z_mask]
    y = y[z_mask]
    z = z[z_mask]

    if move_to_ground:
        z = z - np.min(z)

    if norm:
        z = z / np.max(z)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    max_range = max(x_range, y_range)

    if mode == 'center':
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        x_min_centered = x_center - max_range / 4
        x_max_centered = x_center + max_range / 4
        y_min_centered = y_center - max_range / 4
        y_max_centered = y_center + max_range / 4
    elif mode == "corner":
        side = max_range/2

        x_min_centered = x_min
        x_max_centered = x_min + side
        y_min_centered = y_min
        y_max_centered = y_min + side
    elif mode == "fit_center":
        dist = min([max([x_min, x_max, y_min, y_max]), 100])
        #dist = 50

        x_min_centered = -dist
        x_max_centered = dist
        y_min_centered = -dist
        y_max_centered = dist
    else:
        x_min_centered = x_min
        x_max_centered = x_min + max_range
        y_min_centered = y_min
        y_max_centered = y_min + max_range


    #[x_min_centered, x_max_centered], [y_min_centered, y_max_centered] = [x_min_centered, x_max_centered], [y_min_centered, y_max_centered]

    N = bin_count

    x_edges = np.linspace(x_min_centered, x_max_centered, N + 1)
    y_edges = np.linspace(y_min_centered, y_max_centered, N + 1)

    x_idx = np.digitize(x, x_edges) - 1
    y_idx = np.digitize(y, y_edges) - 1

    valid = (
            (x_idx >= 0) & (x_idx < N) &
            (y_idx >= 0) & (y_idx < N)
    )

    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    z = z[valid]

    data = idx_to_heightmap(x_idx, y_idx, N, z)

    return data

def idx_to_heightmap(x_idx, y_idx, output_size, z):
    max_z = np.full((output_size, output_size), -np.inf)  # init bins set to -inf
    np.maximum.at(max_z, (x_idx, y_idx), z)  # set maximum to bins
    max_z[max_z == -np.inf] = None  # empty bins set to None

    valid_min = np.nanmin(max_z)
    valid_max = np.nanmax(max_z)
    data = np.nan_to_num(max_z, nan=valid_min)

    if valid_max > valid_min:
        data = (data - valid_min) / (valid_max - valid_min)
    else:
        data = np.zeros_like(data)

    return data

def split_lidar_into_blocks(laz_filepath, output_folder, block_size, base_name, bin_count, z_height, mode="center"):
    las = load_laz(laz_filepath)

    min_x, max_x = np.min(las.x), np.max(las.x)
    min_y, max_y = np.min(las.y), np.max(las.y)
    min_z, max_z = np.min(las.z), np.max(las.z)

    print(f"Range X: {min_x:.2f} to {max_x:.2f}")
    print(f"Range Y: {min_y:.2f} to {max_y:.2f}")
    print(f"Range Z: {min_z:.2f} to {max_z:.2f}")

    os.makedirs(output_folder, exist_ok=True)

    x_steps = np.arange(min_x, max_x, block_size)
    y_steps = np.arange(min_y, max_y, block_size)

    total_blocks = len(x_steps) * len(y_steps)
    print(f"Pointcloud would be split to max {total_blocks} blocks (skips empty).")

    saved_blocks = 0

    for x_start in x_steps:
        for y_start in y_steps:
            x_end = x_start + block_size
            y_end = y_start + block_size

            mask = (las.x >= x_start) & (las.x < x_end) & \
                   (las.y >= y_start) & (las.y < y_end)

            if np.any(mask):
                new_las = laspy.LasData(las.header)
                new_las.points = las.points[mask]

                x = new_las.x
                y = new_las.y
                z = new_las.z

                point_cloud = np.vstack((x, y, z)).transpose()
                height_map = pointcloud_to_heightmap(point_cloud, bin_count=bin_count, z_height=z_height, mode=mode)

                filename = f"{base_name}_{z_height}_{saved_blocks}-height_map.png"
                output_path = os.path.join(output_folder, filename)

                plt.imsave(output_path, height_map, cmap="gray")

                # plt.show()

                saved_blocks += 1

    print(f"Done! Saved {saved_blocks} blocks to '{output_folder}'.")

if __name__ == '__main__':
    pass
