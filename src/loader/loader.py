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
Filename: loader.py
Directory: src/loader/
"""

import pathlib
from pathlib import Path
import torch
from tqdm import tqdm
from collections import defaultdict
import torchvision
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.loader import utils
from src.shared.utils import project_directory



class Loader(torch.utils.data.Dataset):
    def __init__(self, dataset_directory, height_map_size=(128, 128), satellite_map_size=(512, 512),
                 heatmap_size=(129, 129), max_data_size: int = float("inf"), shuffle_data=True, negative_pair_probability: float = 0.0,
                 sigma: float = 1.5, augment_data: bool = False, satellite_map_augment_max_angle=10, min_brightness_ratio=0.7,
                 max_brightness_ratio=1.3, min_contrast_ratio=0.7, max_contrast_ratio=1.3, min_saturation_ratio=0.7,
                 max_saturation_ratio=1.3, noise_ratio=0.01, heightmap_augment_max_angle=10, radial_black_pixels_prob=0.9,
                 radial_drop_prob=0.85, satellite_map_crop=False):

        self.height_map_size = height_map_size
        self.satellite_map_size = satellite_map_size
        self.heatmap_size = heatmap_size

        self.__height_map_suffix:str = 'height_map'
        self.__satellite_map_suffix:str = 'satellite_map'

        self.__return_height_map:bool = True        #TODO:add to self.getitem
        self.__return_satellite_map:bool = True     #TODO:add to self.getitem
        self.__return_heatmap:bool = True           #TODO:add to self.getitem
        self.__return_center_position:bool = True   #TODO:add to self.getitem
        self.__return_is_positive:bool = False      #TODO:add to self.getitem

        self.__augment_data:bool = augment_data
        self.__max_data_size:int = max_data_size
        self.__shuffle_data:bool = shuffle_data

        self.__north_up: bool = True

        self.__negative_pair_probability: float = negative_pair_probability
        self.__sigma: float = sigma

        self.__satellite_map_augment_max_angle:float = satellite_map_augment_max_angle
        self.__min_brightness_ratio: float = min_brightness_ratio
        self.__max_brightness_ratio: float = max_brightness_ratio
        self.__min_contrast_ratio: float = min_contrast_ratio
        self.__max_contrast_ratio: float = max_contrast_ratio
        self.__min_saturation_ratio: float = min_saturation_ratio
        self.__max_saturation_ratio: float = max_saturation_ratio
        self.__noise_ratio: float = noise_ratio

        self.__heightmap_augment_max_angle: float = heightmap_augment_max_angle
        self.__radial_black_pixels_prob: float = radial_black_pixels_prob
        self.__radial_drop_prob:float = radial_drop_prob

        self.__satellite_map_crop:bool = satellite_map_crop


        self.dataset_directory = dataset_directory

        self.pairs:list[dict[str, torch.Tensor]] = []

        self.get_all()

    def get_all(self):
        pair_filenames = self.__get_dataset_filenames_pairs()
        if self.__shuffle_data:
            random.shuffle(pair_filenames)

        for pair in pair_filenames:
            self.pairs.append(pair)

            if len(self.pairs) >= self.__max_data_size > 0:
                break

    def __get_dataset_filenames_pairs(self) -> list[dict[str, pathlib.Path]]:
        """ get dataset pairs """
        folder = Path(self.dataset_directory)
        files_by_number = defaultdict(list)

        for file in tqdm(folder.iterdir(), desc="Loading dataset filenames pairs", leave=False):
            filename = str(file.name)
            filename_id = filename.split('-')[0]    #FIXME: revert here
            #filename_id = filename.split('_')[0]    #FIXME: remove

            if file.is_file():
                if filename_id is not None and filename_id != "":
                    files_by_number[filename_id].append(file)
                else:
                    pass

        pairs = []
        for number, files in files_by_number.items():
            if len(files) == 2:
                height_map = None
                satellite_map = None
                if self.__height_map_suffix in files[0].name:
                    height_map = pathlib.Path(files[0])
                    satellite_map = pathlib.Path(files[1])
                elif self.__satellite_map_suffix in files[0].name:
                    satellite_map = pathlib.Path(files[0])
                    height_map = pathlib.Path(files[1])

                dat = {"height_map": height_map, "satellite_map": satellite_map}
                pairs.append(dat)

        return pairs

    @staticmethod
    def safe_crop_center(img_tensor, target_cx, target_cy, crop_size, padding_value=0): #TODO: rewrite
        _, H, W = img_tensor.shape
        half = crop_size // 2

        orig_center_x = W // 2
        orig_center_y = H // 2

        real_cx = orig_center_x - target_cx + half
        real_cy = orig_center_y - target_cy + half

        x1 = real_cx - half
        y1 = real_cy - half
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - W)
        pad_bottom = max(0, y2 - H)

        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant',
                               value=padding_value)
            x1 += pad_left
            y1 += pad_top

        return img_tensor[:, y1:y1 + crop_size, x1:x1 + crop_size]

    def __augment_heightmap_tensor(self, tensor: torch.Tensor):
        # random black pixels
        if random.random() < 0.3 and False:
            tensor = tensor.clone()

            mask = torch.rand(tensor.shape) < 0.3
            tensor[mask] = 0

        # angle augmentation
        angle = random.uniform(-self.__heightmap_augment_max_angle, self.__heightmap_augment_max_angle)
        tensor = torchvision.transforms.functional.rotate(tensor, angle,
                                                        interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR)

        # tiny gaussian noise augmentation
        noise = torch.randn_like(tensor) * self.__noise_ratio
        tensor = tensor + noise
        tensor = torch.clamp(tensor, 0, 1)

        # shift augmentation
        _, H, W = tensor.shape
        x = random.randint(0, int(self.height_map_size[0]) - 1)
        y = random.randint(0, int(self.height_map_size[1]) - 1)
        tensor = self.safe_crop_center(tensor, x, y, self.height_map_size[0])

        # radial random black pixels
        if random.random() < self.__radial_black_pixels_prob:
            tensor = tensor.clone()

            C, H, W = tensor.shape
            cx, cy = W / 2, H / 2

            grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

            distances = torch.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)

            max_distance = torch.sqrt(torch.tensor(cx ** 2 + cy ** 2))
            normalized_distances = distances / max_distance

            drop_probabilities = normalized_distances * self.__radial_drop_prob
            mask = torch.rand(H, W) < drop_probabilities
            tensor[0, mask] = 0.0

        return tensor, (x,y)


    def __augment_satellite_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # angle augmentation
        angle = random.uniform(-self.__satellite_map_augment_max_angle, self.__satellite_map_augment_max_angle)
        tensor = torchvision.transforms.functional.rotate(tensor, angle, interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR)

        # brightness & contrast jitter augmentation
        brightness = random.uniform(self.__min_brightness_ratio, self.__max_brightness_ratio)
        contrast = random.uniform(self.__min_contrast_ratio, self.__max_contrast_ratio)
        saturation = random.uniform(self.__min_saturation_ratio, self.__max_saturation_ratio)

        tensor = torchvision.transforms.functional.adjust_brightness(tensor, brightness)
        tensor = torchvision.transforms.functional.adjust_contrast(tensor, contrast)
        tensor = torchvision.transforms.functional.adjust_saturation(tensor, saturation)

        # tiny gaussian noise augmentation
        noise = torch.randn_like(tensor) * self.__noise_ratio
        tensor = tensor + noise
        tensor = torch.clamp(tensor, 0, 1)

        # shift augmentation
        if self.__satellite_map_crop:
            _, H, W = tensor.shape
            x = random.randint(0, int(self.satellite_map_size[0]/2) - 1)
            y = random.randint(0, int(self.satellite_map_size[1]/2) - 1)
            tensor = self.safe_crop_center(tensor, x, y, self.satellite_map_size[0])
        else:
            _, H, W = tensor.shape
            x,y = H // 2, W // 2

        return tensor, (x, y)

    def __get_pair(self, index: int):
        pair = self.pairs[index]
        is_positive_pair = None

        height_map_image = utils.get_grayscale_image_from_path(pair["height_map"])
        height_map_tensor = utils.img_to_tensor(height_map_image)

        if random.random() >= self.__negative_pair_probability:
            satellite_path = pair["satellite_map"]
            is_positive_pair = True
        else:
            satellite_index = index
            while index == satellite_index:
                satellite_index = random.randint(0, len(self.pairs) - 1)
            satellite_path = self.pairs[satellite_index]["satellite_map"]
            is_positive_pair = False

        satellite_map_image = utils.get_rgb_image_from_path(satellite_path)
        satellite_map_tensor = utils.img_to_tensor(satellite_map_image)

        return height_map_tensor, satellite_map_tensor, is_positive_pair

    @staticmethod
    def gaussian_heatmap(heatmap_size:tuple[int, int], center_position:tuple[int, int], sigma:float=0.1) -> torch.Tensor:
        y = torch.arange(heatmap_size[0]).view(-1, 1)
        x = torch.arange(heatmap_size[1]).view(1, -1)
        cx, cy = center_position
        return torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

    def __get_heatmap(self, center_position:tuple[int, int]=(None, None)):
        if center_position == (None, None):
            height, width = self.heatmap_size
            value_per_pixel = 1.0 / (height * width)
            gt_heatmap = torch.full((height, width), value_per_pixel)

            return gt_heatmap, (None, None)
        else:
            height, width = self.satellite_map_size
            cx, cy = center_position

            H_out, W_out = self.heatmap_size

            cx_out = int(cx * W_out / width)
            cy_out = int(cy * H_out / height)

            output_position = (cx_out, cy_out)

            gt_heatmap = self.gaussian_heatmap(self.heatmap_size, output_position, sigma=self.__sigma)
            gt_heatmap = gt_heatmap / (gt_heatmap.sum() + 1e-8)

            """if gt_heatmap.max() > 0:
                gt_heatmap = gt_heatmap / gt_heatmap.max()"""

            return gt_heatmap, output_position

    def __len__(self):
        return len(self.pairs)

    def transform_satellite_image(self, image:Image.Image) -> torch.Tensor:
        pass

    def __getitem__(self, index:int):
        height_map_tensor, satellite_map_tensor, is_positive_pair = self.__get_pair(index)

        if not is_positive_pair:
            if self.__augment_data:
                satellite_map_tensor, _ = self.__augment_satellite_tensor(satellite_map_tensor)
                height_map_tensor, _ = self.__augment_heightmap_tensor(height_map_tensor)

            heatmap_tensor, _ = self.__get_heatmap()

            return height_map_tensor, satellite_map_tensor, heatmap_tensor, (None, None)
        else:
            if self.__augment_data:
                orig_sat_H, orig_sat_W = satellite_map_tensor.shape[1:]
                orig_heig_H, orig_heig_W = height_map_tensor.shape[1:]

                scale_x = orig_sat_W / orig_heig_W
                scale_y = orig_sat_H / orig_heig_H

                #(sat_x, sat_y) = orig_sat_W/2, orig_sat_H/2
                satellite_map_tensor, (sat_x, sat_y) = self.__augment_satellite_tensor(satellite_map_tensor)
                height_map_tensor, (heig_x, heig_y) = self.__augment_heightmap_tensor(height_map_tensor)

                _, heig_H, heig_W = height_map_tensor.shape

                heig_c_x = heig_W // 2
                heig_c_y = heig_H // 2

                dx_lidar = heig_c_x - heig_x
                dy_lidar = heig_c_y - heig_y

                dx_sat = dx_lidar * scale_x
                dy_sat = dy_lidar * scale_y

                x = int(sat_x + dx_sat)
                y = int(sat_y + dy_sat)
            else:
                height, width = satellite_map_tensor.shape[1:]
                (x, y) = height//2, width//2

            heatmap_tensor, center_position = self.__get_heatmap(center_position=(x, y))

        lidar_min = height_map_tensor.min()
        lidar_max = height_map_tensor.max()
        height_map_tensor = (height_map_tensor - lidar_min) / (lidar_max - lidar_min + 1e-6)

        if satellite_map_tensor.max() > 1.0:
            satellite_map_tensor = satellite_map_tensor / 255.0

        return height_map_tensor, satellite_map_tensor, heatmap_tensor, center_position

def overlay_transparent(sat_img, overlay_img, scale=1.0, offset_x=0, offset_y=0, alpha_scale=0.3):
    numpy_image = np.array(sat_img)
    sat = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    if sat is None:
        raise RuntimeError("Satellite image failed to load")
    sat = cv2.cvtColor(sat, cv2.COLOR_BGR2RGB)

    numpy_image = np.array(overlay_img)
    overlay = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2BGRA)
    if overlay is None:
        raise RuntimeError("Overlay image failed to load")

    if overlay.shape[2] != 4:
        raise RuntimeError("Overlay must have 4 channels (RGBA)")

    ov_rgb = overlay[:, :, :3]
    ov_alpha = overlay[:, :, 3] / 255.0  # normalize 0–1

    if scale != 1.0:
        ov_rgb = cv2.resize(ov_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        ov_alpha = cv2.resize(ov_alpha, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    ov_alpha = np.clip(ov_alpha * alpha_scale, 0, 1)
    canvas = sat.copy().astype(np.float32)

    oh, ow = ov_rgb.shape[:2]
    sh, sw = sat.shape[:2]

    x1 = int(offset_x)
    y1 = int(offset_y)
    x2 = x1 + ow
    y2 = y1 + oh

    x1_c = max(0, x1)
    y1_c = max(0, y1)
    x2_c = min(sw, x2)
    y2_c = min(sh, y2)

    if x1_c < x2_c and y1_c < y2_c:
        ox1 = x1_c - x1
        oy1 = y1_c - y1
        ox2 = ox1 + (x2_c - x1_c)
        oy2 = oy1 + (y2_c - y1_c)

        rgb_roi = ov_rgb[oy1:oy2, ox1:ox2]
        alpha_roi = ov_alpha[oy1:oy2, ox1:ox2][:, :, None]

        canvas[y1_c:y2_c, x1_c:x2_c] = (
            canvas[y1_c:y2_c, x1_c:x2_c] * (1 - alpha_roi)
            + rgb_roi * alpha_roi
        )

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas.astype(np.uint8))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return canvas.astype(np.uint8)

def vis_plt_offset(sat_img, overlay_img, scale=1.0, alpha=0.6, offset_x=0, offset_y=0):
    numpy_image = np.array(sat_img)
    sat = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    numpy_image = np.array(overlay_img)
    hm = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2BGRA)

    if sat is None:
        raise RuntimeError("Satellite image failed to load")
    if hm is None:
        raise RuntimeError("Heightmap failed to load")

    sat = cv2.cvtColor(sat, cv2.COLOR_BGR2RGB)

    if hm.ndim == 2:  # grayscale
        hm_gray = hm
    elif hm.shape[2] == 3:  # RGB to grayscale
        hm_gray = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)
    elif hm.shape[2] == 4:  # RGBA to grayscale
        hm_gray = cv2.cvtColor(hm[:,:,:3], cv2.COLOR_BGR2GRAY)
    else:
        raise RuntimeError("Unsupported heightmap shape")

    hm_norm = cv2.normalize(hm_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    hm_red = np.zeros((hm_norm.shape[0], hm_norm.shape[1], 3), dtype=np.uint8)
    hm_red[:, :, 0] = hm_norm  # Red channel
    hm_red[:, :, 1] = 0        # Green
    hm_red[:, :, 2] = 0        # Blue

    if scale != 1.0:
        hm_red = cv2.resize(hm_red, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros_like(sat)

    h_hm, w_hm = hm_red.shape[:2]
    x1 = offset_x
    y1 = offset_y
    x2 = offset_x + w_hm
    y2 = offset_y + h_hm

    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(canvas.shape[1], x2)
    y2_clip = min(canvas.shape[0], y2)

    hm_x1 = x1_clip - x1
    hm_y1 = y1_clip - y1
    hm_x2 = hm_x1 + (x2_clip - x1_clip)
    hm_y2 = hm_y1 + (y2_clip - y1_clip)

    if x1_clip < x2_clip and y1_clip < y2_clip:
        canvas[y1_clip:y2_clip, x1_clip:x2_clip] = hm_red[hm_y1:hm_y2, hm_x1:hm_x2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(sat)              # background satellite
    ax.imshow(canvas, alpha=alpha)  # red heightmap overlay
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def show_imgs(sat_img, overlay_img, heat):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

    ax1.imshow(overlay_img, cmap='gray')

    ax2.imshow(sat_img)

    ax3.imshow(heat, cmap='jet')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    directory = project_directory(Path("data/dataset"))

    dataset_dir = directory / "processed/test-urban"

    dst = Loader(dataset_dir, negative_pair_probability=0.0, max_data_size=100, augment_data=True, shuffle_data=False, heatmap_size=(65,65), satellite_map_size=(256, 256), height_map_size=(128, 128), radial_black_pixels_prob=0.5, heightmap_augment_max_angle=0, satellite_map_augment_max_angle=0)

    for height_map_tensor, satellite_map_tensor, heatmap_tensor, center_position in dst:
        heightmap_image = utils.tensor_to_rgb_image(height_map_tensor)
        satellite_map_image = utils.tensor_to_rgb_image(satellite_map_tensor)
        heatmap_img = utils.tensor_to_rgb_image(heatmap_tensor)

        print(heightmap_image.size, satellite_map_image.size, center_position)
        show_imgs(satellite_map_image, heightmap_image, heatmap_img)
        #vis_plt_offset(satellite_map_image, heightmap_image,4,alpha=0.3)
        #overlay_transparent(satellite_map_image, heightmap_image,4)
