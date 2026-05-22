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
Filename: utils.py
Directory: src/loader/
"""

import pathlib
import torch
import torchvision
from PIL import Image

def get_rgb_image_from_path(path: pathlib.Path) -> Image.Image:
    """ get rgb image from path """
    return Image.open(path).convert("RGB")

def get_grayscale_image_from_path(path: pathlib.Path) -> Image.Image:
    """ get grayscale image from path """
    return Image.open(path).convert("L")

def img_to_tensor(img: Image.Image) -> torch.Tensor:
    """ convert image to tensor """
    return torchvision.transforms.ToTensor()(img)

def tensor_resize(tensor: torch.Tensor, size: tuple) -> torch.Tensor:
    """ resize tensor to size """
    return torchvision.transforms.Resize(size)(tensor)

def tensor_to_rgb_image(tensor: torch.Tensor) -> Image.Image:
    """ convert tensor to rgb image """
    return torchvision.transforms.ToPILImage()(tensor)

if __name__ == '__main__':
    pass
