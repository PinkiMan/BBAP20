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
Filename: test.py
Directory: src/localizer/
"""

import pathlib
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

from src.loader.loader import Loader
from src.localizer.model import GlobalLocalizationNet
from src.localizer.ranker import calculate_distance, calculate_topk_accuracy, calculate_recall_at_radius
from src.shared.utils import project_directory
from src.loader import utils


def show_custom_overlay(img_sat_pil, heatmap_logits, vmin, vmax, alpha=0.5, cmap='jet', cx=0, cy=0, ID=None):
    img_np = np.array(img_sat_pil)
    h, w = img_np.shape[:2]

    heatmap_resized = cv2.resize(heatmap_logits, (w, h), interpolation=cv2.INTER_CUBIC)

    heatmap_masked = np.ma.masked_where(heatmap_resized < vmin, heatmap_resized)

    fig,ax = plt.subplots(1)
    from matplotlib.patches import Circle
    circ = Circle((cx, cy), 16, color='r', fill=False)
    ax.add_patch(circ)

    #plt.figure(figsize=(8, 8))

    plt.imshow(img_np)

    plt.imshow(heatmap_masked, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)

    plt.title(f"Overlay (vmin={vmin:.2f}, vmax={vmax:.2f})")
    plt.colorbar(fraction=0.046, pad=0.04)  # Zobrazí legendu barev
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def gaussian_heatmap(h, w, cx, cy, sigma=0.1):
    y = torch.arange(h).view(-1, 1)
    x = torch.arange(w).view(1, -1)
    return torch.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))

def get_heatmap(cx, cy, W, H, sigma):
    H_out, W_out = 129, 129

    cx_out = cx * W_out / W
    cy_out = cy * H_out / H

    sigma_out = sigma

    gt_heatmap = gaussian_heatmap(H_out, W_out, cx_out, cy_out, sigma=sigma_out)
    gt_heatmap = gt_heatmap / (gt_heatmap.sum() + 1e-8)
    """if gt_heatmap.max() > 0:
        gt_heatmap = gt_heatmap / gt_heatmap.max()"""

    return gt_heatmap, (cx_out, cy_out)

def vis_plt_offset(sat_img, overlay_img, scale=1.0, alpha=0.7, offset_x=0, offset_y=0, center=(0,0)):
    numpy_image = np.array(sat_img)
    sat = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    numpy_image = np.array(overlay_img)
    hm = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2BGRA)


    if sat is None:
        raise RuntimeError("Satellite image failed to load")
    if hm is None:
        raise RuntimeError("Heightmap failed to load")

    size = sat.shape[0]

    hm = cv2.resize(hm, (size, size), interpolation=cv2.INTER_NEAREST)

    sat = cv2.cvtColor(sat, cv2.COLOR_BGR2RGB)

    # Convert heightmap into RGB if needed
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
    hm_red[:, :, 0] = hm_norm        # Red channel
    hm_red[:, :, 1] = 0        # Green
    hm_red[:, :, 2] = hm_norm        # Blue

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
    from matplotlib.patches import Circle
    circ = Circle(center, 16, color='r', fill=False)
    ax.add_patch(circ)
    plt.tight_layout()
    plt.show()

def show_imgs(satellite_map_tensor, height_map_tensor, heatmap_tensor, center_position, device, model):
    scale = satellite_map_tensor.size(1) / heatmap_tensor.size(0)
    out_center_position = [center_position[0] * scale, center_position[1] * scale]

    # satellite_map_tensor = satellite_map_tensor[0:3,0:384,0:384]
    sat_img = utils.tensor_to_rgb_image(satellite_map_tensor)
    h_map = utils.tensor_to_rgb_image(height_map_tensor)
    gt_heatmap = utils.tensor_to_rgb_image(heatmap_tensor)

    height_map_tensor.unsqueeze_(0)
    satellite_map_tensor.unsqueeze_(0)

    h_img = height_map_tensor.to(device)
    s_img = satellite_map_tensor.to(device)

    with torch.no_grad():
        logits = model(h_img, s_img)

    sH, sW = satellite_map_tensor.shape[2:]

    B, H, W = logits.shape
    prob = F.softmax(logits.view(B, -1), dim=1).view(B, H, W)
    print(f"Max prob: {prob[0].max().item() * 100}")
    print(f"Mean prob: {prob[0].mean().item() * 100}")
    print(f"Median prob: {prob[0].median().item() * 100}")
    print(f"Sum prob: {prob[0].sum().item() * 100}")
    print(f"Max prob reduced: {100 * prob[0].max().item() / (prob[0].sum().item() - W * H * prob[0].mean().item())}")

    y_max, x_max = torch.unravel_index(prob[0].argmax(), (H, W))
    print(f"Max location: x={x_max.item()}, y={y_max.item()}")
    print(f"Max location: x={sW * x_max.item() / W}, y={sH * y_max.item() / H}")

    img_lidar = T.ToPILImage()(h_img[0].cpu())
    img_sat = T.ToPILImage()(s_img[0].cpu())

    heatmap_logits = logits[0].cpu().numpy()
    heatmap_prob = prob[0].cpu().numpy()

    fig, axs = plt.subplots(1, 5, figsize=(15, 4))

    axs[0].imshow(img_lidar, cmap='gray')
    axs[0].set_title("LiDAR Input")

    axs[1].imshow(img_sat)
    axs[1].set_title("Satellite Input")

    axs[2].imshow(gt_heatmap, cmap='viridis')
    axs[2].set_title("Ground Truth")

    h_max = heatmap_logits.max()
    h_avg = heatmap_logits.mean()
    # h_avg = 6e+3
    new = heatmap_logits[16:129 - 16, 16:129 - 16].mean()

    axs[3].imshow(heatmap_logits, cmap='jet', vmin=new, vmax=h_max)
    axs[3].set_title("Logits (Raw Output)")

    axs[4].imshow(heatmap_prob, cmap='magma')
    axs[4].set_title("Probability (Softmax)")

    plt.tight_layout()
    vis_plt_offset(sat_img, heatmap_prob, center=out_center_position)
    plt.show()

def test_net():
    dataset = project_directory()
    with open(dataset/'configs/localizer.yaml', 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    embedding_dim = cfg['model_parameters']['embedding_dim']
    model_path = dataset/cfg['model']['dir_name']/cfg['model']['model_path']
    model_path = dataset / cfg['model']['dir_name'] / pathlib.Path("model_1.pth")
    dataset_dir = dataset/cfg['directory']['pair_dir']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GlobalLocalizationNet(embed_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Model loaded.")

    dst = Loader(dataset/pathlib.Path("data/processed/pairs_test"), max_data_size=100, augment_data=False)

    for height_map_tensor, satellite_map_tensor, heatmap_tensor, center_position in dst:
        center_position = (32, 32)
        show_imgs(satellite_map_tensor, height_map_tensor, heatmap_tensor, center_position, device, model)

def test_solo():
    dataset = project_directory()
    with open(dataset/'configs/localizer.yaml', 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    embedding_dim = cfg['model_parameters']['embedding_dim']
    model_path = dataset/cfg['model']['dir_name']/cfg['model']['model_path']
    dataset_dir = dataset/cfg['directory']['pair_dir']


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GlobalLocalizationNet(embed_dim=embedding_dim).to(device)

    #model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    print("Model loaded")


    dst = Loader(dataset/pathlib.Path("data/pairs_test"), max_data_size=1)

    for height_map_tensor, satellite_map_tensor, heatmap_tensor, center_position in dst:
        print("")
        break

    from src.loader import utils
    satellite_map_tensor = satellite_map_tensor[0:3,0:384,0:384]
    sat_img = utils.tensor_to_rgb_image(satellite_map_tensor)
    h_map = utils.tensor_to_rgb_image(height_map_tensor)

    img_size = 512

    cx, cy = 64, 0   # [-256; 256]
    """cx = random.randint(-128, 128)
    cy = random.randint(-128, 128)"""
    #cx, cy = 0, 0
    start_x, start_y = img_size//4 - cx , img_size//4 - cy
    sat_img = sat_img.crop((start_x, start_y, start_x + img_size//2, start_y + img_size//2))

    H, W = satellite_map_tensor.shape[1:]

    gt_heatmap, (cx_out, cy_out) = get_heatmap(cx, cy, W, H, 1.5)

    height_map_tensor.unsqueeze_(0)
    satellite_map_tensor.unsqueeze_(0)

    h_img = height_map_tensor.to(device)
    s_img = satellite_map_tensor.to(device)
    gt_heatmap = gt_heatmap.to(device)

    with torch.no_grad():
        logits = model(h_img, s_img)

    sH, sW = satellite_map_tensor.shape[2:]

    B, H, W = logits.shape
    prob = F.softmax(logits.view(B, -1), dim=1).view(B, H, W)
    print(f"Max prob: {prob[0].max().item() *100}")
    print(f"Mean prob: {prob[0].mean().item() *100}")
    print(f"Median prob: {prob[0].median().item() *100}")
    print(f"Sum prob: {prob[0].sum().item() *100}")
    print(f"Max prob reduced: {100*prob[0].max().item()/(prob[0].sum().item()-W*H*prob[0].mean().item())}")

    print(f"True location: x={cx_out}, y={cy_out}")
    print(f"True location: x={cx_out/128*512}, y={cy_out/128*512}")

    y_max, x_max = torch.unravel_index(prob[0].argmax(), (H, W))
    print(f"Max location: x={x_max.item()}, y={y_max.item()}")
    print(f"Max location: x={sW*x_max.item()/W}, y={sH*y_max.item()/H}")

    distance = calculate_distance(logits, torch.tensor([cx_out]), torch.tensor([cy_out]))
    print(distance)

    top1_acc = calculate_topk_accuracy(logits, gt_heatmap, k=1)
    top5_acc = calculate_topk_accuracy(logits, gt_heatmap, k=5)
    top10_acc = calculate_topk_accuracy(logits, gt_heatmap, k=10)
    print(
        f"Top-1 Acc: {top1_acc * 100:.2f}% | Top-5 Acc: {top5_acc * 100:.2f}% | Top-10 Acc: {top10_acc * 100:.2f}%")

    rad_3 = calculate_recall_at_radius(logits, gt_heatmap, radius_px=3)
    rad_5 = calculate_recall_at_radius(logits, gt_heatmap, radius_px=5)
    rad_10 = calculate_recall_at_radius(logits, gt_heatmap, radius_px=10)
    print(f"Rad-3 Acc: {rad_3 * 100:.2f}% | Rad-5 Acc: {rad_5 * 100:.2f}% | Rad-10 Acc: {rad_10 * 100:.2f}%")

    img_lidar = T.ToPILImage()(h_img[0].cpu())
    img_sat = T.ToPILImage()(s_img[0].cpu())

    heatmap_gt = gt_heatmap.cpu().numpy()
    heatmap_logits = logits[0].cpu().numpy()
    heatmap_prob = prob[0].cpu().numpy()

    fig, axs = plt.subplots(1, 5, figsize=(15, 4))

    axs[0].imshow(img_lidar, cmap='gray')
    axs[0].set_title("LiDAR Input")

    axs[1].imshow(img_sat)
    axs[1].set_title("Satellite Input")

    axs[2].imshow(heatmap_gt, cmap='viridis')
    axs[2].set_title("Ground Truth")


    h_max = heatmap_logits.max()
    h_avg = heatmap_logits.mean()
    #h_avg = 6e+3
    new = heatmap_logits[16:129-16, 16:129-16].mean()

    axs[3].imshow(heatmap_logits, cmap='jet', vmin=new, vmax=h_max)
    axs[3].set_title("Logits (Raw Output)")

    axs[4].imshow(heatmap_prob, cmap='magma')
    axs[4].set_title("Probability (Softmax)")


    plt.tight_layout()
    #plt.show()

    #show_heatmap_overlay(img_sat, heatmap_logits, alpha=0.6)
    #show_custom_overlay(sat_img, heatmap_logits, vmin=new, vmax=h_max, alpha=0.4)

    cx = cx_out / 128 * 512
    cy = cy_out / 128 * 512

    show_custom_overlay(sat_img, heatmap_prob, vmin=heatmap_prob.mean(), vmax=heatmap_prob.max(), alpha=0.4, cmap='jet', cx=cx, cy=cy)


if __name__ == '__main__':
    #test_solo()
    test_net()

