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
Filename: 05_test_localizer.py
Directory: scripts/
"""

import math
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path


from src.localizer.model import GlobalLocalizationNet
from src.loader.loader import Loader
from src.shared.utils import project_directory


def evaluate_dataset_metrics(model, dataset, device):
    print("\n--- Running mass evaluation of dataset ---")
    model.eval()

    errors = []

    sat_H, sat_W = dataset.satellite_map_size
    hm_H, hm_W = dataset.heatmap_size

    scale_x = sat_W / hm_W
    scale_y = sat_H / hm_H

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Image evaluation"):
            lidar_tensor, sat_tensor, gt_heatmap_tensor, center_position = dataset[i]

            real_x_hm, real_y_hm = center_position
            if real_x_hm is None:
                continue

            sat_batch = sat_tensor.unsqueeze(0).to(device)
            lidar_batch = lidar_tensor.unsqueeze(0).to(device)

            heatmap_logits = model(lidar_batch, sat_batch)
            B, out_H, out_W = heatmap_logits.shape
            heatmap_prob = F.softmax(heatmap_logits.view(B, -1), dim=1).view(B, out_H, out_W)

            pred_y_hm, pred_x_hm = torch.unravel_index(heatmap_prob.view(-1).argmax(), (out_H, out_W))
            pred_x_hm = pred_x_hm.item()
            pred_y_hm = pred_y_hm.item()

            pred_x_sat = pred_x_hm * scale_x
            pred_y_sat = pred_y_hm * scale_y

            real_x_sat = real_x_hm * scale_x
            real_y_sat = real_y_hm * scale_y

            error = math.hypot(pred_x_sat - real_x_sat, pred_y_sat - real_y_sat)
            errors.append(error)

    if not errors:
        print("Dataset do not contain any positive pairs.")
        return

    errors_arr = np.array(errors)

    best_err = np.min(errors_arr)
    worst_err = np.max(errors_arr)
    mean_err = np.mean(errors_arr)
    median_err = np.median(errors_arr)

    recall_1 = np.mean(errors_arr <= 1.0) * 100
    recall_5 = np.mean(errors_arr <= 5.0) * 100
    recall_10 = np.mean(errors_arr <= 10.0) * 100
    recall_50 = np.mean(errors_arr <= 50.0) * 100

    print("\n--- EVALUATION RESULTS (error in pixels on satmap) ---")
    print(f"Number of test pairs: {len(errors_arr)}")
    print(f"Best                : {best_err:.2f} px")
    print(f"Worst               : {worst_err:.2f} px")
    print(f"Mean                : {mean_err:.2f} px")
    print(f"Median              : {median_err:.2f} px")
    print("---------------------------------------------------------")
    print(f"Recall @ 1 px       : {recall_1:.2f} %")
    print(f"Recall @ 5 px       : {recall_5:.2f} %")
    print(f"Recall @ 10 px      : {recall_10:.2f} %")
    print(f"Recall @ 50 px      : {recall_50:.2f} %")
    print("---------------------------------------------------------\n")

def main(model_weights_path):
    directory = project_directory(Path("data"))
    test_dir = directory / "dataset/processed/test/"

    print("Loading dataset...")
    test_dataset = Loader(
        dataset_directory=test_dir,
        height_map_size=(128, 128),
        satellite_map_size=(256, 256),
        heatmap_size=(65, 65),
        max_data_size=10,
        shuffle_data=True,
        augment_data=True,
        negative_pair_probability=0.5,
        radial_black_pixels_prob=1,
        satellite_map_augment_max_angle=0,
        heightmap_augment_max_angle=0
    )

    if len(test_dataset) == 0:
        print("Error: Folder with test dataset does not contain any pairs.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = GlobalLocalizationNet(embed_dim=128).to(device)

    if model_weights_path.exists():
        checkpoint = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Model weights loaded.")
    else:
        print("Warning: model .pth not found. Using random weights.")

    model.eval()

    evaluate_dataset_metrics(model, test_dataset, device)

    sample_index = 0
    print(f"Processing index {sample_index}...")

    lidar_tensor, sat_tensor, gt_heatmap_tensor, center_position = test_dataset[sample_index]

    real_x_hm, real_y_hm = center_position
    if real_x_hm is None:
        real_x_hm, real_y_hm = 0,0

    sat_batch = sat_tensor.unsqueeze(0).to(device)
    lidar_batch = lidar_tensor.unsqueeze(0).to(device)

    print("Running prediction...")
    with torch.no_grad():
        heatmap_logits = model(lidar_batch, sat_batch)
        B, hm_H, hm_W = heatmap_logits.shape
        heatmap_prob = F.softmax(heatmap_logits.view(B, -1), dim=1).view(B, hm_H, hm_W)

    pred_y_hm, pred_x_hm = torch.unravel_index(heatmap_prob.view(-1).argmax(), (hm_H, hm_W))
    pred_x_hm = pred_x_hm.item()
    pred_y_hm = pred_y_hm.item()

    print(heatmap_prob.max().item()*100,"%")

    sat_W = test_dataset.satellite_map_size[1]
    sat_H = test_dataset.satellite_map_size[0]

    scale_x = sat_W / hm_W
    scale_y = sat_H / hm_H

    pred_x_sat = pred_x_hm * scale_x
    pred_y_sat = pred_y_hm * scale_y

    scale_x = sat_W / hm_W
    scale_y = sat_H / hm_H
    real_x = real_x_hm * scale_x
    real_y = real_y_hm * scale_y

    error_distance = ((pred_x_sat - real_x) ** 2 + (pred_y_sat - real_y) ** 2) ** 0.5

    print("Showing outputs...")

    sat_viz = sat_tensor.cpu().permute(1, 2, 0).numpy()
    lidar_viz = lidar_tensor.squeeze().cpu().numpy()
    heatmap_viz = heatmap_prob.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"#{sample_index} - Error: {error_distance:.2f} px (on satmap)",
                 fontsize=14, fontweight='bold')

    axes[0].imshow(lidar_viz, cmap='gray')
    #axes[0].set_title(f"LiDAR {test_dataset.height_map_size}")
    axes[0].axis('off')

    axes[1].imshow(sat_viz)
    axes[1].plot(real_x, real_y, 'go', markersize=10, markerfacecolor='none', markeredgewidth=2,
                 label="GT")
    axes[1].plot(pred_x_sat, pred_y_sat, 'rx', markersize=10, markeredgewidth=2, label="Prediction")
    #axes[1].set_title(f"Satellite {test_dataset.satellite_map_size}")
    axes[1].axis('off')
    axes[1].legend()

    #im = axes[2].imshow(heatmap_viz, cmap='jet', vmin=0, vmax=0.1)
    im = axes[2].imshow(heatmap_viz, cmap='jet')
    #axes[2].plot(real_x_hm, real_y_hm, 'go', markersize=10, markerfacecolor='none', markeredgewidth=2, label="GT")
    #axes[2].plot(pred_x_hm, pred_y_hm, 'rx', markersize=12, markeredgewidth=2, label="Prediction")
    #axes[2].set_title(f"Heatmap ({hm_W}x{hm_H})")
    axes[2].axis('off')
    #axes[2].legend()

    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='Probability')

    plt.tight_layout()


    """fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_trans = fig.dpi_scale_trans.inverted()

    bbox = axes[0].get_tightbbox(renderer).transformed(inv_trans)
    #fig.savefig(directory/f"results/heightmap-{test_dir.name}-axes.png", bbox_inches=bbox, dpi=300)
    bbox = axes[1].get_tightbbox(renderer).transformed(inv_trans)
    fig.savefig(directory/f"results/ortophoto-{test_dir.name}-axes.png", bbox_inches=bbox, dpi=300)
    bbox = axes[2].get_tightbbox(renderer).transformed(inv_trans)
    #fig.savefig(directory/f"results/heatmap-{test_dir.name}-axes.png", bbox_inches=bbox, dpi=300)

    plt.imsave(directory/f"results/heightmap-{test_dir.name}.png", lidar_viz, cmap='gray')
    plt.imsave(directory/f"results/ortophoto-{test_dir.name}.png", sat_viz)
    plt.imsave(directory/f"results/heatmap-{test_dir.name}.png", heatmap_viz, cmap='jet', vmin=0, vmax=0.1)"""

    plt.show()

if __name__ == '__main__':
    # Change this from model_0.pth to model_1.pth for testing new trained model
    model_path = project_directory(Path("data")) / "localizer/models/model_0.pth"
    main(model_path)
