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
Filename: ranker.py
Directory: src/localizer/
"""

import torch
import torch.nn.functional as F


def heatmap_kl_loss(pred_logits, gt_heatmap):
    B = pred_logits.shape[0]

    pred_logprob = F.log_softmax(
        pred_logits.view(B, -1), dim=1
    )

    gt = gt_heatmap.view(B, -1)

    return F.kl_div(pred_logprob, gt, reduction="batchmean")

def calculate_distance(pred_heatmap, x,y):
    valid_mask = x >= 0
    if valid_mask.sum() == 0:
        return 0.0

    pred_valid = pred_heatmap[valid_mask]
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    B, H, W = pred_valid.shape
    prob = F.softmax(pred_valid.view(B, -1), dim=1).view(B, H, W)

    pred_flat_idx = prob.view(B, -1).argmax(dim=1)
    y_max, x_max = torch.unravel_index(pred_flat_idx, (H, W))

    distance = ((x_valid-x_max.cpu()) ** 2 + (y_valid-y_max.cpu()) ** 2) ** 0.5

    return distance.sum().item()


def calculate_recall_at_radius(pred_heatmap, gt_heatmap, radius_px=5, gt_x=None):
    B = pred_heatmap.shape[0]

    if gt_x is not None:
        valid_mask = gt_x >= 0

        if valid_mask.sum() == 0:
            return 0.0

        pred_heatmap = pred_heatmap[valid_mask]
        gt_heatmap = gt_heatmap[valid_mask]

        B = pred_heatmap.shape[0]

    _, H, W = pred_heatmap.shape

    pred_flat = pred_heatmap.view(B, -1)
    gt_flat = gt_heatmap.view(B, -1)

    pred_idx = pred_flat.argmax(dim=1)
    gt_idx = gt_flat.argmax(dim=1)

    py, px = pred_idx // W, pred_idx % W
    gy, gx = gt_idx // W, gt_idx % W

    dists = torch.sqrt((px - gx).float() ** 2 + (py - gy).float() ** 2)

    correct = (dists <= radius_px).float().sum().item()

    return correct / B

def calculate_recall_at_radius_old(pred_heatmap, gt_heatmap, radius_px=5):
    B, H, W = pred_heatmap.shape

    pred_flat = pred_heatmap.view(B, -1)
    gt_flat = gt_heatmap.view(B, -1)

    pred_idx = pred_flat.argmax(dim=1)
    gt_idx = gt_flat.argmax(dim=1)

    py, px = pred_idx // W, pred_idx % W
    gy, gx = gt_idx // W, gt_idx % W

    dists = torch.sqrt((px - gx).float() ** 2 + (py - gy).float() ** 2)

    correct = (dists <= radius_px).float().sum().item()

    return correct / B

def calculate_topk_accuracy(pred_heatmap, gt_heatmap, k=5, gt_x=None):
    B = pred_heatmap.shape[0]

    if gt_x is not None:
        valid_mask = gt_x >= 0

        if valid_mask.sum() == 0:
            return 0.0

        pred_heatmap = pred_heatmap[valid_mask]
        gt_heatmap = gt_heatmap[valid_mask]

        B = pred_heatmap.shape[0]

    pred_flat = pred_heatmap.view(B, -1)
    gt_flat = gt_heatmap.view(B, -1)

    gt_indices = gt_flat.argmax(dim=1)

    _, topk_indices = pred_flat.topk(k, dim=1, largest=True, sorted=True)

    correct = topk_indices.eq(gt_indices.view(B, 1))

    correct_count = correct.sum().float().item()

    return correct_count / B


def modified_focal_loss(pred_heatmap, gt_heatmap, alpha=2, beta=4):
    pred_heatmap = torch.clamp(pred_heatmap, min=1e-6, max=1 - 1e-6)

    pos_inds = gt_heatmap.ge(0.99).float()
    neg_inds = gt_heatmap.lt(0.99).float()

    neg_weights = torch.pow(1 - gt_heatmap, beta)

    pos_loss = torch.log(pred_heatmap) * torch.pow(1 - pred_heatmap, alpha) * pos_inds
    neg_loss = torch.log(1 - pred_heatmap) * torch.pow(pred_heatmap, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss

if __name__ == '__main__':
    pass

