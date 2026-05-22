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
Directory: src/embedder/
"""


from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from src.shared.utils import project_directory


def custom_collate(batch):
    lidar_images = torch.stack([item[0] for item in batch])
    sat_images = torch.stack([item[1] for item in batch])

    ignored_vars = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    return lidar_images, sat_images, ignored_vars, labels

def test_retrieval(model, dataloader, device, top_k=3, reference_id=2):
    model.eval()
    model = model.to(device)

    sat_embeddings_db = []
    sat_images_db = []

    print("1. Generating vectors of satellite images...")
    with torch.no_grad():
        for batch_idx, (lidar_img, sat_img, _, _) in enumerate(tqdm(dataloader)):
            sat_img = sat_img.to(device, dtype=torch.float32)

            sat_emb = model.sat_encoder(sat_img)
            sat_emb = F.normalize(sat_emb, p=2, dim=1)

            sat_embeddings_db.append(sat_emb.cpu())
            sat_images_db.append(sat_img.cpu())

            if batch_idx == 0:
                real_embedding = sat_embeddings_db[0][reference_id]

    sat_embeddings_db = torch.cat(sat_embeddings_db, dim=0)
    sat_images_db = torch.cat(sat_images_db, dim=0)
    print(f"Database has {sat_embeddings_db.shape[0]} satellite vectors.")

    print("\n2. Selecting lidar image")
    test_batch = next(iter(dataloader))
    query_lidar_img = test_batch[0][reference_id].unsqueeze(0).to(device, dtype=torch.float32)
    query_sat_img = test_batch[1][reference_id].to(device, dtype=torch.float32)

    with torch.no_grad():
        query_emb = model.lidar_encoder(query_lidar_img)
        query_emb = F.normalize(query_emb, p=2, dim=1).cpu()

    print("3. Search for closest image...")
    similarities = torch.matmul(sat_embeddings_db, query_emb.T).squeeze()

    show_plot(similarities)

    rank, score = find_rank_of_index(similarities, target_index=reference_id)
    print(f"Ground-truth (index {reference_id}) is at {rank}. place from {len(similarities)} with score {score:.4f}")

    best_scores, best_indices = torch.topk(similarities, k=top_k)
    real_score = torch.matmul(real_embedding, query_emb.T).squeeze()

    print(f"Best scores: {best_scores.tolist()}")

    return query_lidar_img.cpu(), sat_images_db, best_indices, best_scores, query_sat_img.cpu(), real_score


def show_plot(similarities):
    import matplotlib.pyplot as plt
    import numpy as np
    data = similarities.detach().cpu().numpy()

    min_val = np.floor(data.min() * 10) / 10
    max_val = np.ceil(data.max() * 10) / 10
    bins = 0.01

    my_bins = np.arange(min_val, max_val + bins, bins)


    plt.figure(figsize=(10, 6))

    plt.hist(data, bins=my_bins, edgecolor='black', color='skyblue')

    plt.title('Score distribution of images similarities', fontsize=14)
    plt.xlabel('Similarity score', fontsize=12)
    plt.ylabel('Number of images', fontsize=12)

    plt.xticks(my_bins)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


import torch


def find_rank_of_index(similarities, target_index):
    sorted_indices = torch.argsort(similarities, descending=True)
    place = (sorted_indices == target_index).nonzero(as_tuple=True)[0].item()
    rank = place + 1
    skore = similarities[target_index].item()

    return rank, skore

def visualize_results(query_lidar, sat_db, best_indices, best_scores, query_sat, real_score):
    top_k = len(best_indices)

    num_plots = top_k + 2
    cols = 8
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    lidar_2d = query_lidar[0, 0, :, :].numpy()

    number = 4

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_trans = fig.dpi_scale_trans.inverted()

    directory = project_directory(Path("data/results/embedder"))

    axes[0].imshow(lidar_2d, cmap='gray')
    axes[0].axis('off')
    bbox = axes[0].get_tightbbox(renderer).transformed(inv_trans)
    fig.savefig(directory/f"heightmap-{number}.png", bbox_inches=bbox, dpi=300)

    axes[1].imshow(query_sat.permute(1, 2, 0).numpy())
    axes[1].axis('off')
    bbox = axes[1].get_tightbbox(renderer).transformed(inv_trans)
    fig.savefig(directory/f"gt-ortophoto-{number}.png", bbox_inches=bbox, dpi=300)

    for i, idx in enumerate(best_indices):
        score = best_scores[i].item()
        sat_img = sat_db[idx]

        sat_show = sat_img.permute(1, 2, 0).numpy()
        sat_show = sat_show.clip(0, 1)

        axes[i + 2].imshow(sat_show)
        axes[i + 2].axis('off')

        if i<3:
            bbox = axes[i + 2].get_tightbbox(renderer).transformed(inv_trans)
            fig.savefig(directory/f"gt-ortophoto_{i}-{number}.png", bbox_inches=bbox, dpi=300)

    plt.tight_layout()
    plt.show()


def evaluate_entire_dataset(model, dataloader, device, top_k_list=[1, 5, 10, 50]):
    model.eval()
    model = model.to(device)

    lidar_embeddings_db = []
    sat_embeddings_db = []
    from tqdm import tqdm

    print("1. Generating vectors for all Lidar and satellite images...")
    with torch.no_grad():
        for lidar_img, sat_img, _, _ in tqdm(dataloader, desc="Testing", leave=False):
            lidar_img = lidar_img.to(device, dtype=torch.float32)
            sat_img = sat_img.to(device, dtype=torch.float32)

            sat_emb = model.sat_encoder(sat_img)
            sat_emb = F.normalize(sat_emb, p=2, dim=1)

            lidar_emb = model.lidar_encoder(lidar_img)
            lidar_emb = F.normalize(lidar_emb, p=2, dim=1)

            sat_embeddings_db.append(sat_emb.cpu())
            lidar_embeddings_db.append(lidar_emb.cpu())

    sat_embeddings = torch.cat(sat_embeddings_db, dim=0)
    lidar_embeddings = torch.cat(lidar_embeddings_db, dim=0)

    N = sat_embeddings.shape[0]
    print(f"Totally loaded {N} pairs.")

    print("2. Calculating similarity matrix...")
    similarities = torch.matmul(lidar_embeddings, sat_embeddings.T)

    print("3. Evaluating ranks and metrics...\n")
    ranks = []

    for i in range(N):
        sorted_indices = torch.argsort(similarities[i], descending=True)

        position = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        rank = position + 1
        ranks.append(rank)

    ranks = np.array(ranks)

    best_rank = ranks.min()
    worst_rank = ranks.max()
    mean_rank = ranks.mean()
    median_rank = np.median(ranks)

    print("-" * 30)
    print(" RESULTS OF RANKS")
    print("-" * 30)
    print(f"Best rank: {best_rank}")
    print(f"Worst rank: {worst_rank}")
    print(f"Mean rank: {mean_rank:.2f}")
    print(f"Median ranku:  {median_rank}")

    print("\n" + "-" * 30)
    print(" TOP-K ACCURACY (Recall@K)")
    print("-" * 30)
    for k in top_k_list:
        correct_in_k = np.sum(ranks <= k)
        accuracy = (correct_in_k / N) * 100
        print(f"Top-{k:<2}: {accuracy:>6.2f} %  ({correct_in_k}/{N})")

    return ranks


if __name__ == '__main__':
    pass

