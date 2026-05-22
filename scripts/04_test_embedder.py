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
Filename: 04_test_embedder.py
Directory: scripts/
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.loader.loader import Loader
from src.embedder.model import CrossModalNetwork, CrossModalNetwork2, CrossModalNetwork3
from src.embedder.test import custom_collate, test_retrieval, visualize_results
from src.shared.utils import project_directory

def main(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    directory = project_directory(Path("data"))

    model = CrossModalNetwork3(embedding_dim=2048)
    model.load_state_dict(torch.load(model_path, map_location=device))

    dataset = Loader(directory/"dataset/processed/test", augment_data=True)

    test_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate
    )

    query_img, all_sat_imgs, best_idx, scores, query_sat, real_score = test_retrieval(
        model,
        test_loader,
        device,
        top_k=22,
        reference_id=0
    )

    print(len(all_sat_imgs))
    visualize_results(query_img, all_sat_imgs, best_idx, scores, query_sat, real_score)


if __name__ == '__main__':
    # Change this from model_0.pth to model_1.pth for testing new trained model
    model_path = project_directory(Path("data"))/"embedder/models/model_0.pth"
    main(model_path)
