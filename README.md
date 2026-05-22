# GNSS-denied Localization via Cross-modal Matching of Height Maps and Aerial Imagery

This repository contains the source code for a bachelor's thesis focused on the visual geo-localization of Unmanned Aerial Vehicles (UAVs) in GNSS-denied environments. The proposed system utilizes a two-phase (Coarse-to-Fine) architecture to fuse geometric data (LiDAR elevation maps) with visual priors (satellite RGB orthophotos).

## Main Features
* **Automated Dataset Pipeline:** Built-in scripts for fully automated raw data downloading (from Bayern Open Data), spatial alignment, cropping, and dataset splitting.
* **Two-Phase Architecture:** 
  1. *Global Place Recognition* - **Embedder** (cosine similarity for coarse region retrieval).
  2. *Local Pose Estimation* - **Localizer** (dense cross-correlation for precise alignment).
* **Evaluation:** Ready to use training and testing scripts for evaluating Top-K accuracy and generating spatial probability heatmaps.
* **Zero-Shot Transfer:** The model is trained on synthetic/aerial data but is capable of generalizing to real UAV data.
---

## Repository Structure

The project follows a modular architecture, strictly separating the core logic, execution scripts, and data storage.

```text
BBAP20/
├── configs/                  # YAML configuration files for model hyperparameters
│   ├── embedder.yaml
│   └── localizer.yaml
├── data/                     # Local data storage (ignored by git)
│   ├── dataset/              # Meta files, raw TIFF/LAZ data, and processed image pairs
│   │   ├── meta_files/       # Downloaded .meta4 files (paste here)
│   │   ├── processed/        # Cropped and aligned image patches
│   │   │   ├── test/
│   │   │   ├── train/
│   │   │   └── validation/
│   │   └── raw/              # Raw downloaded .tif and .laz files
│   │       ├── test/
│   │       ├── train/
│   │       └── validation/
│   ├── embedder/             # Embedder
│   │   ├── checkpoints/      # Saved training checkpoints
│   │   ├── models/           # Best saved models (.pth)
│   │   └── tensor_boards/    # TensorBoard logging events
│   ├── localizer/            # Localizer
│   │   ├── checkpoints/      
│   │   ├── models/           
│   │   └── tensor_boards/    
│   └── results/              # Output evaluation results and visualizations
├── scripts/                  # High-level execution scripts for the entire pipeline
│   ├── 00_meta_extractor.py  # Extracts URLs and downloads raw data from .meta4 files placed in /data/dataset/meta_files
│   ├── 01_prepare_data.py    # Processes raw files into cropped & aligned height/satellite map pairs
│   ├── 02_train_embedder.py  # Embedder training script
│   ├── 03_train_localizer.py # Localizer training script
│   ├── 04_test_embedder.py   # Embedder evaluation
│   └── 05_test_localizer.py  # Localizer evaluation
├── src/                      # Core source code modules
│   ├── embedder/             # Embedder module
│   │   ├── model.py          # Network architecture
│   │   ├── test.py           # Evaluation logic
│   │   └── train.py          # Training loop logic
│   ├── loader/               # Data loading module
│   │   ├── loader.py         # Custom PyTorch Dataset and DataLoader
│   │   └── utils.py          # Data augmentation and formatting utilities
│   ├── localizer/            # Localizer module
│   │   ├── model.py          # Dense feature extractor network
│   │   ├── ranker.py         # Cross-correlation and probability map generation
│   │   ├── test.py           # Localizer evaluation logic
│   │   └── train.py          # Localizer training loop
│   ├── preprocessor/         # Raw data processing tools
│   │   ├── laz_to_heightmap.py     # Point cloud to 2D elevation map conversion
│   │   └── tif_to_satellite_map.py # Orthophoto patching and alignment
│   └── shared/               # Shared utilities
│       ├── trainer.py        # Base trainer classes
│       ├── utils.py          # General helper functions
│       └── wrappers.py       # Wrappers
└── README.md                 # Project documentation
```

# Installation and Requirements
The project is written in Python (tested on version 3.12) and utilizes the PyTorch framework. To install the required dependencies, run:
```
git clone https://github.com/PinkiMan/BBAP20.git
cd BBAP20
pip install -r requirements.txt
```
_(It is highly recommended to use a virtual environment, like venv)._

# Custom Dataset Generation (Bayern Open Data)
1. For training and testing purposes, the system is primarily designed for data from the Bayer [Open Data portal](https://geodaten.bayern.de/opengeodata/). The entire data preparation process is fully automated.
2. Data Selection: Navigate to the Bayern Open Data portal. Open the map viewers for orthophotos [DOP20](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dop20rgb) and [Laser](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=laserdaten) data.
3. Mass Download: Use the mass download tool to select the desired bounding box. It is critical to use the exact same bounding box for both data types to ensure perfect spatial overlap (the easiest way is to copy the polygon specifications from the first selection and paste them into the second).
4. Obtain Metadata: Download the generated .meta4 (Metalink) files for both modalities and place them into data/dataset/meta_files/.
5. Download Raw Data: Run the extraction script: 
```
python scripts/00_meta_extractor.py
```
6. Prepare Dataset: The following script aligns the raw data (TIFF and LAZ), crops them into the required patches, and automatically splits them into `train`, `validation`, and `test` sets:
```
python scripts/01_prepare_data.py
```

# Model Training
Once the dataset is prepared, you can start the training process. Hyperparameters can be adjusted in the configs/ directory.
To train the **Embedder** (Global Place Recognition) using contrastive InfoNCE loss:
```
python scripts/02_train_embedder.py
```
To train the **Localizer** (Local Pose Estimation) using dense cross-correlation:
```
python scripts/03_train_localizer.py
```
Checkpoints, models, and TensorBoard logs will be automatically saved to `data/embedder/` and `data/localizer/` respectively.

# Evaluation
To test the trained models on the test set or external data, use the provided test scripts.
Evaluate the **Embedder** (Top-K accuracy):
```
python scripts/04_test_embedder.py
```
Evaluate the **Localizer** (Cross-correlation and Heatmap generation):
```
python scripts/05_test_localizer.py
```
Evaluation scripts automatically generate visual samples of the predicted 2D probability heatmaps and bounding boxes into the `data/results/` directory (enable in code).

# Data & Pre-trained Models
To easily reproduce the results presented in the thesis without running the training processes, you can download a sample dataset and the pre-trained weights.
* **[Download Sample Dataset and Pre-trained models](#)**

Simply extract/replace the downloaded folder into the `data/` directory following the repository structure.

# Evaluation
After downloading the dataset and the pre-trained models, the evaluation scripts [04_test_embedder.py](./scripts/04_test_embedder.py) and [05_test_localizer.py](./scripts/05_test_localizer.py) are ready to run out of the box (default set to using the downloaded `model_0.pth` model). 

If you wish to evaluate your own custom-trained models, you must update the model path in the `main` block of the respective testing script.

