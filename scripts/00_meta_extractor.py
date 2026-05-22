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
Filename: 00_meta_extractor.py
Directory: scripts/
"""

import random
import requests
import xml.etree.ElementTree as element_tree
from pathlib import Path

from src.shared.utils import project_directory


def download_file(url, target_path):
    if target_path.exists():
        print(f"  [INFO] File '{target_path.name}' already exists, skipping download.")
        return

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  [OK] Downloaded: {target_path.name}")
    except Exception as e:
        print(f"  [ERROR] While download {target_path.name}: {e}")

def count_pairs(file_path):
    if not file_path.exists():
        return 0
    files = set(f.name for f in file_path.iterdir() if f.is_file())
    laz_files = {f for f in files if f.endswith('.laz')}
    tif_files = {f for f in files if f.endswith('.tif')}

    count = 0
    for laz in laz_files:
        base = laz.replace('.laz', '')
        if f"32{base}.tif" in tif_files:
            count += 1
    return count

def download_and_split_meta_file(xml_files, base_output_dir):
    ns = {'ml': 'urn:ietf:params:xml:ns:metalink'}

    base_dir = Path(base_output_dir)
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    val_dir = base_dir / "validation"

    for d in [train_dir, test_dir, val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    existing_files = set()
    for d in [train_dir, test_dir, val_dir]:
        existing_files.update([f.name for f in d.iterdir() if f.is_file()])

    actual_counts = {
        "train": count_pairs(train_dir),
        "test": count_pairs(test_dir),
        "validation": count_pairs(val_dir)
    }
    total_count = sum(actual_counts.values())

    print("--- ACTUAL FOLDER COUNTS ---")
    print(f"Train: {actual_counts['train']} pairs")
    print(f"Test: {actual_counts['test']} pairs")
    print(f"Validation: {actual_counts['validation']} pairs")
    print(f"Total: {total_count} pairs\n")

    pairs = {}
    print("Loading metadata files...")
    for xml_file in xml_files:
        print(f" Reading: {Path(xml_file).name}")
        tree = element_tree.parse(xml_file)
        root = tree.getroot()

        for file_tag in root.findall('ml:file', ns):
            file_name = file_tag.get('name')
            url_tag = file_tag.find('ml:url', ns)

            if file_name and url_tag is not None:
                download_url = url_tag.text

                if file_name.endswith('.laz'):
                    base = file_name.replace('.laz', '')
                    if base not in pairs: pairs[base] = {}
                    pairs[base]['laz_name'] = file_name
                    pairs[base]['laz_url'] = download_url

                elif file_name.endswith('.tif'):
                    if file_name.startswith('32'):
                        base = file_name[2:].replace('.tif', '')
                        if base not in pairs: pairs[base] = {}
                        pairs[base]['tif_name'] = file_name
                        pairs[base]['tif_url'] = download_url

    complete_pairs = {k: v for k, v in pairs.items() if 'laz_name' in v and 'tif_name' in v}

    pairs_for_download = []
    for base, data in complete_pairs.items():
        if data['laz_name'] in existing_files and data['tif_name'] in existing_files:
            pass
        else:
            pairs_for_download.append((base, data))

    new_pairs_count = len(pairs_for_download)
    print(f"\nFound {len(complete_pairs)} pairs in metadata files.")
    print(f"Files for download: {new_pairs_count} pairs.")

    if new_pairs_count == 0:
        print("All files were already downloaded. Nothing to do.")
        return

    total_for_download = total_count + new_pairs_count

    final_counts = {
        "train": int(0.8 * total_for_download),
        "test": int(0.1 * total_for_download)
    }
    final_counts["validation"] = total_for_download - final_counts["train"] - final_counts["test"]

    added = {"train": 0, "test": 0, "validation": 0}

    for _ in range(new_pairs_count):
        missing = {
            k: final_counts[k] - (actual_counts[k] + added[k])
            for k in ["train", "test", "validation"]
        }
        max_missing = max(missing, key=missing.get)
        added[max_missing] += 1

    print("\n--- DOWNLOAD PLAN ---")
    print(f"New pairs to Train: {added['train']}")
    print(f"New pairs to Test: {added['test']}")
    print(f"New pairs to Validation: {added['validation']}\n")

    random.shuffle(pairs_for_download)

    train_data = pairs_for_download[:added['train']]
    test_data = pairs_for_download[added['train']: added['train'] + added['test']]
    val_data = pairs_for_download[added['train'] + added['test']:]

    distribution = {
        "TRAIN": (train_data, train_dir),
        "TEST": (test_data, test_dir),
        "VALIDATION": (val_data, val_dir)
    }

    for set_name, (pair_set, output_dir) in distribution.items():
        if not pair_set:
            continue

        print(f"--- DOWNLOAD STARTED FOR SET: {set_name} ({len(pair_set)} pairs) ---")
        for base, data in pair_set:
            laz_path = output_dir / data['laz_name']
            tif_path = output_dir / data['tif_name']

            download_file(data['laz_url'], laz_path)
            download_file(data['tif_url'], tif_path)

    print("\n[DONE] Folders are actualized and balanced in ratio 80/10/10.")


if __name__ == '__main__':
    directory = project_directory(Path('data/dataset'))

    metadata_directory = directory/"meta_files"

    output_folder = directory/"raw"

    if not metadata_directory.exists():
        print(f"ERROR: Metadata folder is missing: {metadata_directory}")
    else:
        file_list = [str(file) for file in metadata_directory.glob("*.meta4")]

        if not file_list:
            print(f"ERROR: In folder {metadata_directory} were no .meta4 files found.")
        else:
            print(f"Found {len(file_list)} meta files. Running analysis...\n")
            download_and_split_meta_file(file_list, output_folder)
