import os
import requests
import zipfile
import shutil
from tqdm import tqdm

COCO_URLS = {
    "images_train": "http://images.cocodataset.org/zips/train2017.zip",
    "images_val": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

DATASET_DIR = "coco_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")

def download(url, dest_dir):
    """
    downloads the given url (url) to the destination folder (dest_dir)
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    filename = os.path.join(dest_dir, os.path.basename(url))
    
    with open(filename, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(url)}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    return filename

def extract(zip_path, dest):
    """
    extracts the give zip file (zip_path) to a destination directory (dest)
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        with tqdm(total=total_files, unit='file', desc="Extracting") as bar:
            for file in zip_ref.infolist():
                zip_ref.extract(file, dest)
                bar.update(1)

def setup_coco():
    """
    downloads and extracts the coco training images, validation images, and annotations
    """
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    if not os.path.exists(ANNOTATIONS_DIR):
        os.makedirs(ANNOTATIONS_DIR)

    train_zip = download(COCO_URLS["images_train"], IMAGES_DIR)
    extract(train_zip, IMAGES_DIR)
    os.remove(train_zip)

    val_zip = download(COCO_URLS["images_val"], IMAGES_DIR)
    extract(val_zip, IMAGES_DIR)
    os.remove(val_zip)

    annotations_zip = download(COCO_URLS["annotations"], ANNOTATIONS_DIR)
    extract(annotations_zip, ANNOTATIONS_DIR)
    os.remove(annotations_zip)

if __name__ == "__main__":
    setup_coco()