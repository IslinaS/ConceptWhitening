import os
import pyarrow  # Needed for parquet
import json
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFilter
from pycocotools.coco import COCO

# Define dataset directory paths
JSON_DIR = "/home/users/hvl5/work/ConceptWhitening/data/json/json_coco"
DATASET_DIR = "/usr/project/xtmp/cs474_cv/ConceptWhitening/data/coco_mini"
IMAGES_DIR = os.path.join(DATASET_DIR, "val2017")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")
ANNOTATION_FILE = os.path.join(ANNOTATIONS_DIR, 'instances_val2017.json')

TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "images/train")
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, "images/test")

# Ensure image directories exist and are clean
if os.path.exists(TRAIN_IMAGES_DIR):
    shutil.rmtree(TRAIN_IMAGES_DIR)
if os.path.exists(TEST_IMAGES_DIR):
    shutil.rmtree(TEST_IMAGES_DIR)
os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)


def load_api(annotation_file: str) -> COCO:
    """
    Loads the COCO dataset using pycocotools.
    
    Args:
        annotation_file (str): Path to the COCO annotation JSON file.
    
    Returns:
        COCO: COCO dataset object.
    """
    return COCO(annotation_file)

def read_files(coco: COCO, img_dir: str) -> pd.DataFrame:
    """
    Reads the COCO dataset and creates a structured DataFrame.
    
    Args:
        coco (COCO): COCO dataset object.
        img_dir (str): Directory where images are stored.
    
    Returns:
        pd.DataFrame: Processed dataset containing image metadata and annotations.
    """
    img_dir = os.path.abspath(img_dir)
    print(f"Absolute path to image directory: {img_dir}")
    
    categories = coco.loadCats(coco.getCatIds())
    cat_dict = {cat['id']: (cat['name'], cat['supercategory']) for cat in categories}
    cat_name_to_id = {name: idx for idx, (name, _) in enumerate(cat_dict.values())}
    
    # Initialize lists for dataframe columns
    image_id_list, certainty_id_list, class_list, path_list = [], [], [], []
    is_train_list, low_level_list, high_level_list, coords_list, bbox_list, augmented_list = [], [], [], [], [], []

    for img_id in coco.imgs:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            cat_id = ann['category_id']
            cat_name, supercat_name = cat_dict[cat_id]
            cat_enum = cat_name_to_id[cat_name]
            x, y, width, height = ann['bbox']

            # Append extracted data to lists
            image_id_list.append(str(img_info['id']))  # Ensure IDs are strings
            certainty_id_list.append(0)  # Placeholder value
            class_list.append(cat_enum)
            path_list.append(os.path.join(img_dir, img_info['file_name']))
            is_train_list.append(0)  # Default to test set, updated later
            low_level_list.append(cat_name)
            high_level_list.append(supercat_name)  # Use COCO supercategory
            coords_list.append([0, 0, img_info['width'], img_info['height']])
            bbox_list.append([x, y, x + width, y + height])
            augmented_list.append(0)  # Placeholder value

    # Create DataFrame
    df = pd.DataFrame({
        'image_id': image_id_list,
        'certainty_id': certainty_id_list,
        'class': class_list,
        'path': path_list,
        'is_train': is_train_list,
        'low_level': low_level_list,
        'high_level': high_level_list,
        'coords': coords_list,
        'bbox': bbox_list,
        'augmented': augmented_list,
    })
    
    # Save mappings to JSON
    low_level_dict = {name: idx for idx, name in enumerate(df['low_level'].unique())}
    high_level_dict = {idx: name for idx, name in enumerate(df['high_level'].unique())}
    low_to_high_mapping = dict(zip(df['low_level'], df['high_level']))

    with open(os.path.join(JSON_DIR, "low_level.json"), 'w') as json_file:
        json.dump(low_level_dict, json_file, indent=4)
    with open(os.path.join(JSON_DIR, "high_level.json"), 'w') as json_file:
        json.dump(high_level_dict, json_file, indent=4)
    with open(os.path.join(JSON_DIR, "mappings.json"), "w") as json_file:
        json.dump(low_to_high_mapping, json_file, indent=4)
    
    return df

def split_dataset(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): The complete dataset.
        test_size (float): Fraction of data to use for testing.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        tuple: (train_df, test_df)
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df['is_train'] = 1
    val_df['is_train'] = 0
    return train_df, val_df

def crop_and_augment(df: pd.DataFrame, base_path: str, target_dir: str, target_size=(224, 224)):
    """
    Crops and augments images based on bounding boxes, then saves the processed images.
    
    Args:
        df (pd.DataFrame): DataFrame containing image metadata and annotations.
        base_path (str): Base directory for images.
        target_size (tuple): Target size for resized images.
    
    Returns:
        pd.DataFrame: Augmented dataset.
    """
    augmented_rows = []
    for idx, row in df.iterrows():
        original_path = os.path.join(base_path, row['path'])
        new_path = os.path.join(target_dir, f"{row['image_id']}.jpg")

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        bbox = row['bbox']
        coords = row['coords']

        image = Image.open(original_path)
        # image.save(new_path)
        # image_cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        # image_resized = image_cropped.resize(target_size)
        image_resized = image.resize(target_size)

        image_resized.save(new_path)
        augmented_rows.extend(augment_data(image_resized, row, target_dir))

        # x_scale = target_size[0] / (bbox[2] - bbox[0])
        # y_scale = target_size[1] / (bbox[3] - bbox[1])

        # new_coords = [
        #     int((coords[0] - bbox[0]) * x_scale),
        #     int((coords[1] - bbox[1]) * y_scale),
        #     int((coords[2] - bbox[0]) * x_scale),
        #     int((coords[3] - bbox[1]) * y_scale)
        # ]

        # df.at[idx, 'coords'] = new_coords
        df.at[idx, 'path'] = new_path

    augmented_df = pd.DataFrame(augmented_rows)
    return pd.concat([df, augmented_df], ignore_index=True)

def augment_data(image: Image.Image, original_row: pd.Series, dir_path):
    transformations = {
        'flipped': Image.FLIP_LEFT_RIGHT,
        'rotated': 15,
        'blurred': ImageFilter.GaussianBlur(radius=2)
    }
    new_rows = []

    for suffix, transform in transformations.items():
        if suffix == 'rotated':
            new_image = image.rotate(transform)
        elif suffix == 'flipped':
            new_image = image.transpose(transform)
        else:
            new_image = image.filter(transform)

        new_path = os.path.join(dir_path, f"{original_row['image_id']}_{suffix}.jpg")
        new_image.save(new_path)

        new_row = original_row.copy()
        new_row['image_id'] = f"{original_row['image_id']}_{suffix}"
        new_row['path'] = new_path
        new_row['augmented'] = 1
        new_rows.append(new_row)

    return new_rows

if __name__ == "__main__":
    coco_data = load_api(ANNOTATION_FILE)
    df = read_files(coco_data, IMAGES_DIR)
    train_df, test_df = split_dataset(df)

    train_df: pd.DataFrame = crop_and_augment(train_df, IMAGES_DIR, TRAIN_IMAGES_DIR)
    test_df: pd.DataFrame = crop_and_augment(test_df, IMAGES_DIR, TEST_IMAGES_DIR)

    train_df.to_parquet(os.path.join(DATASET_DIR, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(DATASET_DIR, "test.parquet"), index=False)
    
    print("Mini COCO dataset split into train and test sets with cropping and augmentation.")
