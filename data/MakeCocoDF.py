import os
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO

DATASET_DIR = "coco_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")
ANNOTATION_FILE = os.path.join(ANNOTATIONS_DIR, 'annotations', 'instances_train2017.json')
VALIDATION_FILE = os.path.join(ANNOTATIONS_DIR, 'annotations', 'instances_val2017.json')

def load_api(annotation_file):
    """loads COCO dataset using pycocotools"""
    coco = COCO(annotation_file)
    return coco

def make_CW_df(coco, img_dir, is_train):
    """creates a COCO DataFrame in the CW fromat for training and validation data."""
    categories = coco.loadCats(coco.getCatIds())
    cat_dict = {cat['id']: (cat['name'], cat['supercategory']) for cat in categories}
    cat_name_to_id = {name: idx for idx, (name, _) in enumerate(cat_dict.values())}
    # cat_name_to_id = {supercat: idx for idx, (_, supercat) in enumerate(cat_dict.values())} # if we wanted class to be the superconcept enum

    img_ids = coco.getImgIds()
    image_id_list = []
    certainty_id_list = []
    class_list = []
    path_list = []
    is_train_list = []
    low_level_list = []
    high_level_list = []
    coords_list = []
    bbox_list = []
    augmented_list = []

    for img_id in tqdm(img_ids, desc="Compiling Image Info", unit="image"):
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            cat_id = ann['category_id']
            cat_name = coco.loadCats([cat_id])[0]['name']
            cat_name, supercat_name = cat_dict[cat_id]
            cat_enum = cat_name_to_id[cat_name]
            # cat_enum = cat_name_to_id[supercat_name] # if we wanted class to be the superconcept enum
            x, y, width, height = ann['bbox']

            # append to lists
            image_id_list.append(img_info['id'])
            certainty_id_list.append(0)  # Set to 0 for all images
            class_list.append(cat_enum)
            path_list.append(os.path.join(img_dir, img_info['file_name']))
            is_train_list.append(1 if is_train else 0)
            low_level_list.append(cat_name)
            high_level_list.append(cat_name)  # Using category as high-level too
            # high_level_list.append(supercat_name)  # if we wanted highlevel to be super concepts
            coords_list.append([0, 0, img_info['width'], img_info['height']])
            bbox_list.append([x, y, x + width, y + height])
            augmented_list.append(0)  # Set to 0 for all images

    # create a DataFrame
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

    return df

if __name__ == "__main__":
    # train data
    coco_train = load_api(ANNOTATION_FILE)
    train_df = make_CW_df(coco_train, os.path.join(IMAGES_DIR, 'train2017'), is_train=True)
    # val data
    coco_val = load_api(VALIDATION_FILE)
    val_df = make_CW_df(coco_val, os.path.join(IMAGES_DIR, 'val2017'), is_train=False)
    # concatenate train and val dfs
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    print(full_df.head())

    full_df.to_parquet(os.path.join(DATASET_DIR, "coco_CW.parquet"), index=False)