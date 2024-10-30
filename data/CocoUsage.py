import os
from pycocotools.coco import COCO

DATASET_DIR = "coco_dataset"
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")

def load_api(annotation_file):
    """loads COCO dataset using pycocotools"""
    coco = COCO(annotation_file)
    return coco

def get_cats(coco):
    """Returns all categories in the dataset"""
    cats = coco.loadCats(coco.getCatIds())
    return [cat['name'] for cat in cats]

def get_by_cat(coco, category_name):
    """returns image IDs for a given category"""
    catIds = coco.getCatIds(catNms=[category_name])
    imgIds = coco.getImgIds(catIds=catIds)
    return imgIds

def get_info(coco, img_id):
    """returns information about a specific image"""
    img_info = coco.loadImgs(img_id)[0]
    return img_info

if __name__ == "__main__":
    # load COCO API
    annotation_file = os.path.join(ANNOTATIONS_DIR, 'annotations', 'instances_train2017.json')
    coco = load_api(annotation_file)

    # get COCO categories
    categories = get_cats(coco)
    print(f"COCO Categories: {categories}")

    # get image IDs for the 'person' category
    img_ids = get_by_cat(coco, 'person')
    print(f"Image IDs for 'person': {img_ids[:10]}")

    # get information about the first image
    if img_ids:
        img_info = get_info(coco, img_ids[0])
        print(f"Image Info: {img_info}")
