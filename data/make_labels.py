import os
import json
import pyarrow  # Needed for parquet
import argparse
import pandas as pd
import numpy as np

from PIL import Image, ImageFilter


"""
The objective here is to make the parquet that will be of format:
'image_id', 'certainty_id', 'class', 'path', 'is_train', 'low_level', 'high_level', 'coords', 'bbox', 'augment'

Here the high level concept is the part, and low level the attribute.
Coords are computed from a preset window size, this can be changed easily in the read_file function.
"""


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Read files from a CUB directory to create datasets.")
    parser.add_argument('test_classes', type=str, nargs='+', help='List of test classes as their ID numbers')
    parser.add_argument('--write_json', action='store_true', help='Flag to write output as JSON')
    args = parser.parse_args()

    CUB_PATH = os.getenv("CUB_PATH")
    train_path = os.path.join(CUB_PATH, "datasets/cub200_cw/train.parquet")
    test_path = os.path.join(CUB_PATH, "datasets/cub200_cw/test.parquet")

    train, test = read_files(CUB_PATH, args.test_classes, args.write_json)
    train = crop_and_augment(train, CUB_PATH)
    test = crop_and_augment(test, CUB_PATH)

    train["image_id"] = train['image_id'].astype(str)
    train.to_parquet(train_path, index=None)
    test.to_parquet(test_path, index=None)

    # TODO: patch for weird dimension issue, to be removed
    for image_path in train['path']:
    # Open the image file
        with Image.open(image_path) as img:
            # Check if image dimensions are not 224x224
            if img.size != (224, 224):
                # Resize the image
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                # Optionally save the resized image back to disk
                img.save(image_path)


def read_files(cub_path, test_classes, write_json=False):
    """
    Read files navigates an unaltered cub directory and creates a train and test dataset.
    Test classes are classes that are always witheld for the test set (used to test concepts).
    """
    # Low Level Concepts + Certainty
    # Skip bad lines since there is an issue with image 2275 here
    low_level_path = os.path.join(cub_path, "attributes/image_attribute_labels.txt")
    df = pd.read_csv(low_level_path, delim_whitespace=True, header=None,
                     names=['image_id', 'attribute_id', 'is_present', 'certainty_id', 'temp'],
                     on_bad_lines='skip')
    df = df[df["is_present"] == 1]  # Remove not present concepts
    df = df.drop(columns=["is_present", "temp"], axis=1)

    # Class Values
    class_path = os.path.join(cub_path, "image_class_labels.txt")
    classes = pd.read_csv(class_path, delim_whitespace=True, header=None, names=['image_id', 'class'])
    df = pd.merge(df, classes, on="image_id", how="left")

    # Image Paths
    image_paths = os.path.join(cub_path, "images.txt")
    images = pd.read_csv(image_paths, delim_whitespace=True, header=None, names=['image_id', 'path'])
    df = pd.merge(df, images, on="image_id", how="left")

    # Train test split values
    train_test_split = {}
    train_test_path = os.path.join(cub_path, "train_test_split.txt")
    with open(train_test_path, "r") as file:
        for line in file:
            vals = line.split(" ")
            img_id = int(vals[0])
            split = int(vals[1])
            train_test_split[int(vals[0])] = split
    split_df = pd.DataFrame(list(train_test_split.items()), columns=['image_id', 'is_train'])
    df = pd.merge(df, split_df, on="image_id", how="outer")

    # Low level concept IDs
    low_level = {}
    low_path = os.path.join(cub_path, "attributes/attributes.txt")
    with open(low_path, "r") as file:
        for line in file:
            vals = line.split(" ")
            concept_id = int(vals[0])
            concept = vals[1].strip()
            low_level[concept_id] = concept
    df['low_level'] = df['attribute_id'].map(low_level)
    df = df.drop(columns=['attribute_id'])

    # Also write a json dictionary to map low level concepts to an index
    if write_json:
        rev = {value: key for key, value in low_level.items()}
        json_path = os.path.abspath("data/json/low_level.json")
        with open(json_path, 'w') as json_file:
            json.dump(rev, json_file, indent=4)

    # Low to High level concept mapping
    mapping_path = os.path.abspath("data/json/mappings.json")
    with open(mapping_path, "r") as file:
        mappings = json.load(file)
    df['high_level'] = df['low_level'].map(mappings)

    # High level concept IDs
    high_level = {}
    high_path = os.path.join(cub_path, "parts/parts.txt")
    with open(high_path, "r") as file:
        for line in file:
            vals = line.split(" ")
            concept_id = int(vals[0])
            concept = vals[1:]
            # Sometimes concepts are left/right. This merges them into one high level part.
            if len(concept) > 1:
                concept = concept[1]
            else:
                concept = concept[0]
            concept = concept.strip()
            high_level[concept_id] = concept

    # Also write the high level json mapping
    if write_json:
        json_path = os.path.abspath("data/json/high_level.json")
        with open(json_path, 'w') as json_file:
            json.dump(high_level, json_file, indent=4)

    # High Level Locations
    # Redact handles out of bounds values, so for now the coords can be out of bounds depending on how many concepts
    # are visible, we can infer how zoomed an image is more zoomed out, smaller box size
    image_id_counts = df['image_id'].value_counts().to_dict()
    # Manually set box sizes for different high level concepts
    BOX_DIM = {
        "back": 140,
        "beak": 75,
        "belly": 140,
        "breast": 140,
        "crown": 75,
        "forehead": 75,
        "eye": 40,
        "leg": 75,
        "wing": 140,
        "nape": 40,
        "tail": 75,
        "throat": 75
    }

    concept_locs: dict[int, list[dict]] = {}
    part_path = os.path.join(cub_path, "parts/part_locs.txt")
    with open(part_path, "r") as file:
        for line in file:
            vals = line.split(" ")
            visible = int(vals[4])
            if not visible:
                continue
            img_id = int(vals[0])
            part_id = high_level[int(vals[1])]
            x = float(vals[2])
            y = float(vals[3])

            # Have preset sizes for the concepts
            dim = BOX_DIM[part_id]
            num_visible = image_id_counts[img_id]
            # If it's more zoomed out (>6 concepts visible), then scale it by 1/2
            if num_visible > 6:
                delta_x = int(dim * 0.5)
                delta_y = delta_x
            coords = [x - delta_x, y - delta_y, x + delta_x, y + delta_y]

            if img_id not in concept_locs:
                concept_locs[img_id] = []
            concept_locs[img_id].append({"part_id": part_id, "coords": coords})
    rows = []
    for img_id, entries in concept_locs.items():
        for entry in entries:
            rows.append({'image_id': img_id, 'high_level': entry['part_id'], 'coords': entry['coords']})
    locs_df = pd.DataFrame(rows)
    df = pd.merge(df, locs_df, on=['image_id', 'high_level'], how='left')
    df = df[df["coords"].notna()]

    # Set general to be the whole image
    mask = (df['high_level'] == 'general')
    num_rows = mask.sum()
    coords_series = pd.Series([[-9999.0, -9999.0, 9999.0, 9999.0]] * num_rows, dtype=object)
    df.loc[mask, 'coords'] = coords_series

    # Add Bounding Boxes for Cropping
    image_bboxes = {}
    bbox_path = os.path.join(cub_path, "bounding_boxes.txt")
    with open(bbox_path, "r") as file:
        for line in file:
            vals = line.split(" ")
            img_id = int(vals[0])
            x = float(vals[1])
            y = float(vals[2])
            width = float(vals[3])
            height = float(vals[4])
            bbox = [x, y, width, height]
            if img_id not in image_bboxes:
                image_bboxes[img_id] = bbox
    bbox_df = pd.DataFrame(list(image_bboxes.items()), columns=['image_id', 'bbox'])
    df = pd.merge(df, bbox_df, on=['image_id'], how='left')

    # This is needed to separate the original from the augmented later on
    df["augmented"] = 0

    # Add all test classes to the test set, then split
    test_class_mask = df["class"].isin(test_classes)
    df.loc[test_class_mask, 'is_train'] = 0
    is_train = df["is_train"] == 1
    train_df = df[is_train]
    test_df = df[~is_train]

    return train_df, test_df


def crop_and_augment(df: pd.DataFrame, base_path, target_size=(224, 224)):
    augmented_rows = []
    for idx, row in df.iterrows():

        original_path = os.path.join(base_path, f"images/{row['path']}")
        new_dir = os.path.join(base_path, 'datasets', 'cub200_cw', 'train' if row['is_train'] else 'test')
        new_path = os.path.join(new_dir, f"{row['image_id']}.jpg")

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        bbox = row['bbox']
        coords = row['coords']

        if not os.path.exists(new_path):
            image = Image.open(original_path)
            image_cropped = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            image_resized = image_cropped.resize(target_size)
            image_resized.save(new_path)

            # Only do this once per image
            if row['is_train']:
                augmented_rows.extend(augment_data(image_resized, row, new_dir))

        x_scale = target_size[0] / bbox[2]
        y_scale = target_size[1] / bbox[3]

        # When we crop, the new pixel is just relative to where the new bottom left corner is.
        # We always assume its visible in the bbox since the labels are reliable from amazon turk
        new_coords = [
            int((coords[0] - bbox[0]) * x_scale),
            int((coords[1] - bbox[1]) * y_scale),
            int((coords[2] - bbox[0]) * x_scale),
            int((coords[3] - bbox[1]) * y_scale)
        ]

        # Update df
        df.at[idx, 'coords'] = new_coords
        df.at[idx, 'path'] = new_path

    # Add augmented rows to the original DataFrame
    augmented_df = pd.DataFrame(augmented_rows)
    return pd.concat([df, augmented_df], ignore_index=True)


def augment_data(image: Image.Image, original_row: pd.Series, dir_path):
    # These are the transformations we do, but can definitely add more
    transformations = {
        'flipped': Image.FLIP_LEFT_RIGHT,
        'rotated': 15,  # degrees
        'noisy': 'add_noise',
        'blurred': (ImageFilter.GaussianBlur(radius=2))
    }

    new_rows = []
    for suffix, transform in transformations.items():
        if suffix == 'rotated':
            new_image = image.rotate(transform, expand=True)
        elif suffix == 'noisy':
            new_image = add_noise(image)
        elif suffix == 'flipped':
            new_image = image.transpose(transform)
        else:
            new_image = image.filter(transform)

        # Resize again just in case the size changed (rotation might)
        # This name is super weird, but the reason
        new_path = os.path.join(dir_path, f"{original_row['image_id']}_{suffix}.jpg")
        new_image.save(new_path)

        # Clone original row and update necessary fields
        new_row = original_row.copy()
        new_row['image_id'] = f"{original_row['image_id']}_{suffix}"
        new_row['path'] = new_path
        new_row['augmented'] = 1
        new_rows.append(new_row)

    return new_rows


def add_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(loc=0, scale=35, size=np_image.shape)
    np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_image)


if __name__ == '__main__':
    main()
