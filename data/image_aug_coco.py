import os
import re
import pyarrow  # Needed for parquet
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
    base_path = "/usr/project/xtmp/cs474_cv/ConceptWhitening/coco_dataset"
    train_path = os.path.join(base_path, "train.parquet")
    test_path = os.path.join(base_path, "test.parquet")

    #train_aug = size_and_augment(train_path, base_path, aug=True)
    #train_aug.to_parquet("/usr/project/xtmp/cs474_cv/ConceptWhitening/coco_dataset/aug_train.parquet")
    #print(train_aug.shape, flush=True)

    #size_and_augment(test_path, base_path, aug=False)
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    counter = 0
    for index, row in train.iterrows():
        if '/usr/project/xtmp/cs474_cv/' not in row['path']:
            counter += 1
            train.loc[index, 'path'] = f"/usr/project/xtmp/cs474_cv/ConceptWhitening/coco_dataset/images_resized/train/{row['image_id']}.jpg"
    print(counter)
    train.to_parquet("/usr/project/xtmp/cs474_cv/ConceptWhitening/coco_dataset/train.parquet")
    counter = 0
    for index, row in test.iterrows():
        if '/usr/project/xtmp/cs474_cv/' not in row['path']:
            counter += 1
            test.loc[index, 'path'] = f"/usr/project/xtmp/cs474_cv/ConceptWhitening/coco_dataset/images_resized/test/{row['image_id']}.jpg"
    print(counter)
    test.to_parquet("/usr/project/xtmp/cs474_cv/ConceptWhitening/coco_dataset/test.parquet")


def size_and_augment(df_path, base_path, aug, target_size=(224, 224)):
    augmented_rows = []
    df = pd.read_parquet(df_path)
    unique_df = df.drop_duplicates(subset='image_id')

    for _, row in unique_df.iterrows():
        image_id = row['image_id']
        original_path = row['path']
        new_dir = os.path.join(base_path, 'images_resized', 'train' if row['is_train'] else 'test')
        new_path = os.path.join(new_dir, f"{row['image_id']}.jpg")
        row['path'] = new_path

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        image = Image.open(original_path)
        image_resized = image.resize(target_size)
        image_resized.save(new_path)

        if aug:
            augmented_rows.extend(augment_data(image_resized, df, image_id, new_dir))

    # Add augmented rows to the original DataFrame
    augmented_df = pd.DataFrame(augmented_rows)
    return pd.concat([df, augmented_df], ignore_index=True)


def augment_data(image: Image.Image, df, image_id, dir_path):
    # These are the transformations we do, but can definitely add more
    transformations = {
        'flipped': Image.FLIP_LEFT_RIGHT,
        'rotated': 15,  # degrees
        'noisy': 'add_noise',
        'blurred': (ImageFilter.GaussianBlur(radius=2))
    }

    relevant_rows = df[df['image_id'] == image_id]

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

        # This name is super weird, but the reason
        new_path = os.path.join(dir_path, f"{image_id}_{suffix}.jpg")
        new_image.save(new_path)

        for _, original_row in relevant_rows.iterrows():
            # Clone each row and update necessary fields
            new_row = original_row.copy()
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
