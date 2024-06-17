import os
import json
import pyarrow  # Needed for parquet
import pandas as pd

"""
The objective here is to make the parquet that will be of format:

image id, class, high level concept, low level concept, coords, certainty

Here the high level concept is the part, and low level the attribute.
Coords are computed from a preset window size, this can be changed easily.
"""

cub_path = "/usr/xtmp/aak61/CUB_200_2011"

# Step 1, make the dict
# Low Level Concepts + Certainty
low_level_path = os.path.join(cub_path, "attributes/image_attribute_labels.txt")
df = pd.read_csv(low_level_path, delim_whitespace=True, header=None, names=['image_id', 'attribute_id', 'is_present', 'certainty_id', 'temp'])
df = df[df["is_present"] == 1]  # Remove not present concepts
df = df.drop(columns=["is_present", "temp"], axis=1)
print(f"Initial df Read: {df.shape}")


# Class Values
class_path = os.path.join(cub_path, "image_class_labels.txt")
classes = pd.read_csv(class_path, delim_whitespace=True, header=None, names=['image_id', 'class'])
df = pd.merge(df, classes, on="image_id", how="left")
print(f"Classes Made: {df.shape}")


# Image Paths
image_paths = os.path.join(cub_path, "images.txt")
images = pd.read_csv(image_paths, delim_whitespace=True, header=None, names=['image_id', 'path'])
df = pd.merge(df, images, on="image_id", how="left")
print(f"Paths Added: {df.shape}")


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
print(f"Train Test Made: {df.shape}")

# Low level concept IDs
low_level = {}
low_path = os.path.join(cub_path, "attributes.txt")
with open(low_path, "r") as file:
    for line in file:
        vals = line.split(" ")
        concept_id = int(vals[0])
        concept = vals[1].strip()
        low_level[concept_id] = concept
df['low_level'] = df['attribute_id'].map(low_level)
df = df.drop(columns=['attribute_id'])
print(f"Low Level Concepts Made: {df.shape}")
rev = {value:key for key, value in low_level.items()}
with open("low_level.json", 'w') as json_file:
    json.dump(rev, json_file, indent=4)

# Low to High level concept mapping
mapping_path = os.path.abspath("mappings.json")
with open(mapping_path, "r") as file:
    mappings = json.load(file)
df['high_level'] = df['low_level'].map(mappings)
print(f"High to Low Mappings Made: {df.shape}")

# High level concept IDs
high_level = {}
high_path = os.path.join(cub_path, "parts/parts.txt")
with open(high_path, "r") as file:
    for line in file:
        vals = line.split(" ")
        concept_id = int(vals[0])
        concept = vals[1:]
        # Sometimes concepts are left/right. This merges them into one high level part
        if len(concept) > 1:
            concept = concept[1]
        else:
            concept = concept[0]
        concept = concept.strip()
        high_level[concept_id] = concept
with open("high_level.json", 'w') as json_file:
    json.dump(high_level, json_file, indent=4)

# High Level Locations
# Redact handles out of bounds values, so for now the coords can be out of bounds
BOX_HEIGHT = 80
BOX_WIDTH = 80
concept_locs = {}
part_path = os.path.join(cub_path, "parts/part_locs.txt")
with open(part_path, "r") as file:
    delta_x = BOX_HEIGHT // 2
    delta_y = BOX_HEIGHT // 2
    for line in file:
        vals = line.split(" ")
        visible = int(vals[4])
        if not visible:
            continue
        img_id = int(vals[0])
        part_id = high_level[int(vals[1])]
        x = float(vals[2])
        y = float(vals[3])
        coords = [x - delta_x, y - delta_y, x + delta_x, y + delta_y]
        if img_id not in concept_locs:
            concept_locs[img_id] = []
        concept_locs[img_id].append({"part_id": part_id, "coords": coords})
rows = []
print(len(concept_locs))
for img_id, entries in concept_locs.items():
    for entry in entries:
        rows.append({'image_id': img_id, 'high_level': entry['part_id'], 'coords': entry['coords']})
locs_df = pd.DataFrame(rows)
df = pd.merge(df, locs_df, on=['image_id', 'high_level'], how='left')
df = df[df["coords"].notna()]

# Set general to be the whole image
mask = df['high_level'] == 'general'
num_rows = mask.sum() 
coords_series = pd.Series([[-9999.0, -9999.0, 9999.0, 9999.0]] * num_rows, dtype=object)
df.loc[mask, 'coords'] = coords_series

print(f"Coord Mappings Made: {df.shape}")


# Add Bounding Boxes for Crop
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
print(f"BBoxes Made: {df.shape}")


# Save the train and test dfs
train_path = os.path.join(cub_path, "datasets/cub200_cw/train.parquet")
test_path = os.path.join(cub_path, "datasets/cub200_cw/test.parquet")
is_train = df["is_train"] == 1

train_df = df[is_train]
print(train_df.shape)
train_df.to_parquet(train_path, index=None)
test_df = df[~is_train]
print(test_df.shape)
test_df.to_parquet(test_path, index=None)






