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
        concept = vals[1].strip()
        high_level[concept_id] = concept

# High Level Locations
# Redact handles out of bounds values, so for now the coords can be out of bounds
BOX_HEIGHT = 50
BOX_WIDTH = 50
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
for img_id, entries in concept_locs.items():
    for entry in entries:
        rows.append({'image_id': img_id, 'high_level': entry['part_id'], 'coords': entry['coords']})
locs_df = pd.DataFrame(rows)
df = pd.merge(df, locs_df, on=['image_id', 'high_level'], how='left')
t= df[df["coords"].isna()]
df.loc[df['high_level'] == 'general', 'coords'] = pd.Series([[-1.0, -1.0, 999.0, 999.0]] * len(df[df['high_level'] == 'general']))  # Set general to be the whole image
print(f"Coord Mappings Made: {df.shape}")
df = df[df["coords"].notna()]
print(f"Final df Shape: {df.shape}")

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






