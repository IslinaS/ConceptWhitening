import os
import shutil
import random

from argparse import ArgumentParser
from PIL import Image

parser = ArgumentParser(description="Create CUB_200_2011 auxiliary datasets.")
parser.add_argument("--cub-path", type=str, required=True, help="Path to the CUB_200_2011 dataset directory")
parser.add_argument("--concept-path", type=str, required=True, help="Path to the concept directory")
parser.add_argument("--split-dataset", action="store_true",
                    help="Whether to split the CUB_200_2011 dataset into train, test, and val folders")
parser.add_argument("--keep-previous", action="store_true",
                    help="Whether to keep previous concept, train, test, and val folders")
args = parser.parse_args()

CUB_DIR = args.cub_path
CONCEPT_DIR = args.concept_path
SPLIT_DATASET = args.split_dataset
KEEP_PREVIOUS = args.keep_previous

# Define paths
PARTS_FILE = os.path.join(CUB_DIR, "parts", "parts.txt")
PART_LOCS_FILE = os.path.join(CUB_DIR, "parts", "part_locs.txt")
ATTRIBUTES_FILE = os.path.join(CUB_DIR, "attributes", "attributes.txt")
IMAGE_ATTRIBUTE_LABELS_FILE = os.path.join(CUB_DIR, "attributes", "image_attribute_labels.txt")
BOUNDING_BOX_FILE = os.path.join(CUB_DIR, "bounding_boxes.txt")
IMAGES_FILE = os.path.join(CUB_DIR, "images.txt")
TRAIN_TEST_SPLIT_FILE = os.path.join(CUB_DIR, "train_test_split.txt")
IMAGES_DIR = os.path.join(CUB_DIR, "images")

CONCEPT_DIR_TRAIN = os.path.join(CONCEPT_DIR, "concept_train")
CONCEPT_DIR_TEST = os.path.join(CONCEPT_DIR, "concept_test")
TRAIN_DIR = os.path.join(CUB_DIR, 'train')
VAL_DIR = os.path.join(CUB_DIR, 'val')
TEST_DIR = os.path.join(CUB_DIR, 'test')

VAL_TO_TRAIN_RATIO = 0.2
BOUNDING_BOX_SIZE = 100

# Create concept directories if they don't exist
os.makedirs(CONCEPT_DIR, exist_ok=True)
os.makedirs(CONCEPT_DIR_TRAIN, exist_ok=True)
os.makedirs(CONCEPT_DIR_TEST, exist_ok=True)

if not KEEP_PREVIOUS:
    shutil.rmtree(CONCEPT_DIR_TRAIN, ignore_errors=True)
    shutil.rmtree(CONCEPT_DIR_TEST, ignore_errors=True)
    shutil.rmtree(TRAIN_DIR, ignore_errors=True)
    shutil.rmtree(VAL_DIR, ignore_errors=True)
    shutil.rmtree(TEST_DIR, ignore_errors=True)

if SPLIT_DATASET:
    # Get the list of classes and create class directories
    for class_name in os.listdir(IMAGES_DIR):
        os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

# Read images file to create a mapping of image_id to image name
IMAGES = {}
with open(IMAGES_FILE, "r") as file:
    for line in file:
        image_id, image_name = line.strip().split()
        image_id = int(image_id)
        
        IMAGES[image_id] = image_name

# Read bounding boxes file to create a mapping of image_id to cropped images
CROPPED_IMAGES = {}
with open(BOUNDING_BOX_FILE, "r") as file:
    for line in file:
        image_id, x, y, width, height = map(lambda x: int(float(x)), line.strip().split())
        image_name = IMAGES[image_id]

        # Open the image
        image_path = os.path.join(IMAGES_DIR, image_name)
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            continue

        cropped_image = image.crop((x, y, x + width, y + height))
        CROPPED_IMAGES[image_id] = cropped_image

# Read parts file to create a mapping of part_id to part name
print("Creating mapping from part_id to part_name...")       
PARTS = {}
with open(PARTS_FILE, "r") as file:
    for line in file:
        part_id, part_name = line.strip().split(' ', 1)
        part_id = int(part_id)
        part_name = part_name.replace(' ', '_')

        PARTS[part_id] = part_name
        print(f"    {part_id} {part_name}")     
        
    # Create output directory for this part if it doesn't exist
        part_output_train_dir = os.path.join(CONCEPT_DIR_TRAIN, part_name, part_name)
        os.makedirs(part_output_train_dir, exist_ok=True)

        part_output_test_dir = os.path.join(CONCEPT_DIR_TEST, part_name)
        os.makedirs(part_output_test_dir, exist_ok=True)

print()

# Read parts file to create a mapping of part_id to part name
print("Creating mapping from attribute_id to attribute_name...")
ATTRIBUTES = {}
with open(ATTRIBUTES_FILE, "r") as file:
    for line in file:
        attribute_id, attribute_name = line.strip().split()
        attribute_id = int(attribute_id)
        attribute_category, data = attribute_name.split("::")

        ATTRIBUTES[attribute_id] = (attribute_name, attribute_category, data)
        print(f"    {attribute_id} {attribute_category} {data}")     
        
    # Create output directory for this part if it doesn't exist
        attribute_output_train_dir = os.path.join(CONCEPT_DIR_TRAIN, attribute_name, attribute_name)
        os.makedirs(attribute_output_train_dir, exist_ok=True)

        attribute_output_test_dir = os.path.join(CONCEPT_DIR_TEST, attribute_name)
        os.makedirs(attribute_output_test_dir, exist_ok=True)

print()

# Define attribute mappings to parts
ATTRIBUTE_TO_PART_MAP = {
    "has_back_color": "back",
    "has_back_pattern": "back",
    "has_bill_shape": "beak",
    "has_bill_length": "beak",
    "has_bill_color": "beak",
    "has_belly_color": "belly",
    "has_belly_pattern": "belly",
    "has_breast_pattern": "breast",
    "has_breast_color": "breast",
    "has_crown_color": "crown",
    "has_forehead_color": "forehead",
    "has_eye_color": ["left_eye", "right_eye"],
    "has_leg_color": ["left_leg", "right_leg"],
    "has_wing_color": ["left_wing", "right_wing"],
    "has_wing_shape": ["left_wing", "right_wing"],
    "has_wing_pattern": ["left_wing", "right_wing"],
    "has_nape_color": "nape",
    "has_tail_shape": "tail",
    "has_upper_tail_color": "tail",
    "has_under_tail_color": "tail",
    "has_tail_pattern": "tail",
    "has_throat_color": "throat",
    # General attributes
    "has_upperparts_color": "general",
    "has_underparts_color": "general",
    "has_head_pattern": "general",
    "has_size": "general",
    "has_shape": "general",
    "has_primary_color": "general"
}

# Read part locations file
PART_LOCATIONS = {}
with open(PART_LOCS_FILE, "r") as file:
    for line in file:
        parts = line.strip().split()
        image_id = int(parts[0])
        part_id = int(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        visible = int(parts[4])

        if image_id not in PART_LOCATIONS:
            PART_LOCATIONS[image_id] = {}
        if visible:
            PART_LOCATIONS[image_id][part_id] = (x, y) 

# Read image attribute labels file
IMAGE_ATTRIBUTES = {}
with open(IMAGE_ATTRIBUTE_LABELS_FILE, "r") as file:
    for line in file:
        attributes = line.strip().split()
        image_id, attribute_id, present = map(int, attributes[:3])

        if image_id not in IMAGE_ATTRIBUTES:
            IMAGE_ATTRIBUTES[image_id] = []
        if present:
            IMAGE_ATTRIBUTES[image_id].append(attribute_id)

# Read train_test_split.txt file
with open(TRAIN_TEST_SPLIT_FILE, "r") as file:
    for line in file:
        image_id, is_training_image = map(int, line.strip().split())
        image_name = IMAGES[image_id]
        save_dir = CONCEPT_DIR_TRAIN if is_training_image else CONCEPT_DIR_TEST

        # Open the image
        image_path = os.path.join(IMAGES_DIR, image_name)
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            continue

        if SPLIT_DATASET:
            if not is_training_image:
                dest_path = os.path.join(TEST_DIR, image_name)
            elif random.random() < VAL_TO_TRAIN_RATIO:
                dest_path = os.path.join(VAL_DIR, image_name)
            else:
                dest_path = os.path.join(TRAIN_DIR, image_name)

            shutil.copy(image_path, dest_path)

        # Get part locations for the image
        parts = PART_LOCATIONS[image_id]
        
        # Get attributes for the image
        attributes = IMAGE_ATTRIBUTES[image_id]

        part_images = {}

        # Save crops for each part location
        for part_id in parts:
            x, y = parts[part_id]
            part_name = PARTS[part_id]

            # Calculate bounding box coordinates centered around the part location
            left = max(0, x - BOUNDING_BOX_SIZE // 2)  # x - half of side length
            upper = max(0, y - BOUNDING_BOX_SIZE // 2)  # y - half of side length
            right = left + BOUNDING_BOX_SIZE
            lower = upper + BOUNDING_BOX_SIZE

            # Crop the image
            cropped_part = image.crop((left, upper, right, lower))
            part_images[part_name] = cropped_part

            # Save the cropped image
            part_output_dir = os.path.join(save_dir, part_name)
            if is_training_image:
                part_output_dir = os.path.join(part_output_dir, part_name)

            output_path = os.path.join(part_output_dir, f"{image_id}.png")
            cropped_part.save(output_path)

        # Save crops for each attribute
        for attribute_id in attributes:
            # Determine the part corresponding to the attribute (if any)
            attribute_name, attribute_category, data = ATTRIBUTES[attribute_id]
            part_names = ATTRIBUTE_TO_PART_MAP[attribute_category]

            if type(part_names) is not list:
                part_names = [part_names]

            for part_name in part_names:
                if part_name in part_images or part_name == "general":
                    cropped_image = CROPPED_IMAGES[image_id] if part_name == "general" else part_images[part_name]

                    # Save cropped image in output directory
                    attribute_output_dir = os.path.join(save_dir, attribute_name)
                    if is_training_image:
                        attribute_output_dir = os.path.join(attribute_output_dir, attribute_name)

                    output_path = os.path.join(attribute_output_dir, f"{image_id}_{part_name}.jpg")
                    cropped_image.save(output_path)

        print(f"Saved cropped parts from {image_id} {image_name}")
