from PIL import Image
import os

# Define paths
PARTS_FILE = "parts/parts.txt"
PART_LOCS_FILE = "parts/part_locs.txt"
ATTRIBUTES_FILE = "attributes/attributes.txt"
IMAGE_ATTRIBUTE_LABELS_FILE = "attributes/image_attribute_labels.txt"
BOUNDING_BOX_FILE = "bounding_boxes.txt"
IMAGES_FILE = "images.txt"
TRAIN_TEST_SPLIT_FILE = "train_test_split.txt"
IMAGES_DIR = "images/"

CONCEPT_DIR = "concept"
CONCEPT_DIR_TRAIN = CONCEPT_DIR + "_train/"
CONCEPT_DIR_TEST = CONCEPT_DIR + "_test/"

BOUNDING_BOX_SIZE = 100

# Create output directory if it doesn't exist
if not os.path.exists(CONCEPT_DIR_TRAIN):
    os.makedirs(CONCEPT_DIR_TRAIN)
if not os.path.exists(CONCEPT_DIR_TEST):
    os.makedirs(CONCEPT_DIR_TEST)

# Read images file to create a mapping of image_id to image name
IMAGES = {}
with open(IMAGES_FILE, "r") as file:
    for line in file:
        image = line.strip().split()
        image_id = int(image[0])
        image_name = image[1]
        
        IMAGES[image_id] = image_name

# Read bounding boxes file to create a mapping of image_id to cropped images
CROPPED_IMAGES = {}
with open(BOUNDING_BOX_FILE, "r") as file:
    for line in file:
        image_id, x, y, width, height = map(int, line.strip().split())
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
PARTS = {}
with open(PARTS_FILE, "r") as file:
    for line in file:
        part_id, part_name = line.strip().split()
        PARTS[part_id] = part_name        
        
    # Create output directory for this part if it doesn't exist
        part_output_train_dir = os.path.join(CONCEPT_DIR_TRAIN, part_name)
        if not os.path.exists(part_output_train_dir):
            os.makedirs(part_output_train_dir)

        part_output_test_dir = os.path.join(CONCEPT_DIR_TEST, part_name)
        if not os.path.exists(part_output_test_dir):
            os.makedirs(part_output_test_dir)

# Read parts file to create a mapping of part_id to part name
ATTRIBUTES = {}
with open(ATTRIBUTES_FILE, "r") as file:
    for line in file:
        attribute_id, attribute_name = line.strip().split()
        attribute_category, data = attribute_name.split("::")
        ATTRIBUTES[part_id] = (attribute_category, data)       
        
    # Create output directory for this part if it doesn't exist
        attribute_output_train_dir = os.path.join(CONCEPT_DIR_TRAIN, attribute_name)
        if not os.path.exists(attribute_output_train_dir):
            os.makedirs(attribute_output_train_dir)

        attribute_output_test_dir = os.path.join(CONCEPT_DIR_TEST, attribute_name)
        if not os.path.exists(attribute_output_test_dir):
            os.makedirs(attribute_output_test_dir)

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
    "has_head_pattern": "general",
    "has_size": "general",
    "has_shape": "general",
    "has_primary_color": "general"
}

PART_LOCATIONS = {}
IMAGE_ATTRIBUTES = {}

# Read part locations file
with open(PART_LOCS_FILE, "r") as file:
    for line in file:
        parts = line.strip().split()
        image_id = int(parts[0])
        part_id = parts[1]
        x = int(parts[2])
        y = int(parts[3])
        visible = int(parts[4])

        if image_id not in PART_LOCATIONS:
            PART_LOCATIONS[image_id] = {}
        if visible:
            PART_LOCATIONS[image_id][part_id] = (x, y) 

# Read image attribute labels file
with open(IMAGE_ATTRIBUTE_LABELS_FILE, "r") as file:
    for line in file:
        attributes = line.strip().split()
        image_id, attribute_id, present = map(int, attributes[:3])

        if image_id not in IMAGE_ATTRIBUTES:
            IMAGE_ATTRIBUTES[image_id] = {}
        if present:
            IMAGE_ATTRIBUTES[image_id].append(attribute_id)

# Read train_test_split.txt file
with open(TRAIN_TEST_SPLIT_FILE, "r") as file:
    lines = file.readlines()

# Process each image
for line in lines:
    image_id, is_training_image = map(int, line.strip().split())
    image_name = IMAGES[image_id]
    split = "train" if is_training_image else "test"

    # Open the image
    image_path = os.path.join(IMAGES_DIR, image_name)
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        continue

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
        part_output_dir = os.path.join(f"{CONCEPT_DIR}_{split}/", part_name)
        output_path = os.path.join(part_output_dir, f"{image_id}.png")
        cropped_part.save(output_path)

    # Save crops for each attribute
    for attribute_id in attributes:
        # Determine the part corresponding to the attribute (if any)
        attribute_category, data = ATTRIBUTES[attribute_id]
        part_names = ATTRIBUTE_TO_PART_MAP[attribute_category]

        if type(part_names) is not list:
            part_names = [part_names]

        for part_name in part_names:
            if part_name in part_images or part_name == "general":
                cropped_image = CROPPED_IMAGES[image_id] if part_name == "general" else part_images[part_name]

                # Save cropped image in output directory
                attribute_output_dir = os.path.join(f"{CONCEPT_DIR}_{split}/", f"{attribute_category}_{data}")
                output_path = os.path.join(attribute_output_dir, f"{image_id}.jpg")
                cropped_image.save(output_path)
