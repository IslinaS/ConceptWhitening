from torch.utils.data import Dataset
from PIL import Image

# It is, from a code cleanliness standpoint, beneficial to have two different types of datasets.
# Backbone returns the image and class, which is used to improve accuracy
# CW returns the image and the concept bounding box, which is used to 

class BackboneDataset(Dataset):
    """
    The backbone dataset on a __getitem__ call will return the image and the class label.
    This is distinct from the CWDataset, which returns different attributes.
    """
    def __init__(self, image_folder, annotations, transform=None):
        """
        Annotation file is crucial. It is the parquet with columns image_id and class (plus others).
        There can be any number of classes and image_ids here.
        """
        self.image_folder = image_folder
        self.annotations = annotations.copy()
        self.annotations = self.annotations.drop_duplicates(subset='image_id', keep='first')
        self.image_ids = self.annotations['image_id'].values
        self.labels = self.annotations['class'].values
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = self.labels[idx]
        img_path = f"{self.image_folder}/{img_id}.jpg"
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, label
    

class CWDataset(Dataset):
    """
    The CW dataset on a __getitem__ call will return the image and the bounding box for the concept.
    Unlike BackboneDataset, this does NOT return the class labels
    """
    def __init__(self, image_folder, annotations, transform=None):
        """
        Annotation file is crucial. It is the parquet with columns image_id and class (plus others).
        For the CWDataset, it must ONLY contain rows with the same value for.

        Difference between these sets is Backbone contains one occurrence of each image_id,
        CW contains only image_ids with a given concept.
        """
        self.image_folder = image_folder
        self.annotations = annotations.copy()
        self.image_ids = self.annotations["image_id"].values
        self.bboxes = self.annotations["coords"].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        bbox = self.bboxes[idx]
        img_path = f"{self.image_folder}/{img_id}.jpg"
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, bbox
