from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

# It is, from a code cleanliness standpoint, beneficial to have two different types of datasets.
# Backbone returns the image and class, which is used to improve accuracy
# CW returns the image and the concept bounding box, which is used to 

class BackboneDataset(Dataset):
    """
    The backbone dataset on a __getitem__ call will return the image and the class label.
    This is distinct from the CWDataset, which returns different attributes.
    """
    def __init__(self, annotations, transform=None):
        """
        Annotation file is crucial. It is the parquet with columns image_id and class (plus others).
        There can be any number of classes and image_ids here.
        """
        self.annotations = annotations.copy()
        self.annotations = self.annotations.drop_duplicates(subset='image_id', keep='first')
        self.paths = self.annotations['path'].values
        self.labels = self.annotations['class'].values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, label
    

class CWDataset(Dataset):
    """
    The CW dataset on a __getitem__ call will return the image and the bounding box for the concept.
    Unlike BackboneDataset, this does NOT return the class labels
    """
    def __init__(self, annotations, high_level, low_level, n_free=2):
        """
        Annotation file is crucial. It is the parquet with columns image_id and class (plus others).
        For the CWDataset, it must ONLY contain rows with the same value for.

        Difference between these sets is Backbone contains one occurrence of each image_id,
        CW contains only image_ids with a given concept.
        """
        # The mode determines which concepts are returned
        self.mode = -1

        # Data itself
        self.annotations = annotations.copy()

        # Used to create unknown concepts
        self.high_level = high_level
        self.low_level = low_level
        self.n_free = n_free
        self.make_free_concepts()

    def make_free_concepts(self):
        # Create new rows for each high_level, n_free times
        new_rows = []

        for hl in self.high_level:
            # Filter rows that include the current high_level
            hl_rows = self.annotations[self.annotations['high_level'] == hl].copy()
            hl_rows = hl_rows.drop_duplicates(subset='image_id', keep='first')
            # Generate n_free new low_level concepts for each high_level
            for i in range(1, self.n_free + 1):
                new_low_level = f"{hl}_free_{i}"
                modified_rows = hl_rows.copy()
                modified_rows['low_level'] = new_low_level
                new_rows.append(modified_rows)

        # Concatenate all new rows to the original annotations dataframe
        if new_rows:
            new_rows_df = pd.concat(new_rows, ignore_index=True)
            self.annotations = pd.concat([self.annotations, new_rows_df], ignore_index=True)

    def mode(self, mode):
        self.mode = mode

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, idx):
        # Only select items that have the current specified low level concept
        df = self.annotations[self.annotations["low_level"] == self.mode]
        path = df["path"].iloc[idx]
        bbox = df["coords"].iloc[idx]
        
        image = Image.open(path).convert("RGB")

        return image, bbox
