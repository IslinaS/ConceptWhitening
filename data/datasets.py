"""
It is, from a code cleanliness standpoint, beneficial to have two different types of datasets.
Backbone returns the image and class, which is used to improve accuracy.
CW returns the image and the concept bounding box, which is used to ???  # TODO: fill in here
"""

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


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
    Unlike BackboneDataset, this does NOT return the class labels.
    """
    def __init__(self, annotations, high_level, low_level, n_free=2, transform=None):
        """
        Annotation file is crucial. It is the parquet with columns image_id and class (plus others).
        For the CWDataset, it must ONLY contain rows with the same value for.

        This dataset supports filtering for a specific concept via the set_mode method.
        """
        # Data itself. We make sure we only have original and unaugmented images.
        self.annotations = annotations.copy()
        self.annotations = self.annotations[self.annotations["augment"] == 0]

        # Used to create unknown concepts
        self.high_level = high_level
        self.low_level = low_level
        self.n_free = n_free
        self.n_concepts = 0  # This is updated

        # Used for making images tensors
        self.transform = transform

        # Configuring the free concepts
        self._make_free_concepts()
        self._index_low_level()
        self._set_n_concepts()

        # Set a default mode, this initializes filtered_df
        self.set_mode(1)

    def set_mode(self, mode):
        """
        This determines which concept is going to be loaded.
        When set to a value, only the concept associated with that value will be available to __getitem__().

        Does NOT enforce illegal values! Do NOT call after putting it into a data loader.
        """
        self.mode = mode
        self.filtered_df = self.annotations[self.annotations["low_level"] == self.mode]

    def _set_n_concepts(self):
        """
        Store the number of low level concepts we have for later use.
        """
        self.n_concepts = len(self.low_level)

    def _make_free_concepts(self):
        """
        Responsible for allocating the "free" low level concepts to each high level concepts.
        This is performed by adding n "free" low level concepts to its associated high level
        concept in all data points where that high level concept was visible.
        """
        # Only need to do this if we have free concepts
        if self.n_free == 0:
            return

        # Create new rows for each high_level, n_free times
        new_rows = []
        new_mode = max(self.low_level.values()) + 1  # This will be the new "index" of the concept

        for hl in self.high_level.values():
            # Filter rows that include the current high_level. We get 1 row per image ID so we can copy its other values
            hl_rows = self.annotations[self.annotations['high_level'] == hl].copy()
            hl_rows = hl_rows.drop_duplicates(subset='image_id', keep='first')

            # Generate n_free new low_level concepts for each high_level
            for i in range(1, self.n_free + 1):
                new_low_level = f"{hl}_free_{i}"
                modified_rows = hl_rows.copy()
                modified_rows['low_level'] = new_low_level
                new_rows.append(modified_rows)

                # Add the new concept to the low_level dict for indexing later
                self.low_level[new_low_level] = new_mode
                new_mode += 1

        # Concatenate all new rows to the original annotations dataframe
        if new_rows:
            new_rows_df = pd.concat(new_rows, ignore_index=True)
            self.annotations = pd.concat([self.annotations, new_rows_df], ignore_index=True)

    def _index_low_level(self):
        """
        Replace the names of the low level concepts with their index values.
        """
        self.annotations["low_level"] = self.annotations["low_level"].replace(self.low_level)

    def __len__(self):
        return self.filtered_df.shape[0]

    def __getitem__(self, idx):
        # Only select items that have the current specified low level concept
        path = self.filtered_df["path"].iloc[idx]
        bbox = self.filtered_df["coords"].iloc[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, bbox
