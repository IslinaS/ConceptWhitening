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
    def __init__(self, annotations: pd.DataFrame, transform=None):
        """
        Annotation file is crucial. It is the parquet with columns image_id and class (plus others).
        There can be any number of classes and image_ids here.

        Params:
        -------
        - annotations (pd.DataFrame): Dataframe of labeled concepts
        - transform (TorchVision Transform): Transform to be applied to each image (mainly to make it a tensor)
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

        return image, path, label


class CWDataset(Dataset):
    """
    The CW dataset on a __getitem__ call will return the image and the bounding box for the concept.
    Unlike BackboneDataset, this does NOT return the class labels.

    Each CW dataset is associated with only one concept, and one concept per data loader is needed.
    """
    def __init__(self, annotations: pd.DataFrame, low_level, mode, transform=None):
        """
        Initialize a CWDataset.

        Params:
        -------
        - annotations (pd.DataFrame): Dataframe of labeled concepts
        - low_level (dictionary): Mapping of each low level concept to an index number. See data/json/low_level.json.
        - mode (int): Concept index to keep
        - transform (TorchVision Transform): Transform to be applied to each image (mainly to make it a tensor)
        """
        # Data itself. We make sure we only have original and unaugmented images
        # since concepts are not labeled in augmented ones.
        self.annotations = annotations.copy()
        self.annotations = self.annotations[self.annotations["augmented"] == 0]

        # Used to create unknown concepts
        self.low_level = low_level

        # Used for making images tensors
        self.transform = transform

        # Make our low level concepts indices, then filter the data to only include those
        self._index_low_level()
        self.annotations = self.annotations[self.annotations["low_level"] == mode]

    def _index_low_level(self):
        """
        Replace the names of the low level concepts with their index values.
        """
        # TODO: Fix warning associated with this
        # FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version.
        # To retain the old behavior, explicitly call `result.infer_objects(copy=False)`.
        # To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
        self.annotations["low_level"] = self.annotations["low_level"].replace(self.low_level).infer_objects(copy=False)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        path = self.annotations["path"].iloc[idx]
        bbox = self.annotations["coords"].iloc[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, bbox

    @staticmethod
    def make_free_concepts(annotations: pd.DataFrame, n_free, low_level: dict, high_level: dict, mappings: dict):
        """
        Responsible for allocating the "free" low level concepts to each high level concept.
        This is performed by adding `n_free` "free" low level concepts to its associated high level
        concept in all data points where that high level concept was visible.

        This is a STATIC method and is used to preprocess the data.

        Params:
        -------
        - annotations (pd.DataFrame): The annotated data
        - n_free (int): Number of "free" low level concepts to allocate to each high level concept
        - low_level (dictionary): Mapping of each low level concept to an index number. See data/json/low_level.json.
        - high_level (dictionary): Mapping of an index to a high level concept. See data/json/high_level.json.
        - mappings (dictionary): Mapping of each low level concept to a high level concept. See data/json/mappings.json.

        Returns:
        --------
        - pd.DataFrame: The original annotations df, but with the free concept rows added
        - dictionary: The original low_level dict, but with the free concept mappings added
        - dictionary: The original mappings dict, but with the free concept mappings added
        """
        # Only need to do this if we have free concepts
        if n_free == 0:
            return

        # Create new rows for each high_level, n_free times
        new_rows = []
        new_mode = max(low_level.values()) + 1  # This will be the new "index" of the concept

        for hl in set(high_level.values()):
            # Filter rows that include the current high_level. We get 1 row per image ID so we can copy its other values
            hl_rows: pd.DataFrame = annotations[annotations['high_level'] == hl].copy()
            hl_rows = hl_rows.drop_duplicates(subset='image_id', keep='first')

            # Generate n_free new low_level concepts for each high_level
            for i in range(1, n_free + 1):
                new_low_level = f"{hl}_free_{i}"
                modified_rows = hl_rows.copy()
                modified_rows['low_level'] = new_low_level
                new_rows.append(modified_rows)

                # Add the new concept to the low_level dict for indexing later
                low_level[new_low_level] = new_mode
                mappings[new_low_level] = hl
                new_mode += 1

        # Concatenate all new rows to the original annotations dataframe
        if new_rows:
            new_rows_df = pd.concat(new_rows, ignore_index=True)
            annotations = pd.concat([annotations, new_rows_df], ignore_index=True)

        return annotations, low_level, mappings

    @staticmethod
    def generate_low_level_cw_mappings(low_level: dict, mappings: dict):
        """
        In a CW module, the concepts are indexed based on their order in `low_level`,
        but translated to start from index 0. This function generates two mappings:
        - From each high level concept (by name) to low level concepts (by index) belonging to that high level concept.
        This mapping is used to generate the concept indicator matrix and the latent space mappings in the CW layer.
        - From each translated concept index to its concept name

        This is a STATIC method and is used to preprocess the data.

        Params:
        -------
        - low_level (dictionary): Mapping of each low level concept to an index number. See data/json/low_level.json.
        - mappings (dictionary): Mapping of each low level concept to a high level concept. See data/json/mappings.json.

        Returns:
        --------
        - dictionary: Mapping from high level concept to low level concept
        - dictionary: Mapping from low level concept index (as used in the CW layer) to its name
        """
        min_concept = min(low_level.values())

        # Reverse the mappings dictionary to map high level concept to low level indices
        high_to_low: dict[str, list[int]] = {}
        low_level_names: dict[int, str] = {}

        for low_concept, high_concept in mappings.items():
            if high_concept not in high_to_low:
                high_to_low[high_concept] = []

            concept_cw_index = low_level[low_concept] - min_concept
            high_to_low[high_concept].append(concept_cw_index)
            low_level_names[concept_cw_index] = low_concept

        return high_to_low, low_level_names
