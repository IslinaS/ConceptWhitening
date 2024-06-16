import os
import pandas as pd

"""
The objective here is to make the parquet that will be of format:

image id, class, high level concept, low level concept, coords, certainty

Here the high level concept is the part, and low level the attribute.
Coords are computed from a preset window size, this can be changed easily.
"""

CUB_PATH = "/usr/xtmp/aak61/CUB_200_2011"
HIGH_LEVEL_ASSIGNMENTS = {
    
}

df = pd.DataFrame()

# Step 1, make the dict
train_test_split = {}
high_level = {}
low_level = {}
