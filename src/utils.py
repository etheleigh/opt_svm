import numpy as np
import pandas as pd
import bz2file as bz2
import os
from typing import Tuple, Optional

def helper_function():
    return "This is a helper function"

def read_gisette_data(file_path: str, max_lines: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the Gisette data from a bz2 compressed file.

    Parameters:
    - file_path: str, the relative or absolute path to the bz2 file.
    - max_lines: Optional[int], maximum number of lines to read from the file. If None, reads all lines.

    Returns:
    - labels: numpy array of shape (num_samples,)
    - features: numpy array of shape (num_samples, 5000)
    """
    # Initialize lists to store the labels and features
    labels = []
    features = []

    # Open and read the compressed file
    with bz2.open(file_path, 'rb') as file:
        for i, line in enumerate(file):
            if max_lines is not None and i >= max_lines:
                break
            # Decode the line from bytes to string
            line = line.decode('utf-8').strip()

            # Split the line into label and features
            parts = line.split()
            label = int(parts[0])
            feature_pairs = parts[1:]

            # Extract features
            feature_vector = np.zeros(5000)
            for pair in feature_pairs:
                index, value = pair.split(':')
                feature_vector[int(index) - 1] = float(value)

            # Append label and feature vector to lists
            labels.append(label)
            features.append(feature_vector)

    # Convert lists to numpy arrays
    labels = np.array(labels)
    features = np.array(features)

    return labels, features