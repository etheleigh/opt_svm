import numpy as np
import bz2file as bz2
from typing import Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

from src.SMO import SVM_classifier


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


def tune_hyperparameter_C(X, y, kernel='linear', metric='accuracy', C_range=[0.1, 1, 10, 100], cv=5):
    """
    Tunes the hyperparameter C for the SVM_classifier using cross-validation based on the selected metric.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target labels.
    kernel (str): The kernel type ('linear' or 'rbf').
    metric (str): The metric to optimize during hyperparameter tuning ('accuracy', 'precision', 'recall', 'f1').
    C_range (list): List of C values to try during cross-validation.
    cv (int): Number of cross-validation folds.

    Returns:
    float: Best value of C found during hyperparameter tuning.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    best_C = None
    best_score = -np.inf if metric in ['precision', 'recall', 'f1'] else np.inf

    for C in C_range:
        scores = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            clf = SVM_classifier(X_train, y_train, kernel=kernel, C=C)
            clf.fit()

            y_pred, _ = clf.predict(X_val)
            score = calculate_score(y_val, y_pred, metric)
            scores.append(score)

        mean_score = np.mean(scores)

        if (metric in ['precision', 'recall', 'f1'] and mean_score > best_score) or \
                (metric == 'accuracy' and mean_score < best_score):
            best_score = mean_score
            best_C = C

    return best_C


def calculate_score(y_true, y_pred, metric):
    """
    Calculates the score based on the specified metric.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    metric (str): The metric to calculate ('accuracy', 'precision', 'recall', 'f1').

    Returns:
    float: Score based on the metric.
    """
    if metric == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric == 'precision':
        return precision_score(y_true, y_pred)
    elif metric == 'recall':
        return recall_score(y_true, y_pred)
    elif metric == 'f1':
        return f1_score(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Please choose from 'accuracy', 'precision', 'recall', 'f1'.")