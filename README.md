# SVM Classifier for Gisette Dataset

This contains code to train a Support Vector Machine (SVM) classifier on the Gisette dataset using a custom implementation of the Sequential Minimal Optimization (SMO) algorithm.

## Prerequisites

Before running the code, ensure you have Python installed on your system. It's recommended to use a virtual environment to manage dependencies.

## Installation
    
Install the required Python packages using pip and the provided requirements.txt file:
```bash
pip install -r requirements.txt
```

## Dataset

The Gisette dataset files (gisette_scale.bz2 and gisette_scale.t.bz2) are placed in the `data` directory within the project folder. These files contain pre-scaled and preprocessed data ready for training and testing.

## Run the code
In order to run the code we must move at the src directory.
```bash
cd src/
python src/main.py
```
### 1. Hyperparameter Tuning (Optional)

To perform hyperparameter tuning for the SVM classifier:

Follow the prompts to choose whether to perform hyperparameter tuning. If selected, the script will execute and output the best hyperparameters (C) found during tuning.

### 2. Training the SVM Classifier

Once you have the best hyperparameters (or if you choose not to tune), proceed to train the SVM classifier:


Follow the prompts to specify the kernel type (linear or rbf) and the value of C.

### 3. Viewing Results

After training, the script will evaluate the classifier on the test set and print the following performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score

## Folder Structure
```bash
.
├── data
│   ├── gisette_scale.bz2
│   └── gisette_scale.t.bz2
├── README.md
├── requirements.txt
├── src
    ├── __init__.py
    ├── main.py
    ├── SMO.py
    └── utils.py
```



