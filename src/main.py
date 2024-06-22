from utils import read_gisette_data, tune_hyperparameter_C
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from SMO import SVM_classifier

def main():

    # Read gisette dataset
    MAX_LINES = 4500

    # File paths relative to the base directory
    file_path_train = os.path.join('..', 'data', 'gisette_scale.bz2')
    file_path_test = os.path.join('..', 'data', 'gisette_scale.t.bz2')

    y_train, X_train = read_gisette_data(file_path_train, max_lines=MAX_LINES)
    y_test, X_test = read_gisette_data(file_path_test, max_lines=MAX_LINES)

    # Ask user if they want to perform hyperparameter tuning
    perform_tuning = input("Do you want to perform hyperparameter tuning? (Y/N): ").strip().lower()

    if perform_tuning == 'y':
        print("Performing hyperparameter tuning...")
        # Perform hyperparameter tuning
        best_C = tune_hyperparameter_C(X_train, y_train)
        kernel = 'linear'  # Assuming linear kernel for Gisette dataset
        print(f"Hyperparameter tuning complete. Best C found: {best_C}")

    else:
        # Use default or user-provided kernel and C
        kernel = input("Enter kernel type (linear/rbf): ").strip().lower()
        best_C = float(input("Enter the value of C: ").strip())

    print("Training the SVM classifier...")
    # Initialize and train SVM classifier
    clf = SVM_classifier(X_train, y_train, kernel=kernel, C=best_C)
    clf.fit()
    print("Training complete.")

    print("Predicting on test set...")
    # Predict on test set
    y_pred, _ = clf.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print results
    print("\nResults on test set:")
    print(f"Accuracy of SVM: {accuracy:.4f}")
    print(f"Precision of SVM: {precision:.4f}")
    print(f"Recall of SVM: {recall:.4f}")
    print(f"F1 score of SVM: {f1:.4f}")

if __name__ == "__main__":
    main()
