# detector.py
# Main script for the fraud detection pipeline.

import os
import time # For timing operations
import pandas as pd # For inspecting DataFrames if needed
import numpy as np # For np.unique in confusion matrix formatting

# Import functions from our data_input module
from src.data_input import (load_csv_data,
                            generate_synthetic_data_scratch,
                            engineer_features_from_data,
                            augment_data_smote)

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV # GridSearchCV is for optional tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, f1_score


def run_fraud_detection_pipeline():
    """
    Executes the fraud detection pipeline:
    1. Load/Generate Data & Initial Feature Engineering
    2. Split Data
    3. Optional: Apply SMOTE to Training Data
    4. Train Model
    5. Evaluate Model
    6. Display Reports
    """
    print("--- Initiating Fraud Detection Pipeline ---")

    X_final, y_final = None, None

    # --- DATA SOURCE SELECTION ---
    # Change to True to use real data, False for synthetic
    USE_REAL_DATA = True # <<< Set to True to use creditcard.csv

    if USE_REAL_DATA:
        print("\nSelected: Option A - Loading real data.")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path_real_data = os.path.join(script_dir, 'data', 'creditcard.csv')
        target_col = 'Class'
        X_initial, y_initial = load_csv_data(file_path=file_path_real_data, target_column_name=target_col)

        if X_initial is not None:
            print("\nApplying feature engineering...")
            X_processed = engineer_features_from_data(X_initial)
            if X_processed is not None and y_initial is not None:
                X_final = X_processed
                y_final = y_initial
            else:
                print("Feature engineering failed.")
        else:
            print("Initial data loading failed.")
    else: # USE_REAL_DATA is False
        print("\nSelected: Option B - Generating synthetic data.")
        X_final, y_final = generate_synthetic_data_scratch(
            n_samples=20000,
            n_features=20, # Note: engineer_features_from_data expects specific column names
            class_weights=[0.99, 0.01], # Simulates imbalance
            random_state=42
        )
        print("(Feature engineering is typically skipped for generic synthetic data in this setup)")

    if X_final is None or y_final is None:
        print("\n--- Failed to obtain data. Exiting pipeline. ---")
        return

    print(f"\nFinal X shape before split: {X_final.shape}")
    if isinstance(X_final, pd.DataFrame):
        print("First 3 rows of X_final (before split):\n", X_final.head(3))
    print(f"Final y distribution before split:\n{y_final.value_counts(normalize=True)}")

    # --- DATA SPLITTING ---
    print("\nSplitting data into training and testing sets...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final,
            test_size=0.25,
            random_state=42,
            stratify=y_final # Important for imbalanced data
        )
        print("Data split successfully.")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print(f"Original train target distribution:\n{y_train.value_counts(normalize=True)}")
    except Exception as e:
        print(f"Error during data splitting: {e}")
        return

    # --- OPTIONAL: SMOTE ON TRAINING DATA ---
    APPLY_SMOTE_TO_TRAIN = True # <<< Set to True to apply SMOTE to the training set

    X_train_processed = X_train.copy()
    y_train_processed = y_train.copy()

    if APPLY_SMOTE_TO_TRAIN:
        print("\nApplying SMOTE to the training set...")
        X_train_processed, y_train_processed = augment_data_smote(X_train, y_train, random_state=42)
        if X_train_processed.shape[0] > X_train.shape[0]:
             print(f"X_train shape after SMOTE: {X_train_processed.shape}")
             print(f"y_train distribution after SMOTE:\n{y_train_processed.value_counts(normalize=True)}")
        # augment_data_smote prints its own warnings if imblearn is not available or SMOTE fails
    else:
        print("\nSMOTE not applied to the training set.")

    # --- MODEL TRAINING ---
    # RandomForestClassifier parameters
    model_params = {
        'n_estimators': 115,
        'random_state': 42,
        # 'max_depth': 5,
        # 'min_samples_leaf': 2,
        'verbose': 1, # Set to 0 for no scikit-learn messages, 1 or 2 for more
        'n_jobs': -1   # Use all available CPU cores for training
    }

    # Adjust class_weight if not using SMOTE to balance the training data
    if not APPLY_SMOTE_TO_TRAIN:
        model_params['class_weight'] = 'balanced'
        print("Using class_weight='balanced' in the model (SMOTE not applied to train set).")
    else:
        print("Not using class_weight='balanced' (assuming SMOTE balanced the train set).")

    model = RandomForestClassifier(**model_params)

    print("\nTraining RandomForestClassifier...")
    print("(Scikit-learn's 'verbose=1' will show tree building progress below)")
    start_training_time = time.time() # Start timer
    try:
        model.fit(X_train_processed, y_train_processed)
        end_training_time = time.time() # End timer
        training_duration = end_training_time - start_training_time
        print(f"Model trained successfully in {training_duration:.2f} seconds.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # --- MODEL EVALUATION ---
    print("\nEvaluating model on the test set...")
    try:
        predictions = model.predict(X_test)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    print("\n--- Classification Report ---")
    try:
        report = classification_report(y_test, predictions, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Error generating classification report: {e}")

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    try:
        tn, fp, fn, tp = (0,0,0,0) # Default values
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1 and len(np.unique(y_test)) == 1 :
             if y_test.iloc[0] == 0: tn = cm[0,0]
             else: tp = cm[0,0]
        else:
             print("Warning: Confusion matrix has an unexpected format.")

        print(f"True Negatives (Non-Frauds OK): {tn}")
        print(f"False Positives (Non-Frauds predicted as Fraud): {fp}")
        print(f"False Negatives (Frauds predicted as Non-Fraud): {fn}")
        print(f"True Positives (Frauds OK): {tp}")
    except Exception as e_cm:
        print(f"Could not unpack confusion matrix details: {e_cm}")

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # --- Optional: Placeholder for Hyperparameter Tuning (GridSearchCV) ---
    # If you want to run GridSearchCV, you would uncomment and adapt the block
    # I provided in the previous message. It would replace the single model training above.
    # print("\n(Hyperparameter tuning with GridSearchCV can be added here)")

    print("\n--- Fraud Detection Pipeline Complete ---")

# This ensures run_fraud_detection_pipeline() is called only when detector.py is executed directly
if __name__ == "__main__":
    run_fraud_detection_pipeline()