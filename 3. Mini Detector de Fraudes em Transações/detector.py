# detector.py
# Main script for the fraud detection pipeline.

import os
import time 
import pandas as pd
import numpy as np

# Import functions from our data_input module
from src.data_input import (load_csv_data,
                            generate_synthetic_data_scratch,
                            engineer_features_from_data,
                            augment_data_smote)

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV # Import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, f1_score # Import make_scorer, f1_score

def run_fraud_detection_pipeline():
    """
    Executes the fraud detection pipeline with GridSearchCV for hyperparameter tuning.
    """
    print("--- Initiating Fraud Detection Pipeline with GridSearchCV ---")

    X_final, y_final = None, None
    USE_REAL_DATA = True # <<< Set to True to use creditcard.csv, False for synthetic
    APPLY_SMOTE_TO_TRAIN = True # <<< Set to True to apply SMOTE to the training set

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
    else: 
        print("\nSelected: Option B - Generating synthetic data.")
        X_final, y_final = generate_synthetic_data_scratch(
            n_samples=20000,
            n_features=20,
            class_weights=[0.99, 0.01],
            random_state=42
        )
        print("(Feature engineering is typically skipped for generic synthetic data)")

    if X_final is None or y_final is None:
        print("\n--- Failed to obtain data. Exiting pipeline. ---")
        return

    print(f"\nFinal X shape before split: {X_final.shape}")
    print(f"Final y distribution before split:\n{y_final.value_counts(normalize=True)}")

    print("\nSplitting data into training and testing sets...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final,
            test_size=0.25,
            random_state=42,
            stratify=y_final
        )
        print("Data split successfully.")
        print(f"Original train target distribution:\n{y_train.value_counts(normalize=True)}")
    except Exception as e:
        print(f"Error during data splitting: {e}")
        return

    X_train_processed = X_train.copy()
    y_train_processed = y_train.copy()

    if APPLY_SMOTE_TO_TRAIN:
        print("\nApplying SMOTE to the training set...")
        X_train_processed, y_train_processed = augment_data_smote(X_train, y_train, random_state=42)
        if X_train_processed.shape[0] > X_train.shape[0]:
             print(f"X_train shape after SMOTE: {X_train_processed.shape}")
             print(f"y_train distribution after SMOTE:\n{y_train_processed.value_counts(normalize=True)}")
    else:
        print("\nSMOTE not applied to the training set.")

    # --- HYPERPARAMETER TUNING WITH GridSearchCV ---
    print("\nInitiating Hyperparameter Tuning with GridSearchCV...")

    param_grid = {
        'n_estimators': [100, 150],       # 2 values
        'max_depth': [20, None],          # 2 values #best found = none
        'min_samples_split': [5, 10],     # 2 values 
        'min_samples_leaf': [2, 4],       # 2 values
    }
    # If SMOTE was not applied, 'balanced' is a good candidate for class_weight.
    # If SMOTE was applied, the training data is more balanced, so 'None' or not including
    # class_weight in the grid might be fine. Let's make it conditional for the base estimator.
    base_rf_params = {'random_state': 42}
    if not APPLY_SMOTE_TO_TRAIN:
        # If not using SMOTE, it's often good to ensure 'balanced' is an option for class_weight
        # If 'class_weight' is in param_grid, GridSearchCV will test it.
        # If not, we can set a default for the base estimator.
        # For simplicity in the grid, let's assume if SMOTE isn't used, 'balanced' is desirable.
        # The grid can still override this if 'class_weight' is added to param_grid.
        if 'class_weight' not in param_grid: # If not tuning class_weight in grid
             base_rf_params['class_weight'] = 'balanced'


    # Define the scorer to optimize for F1-score of the fraud class (Class 1)
    f1_scorer_class1 = make_scorer(f1_score, pos_label=1, average='binary')

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(**base_rf_params), # Base model with random_state
        param_grid=param_grid,
        scoring=f1_scorer_class1,
        cv=3,  # Number of cross-validation folds. Start with 2 or 3 for speed.
        verbose=2, # Prints updates during the search
        n_jobs=-1  # Use all available CPU cores
    )

    print("Fitting GridSearchCV (this may take a while)...")
    start_tuning_time = time.time()
    try:
        grid_search.fit(X_train_processed, y_train_processed)
        end_tuning_time = time.time()
        tuning_duration = end_tuning_time - start_tuning_time
        print(f"GridSearchCV fitting completed in {tuning_duration:.2f} seconds.")

        print("\nBest hyperparameters found by GridSearchCV:")
        print(grid_search.best_params_)
        print(f"\nBest F1-score (for Class 1) on validation sets during CV: {grid_search.best_score_:.4f}")

        # The best model is already refitted on the whole training data (X_train_processed)
        model = grid_search.best_estimator_
        print("Best model obtained from GridSearchCV.")

    except Exception as e:
        print(f"Error during GridSearchCV: {e}")
        print("--- Exiting pipeline. ---")
        return
    
    # --- MODEL EVALUATION (using the best model from GridSearchCV) ---
    print("\nEvaluating the best model on the test set...")
    try:
        predictions = model.predict(X_test)
    except Exception as e:
        print(f"Error during prediction with the best model: {e}")
        return

    print("\n--- Classification Report (Optimized Model) ---")
    try:
        report = classification_report(y_test, predictions, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Error generating classification report: {e}")

    print("\n--- Confusion Matrix (Optimized Model) ---")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    try:
        tn, fp, fn, tp = (0,0,0,0) 
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

    print("\n--- Fraud Detection Pipeline Complete ---")

if __name__ == "__main__":
    run_fraud_detection_pipeline()