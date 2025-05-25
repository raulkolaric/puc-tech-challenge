# detector.py
# Main script for the fraud detection pipeline, now refactored to be callable
# and accept a classification threshold.

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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # Removed f1, precision, recall from here as they are in report_dict

# This is the function that will be called by run_experiments.py
def run_fraud_detection_pipeline_with_params(
    use_real_data=True,
    apply_smote_to_train=True,
    model_params_override=None,
    classification_threshold=0.5 # New parameter with default
    ):
    """
    Executes the fraud detection pipeline with specified configurations
    and returns key performance metrics.

    Args:
        use_real_data (bool): True to load real data, False for synthetic.
        apply_smote_to_train (bool): True to apply SMOTE to the training data.
        model_params_override (dict): Dictionary of parameters for RandomForestClassifier.
        classification_threshold (float): Threshold to use for converting probabilities to class predictions.

    Returns:
        dict: A dictionary containing performance metrics,
              or None if the pipeline fails before evaluation.
    """
    print("\n--- Executando Pipeline de Detecção de Fraudes com Parâmetros ---")
    # ... (Data loading and initial feature engineering logic - X_final, y_final obtained) ...
    # This part is mostly the same as before, controlled by use_real_data
    # and calls functions from src.data_input
    X_final, y_final = None, None
    if use_real_data:
        # ... (code to load real data and engineer features) ...
        print("Opção de Dados: Carregando dados reais.")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path_real_data = os.path.join(script_dir, 'data', 'creditcard.csv')
        target_col = 'Class'
        X_initial, y_initial = load_csv_data(file_path=file_path_real_data, target_column_name=target_col)

        if X_initial is not None:
            print("Aplicando engenharia de features...")
            X_processed = engineer_features_from_data(X_initial)
            if X_processed is not None and y_initial is not None:
                X_final = X_processed
                y_final = y_initial
            else:
                print("Engenharia de features falhou.")
        else:
            print("Carregamento inicial dos dados falhou.")
    else: 
        print("Opção de Dados: Gerando dados sintéticos.")
        X_final, y_final = generate_synthetic_data_scratch(
            n_samples=20000, n_features=20, class_weights=[0.99, 0.01], random_state=42
        )
        print("(Engenharia de features geralmente pulada para dados sintéticos genéricos)")

    if X_final is None or y_final is None:
        print("--- Falha ao obter os dados. Pipeline encerrado. ---")
        return None
    # ... (Print X_final shape and y_final distribution) ...

    # ... (Data splitting logic - X_train, X_test, y_train, y_test obtained) ...
    # This part is the same.
    print("\nDividindo os dados em conjuntos de treino e teste...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.25, random_state=42, stratify=y_final
        )
        print("Dados divididos com sucesso.")
        print(f"Distribuição do alvo no treino original:\n{y_train.value_counts(normalize=True)}")
    except Exception as e:
        print(f"Erro durante a divisão dos dados: {e}")
        return None
        
    X_train_processed = X_train.copy()
    y_train_processed = y_train.copy()

    if apply_smote_to_train:
        # ... (SMOTE application logic - X_train_processed, y_train_processed updated) ...
        # This part is the same.
        print("Aplicando SMOTE ao conjunto de treino...")
        X_train_processed, y_train_processed = augment_data_smote(X_train, y_train, random_state=42)
        if X_train_processed.shape[0] > X_train.shape[0]:
             print(f"Shape de X_train após SMOTE: {X_train_processed.shape}")
             print(f"Distribuição de y_train após SMOTE:\n{y_train_processed.value_counts(normalize=True)}")
    else:
        print("SMOTE não aplicado ao conjunto de treino.")


    # --- MODEL TRAINING ---
    # Use provided model_params_override or defaults
    current_model_params = { # Basic defaults if nothing is passed
        'n_estimators': 100, 'random_state': 42, 'verbose': 0, 'n_jobs': -1
    } 
    if model_params_override: # If parameters are passed from ui.py, update the defaults
        current_model_params.update(model_params_override)
    
    # Adjust class_weight based on SMOTE application and if not already in overridden params
    if not apply_smote_to_train and 'class_weight' not in current_model_params:
        current_model_params['class_weight'] = 'balanced'
        print("Usando class_weight='balanced' no modelo.")
    # ... (warnings about class_weight and SMOTE) ...

    model = RandomForestClassifier(**current_model_params) # Unpack the chosen parameters

    print(f"\nTreinando RandomForestClassifier com parâmetros: {current_model_params}...")
    # ... (model.fit() and timing logic - same as before) ...
    start_training_time = time.time()
    try:
        model.fit(X_train_processed, y_train_processed)
        end_training_time = time.time()
        training_duration = end_training_time - start_training_time
        print(f"Modelo treinado com sucesso em {training_duration:.2f} segundos.")
    except Exception as e:
        print(f"Erro durante o treinamento do modelo: {e}")
        return None

    # --- MODEL EVALUATION ---
    print("\nAvaliando o modelo no conjunto de teste...")
    try:
        proba_predictions = model.predict_proba(X_test)[:, 1] # Get probabilities for class 1 (fraud)

        # Apply the custom or default threshold
        print(f"Utilizando limiar de classificação: {classification_threshold}")
        predictions = (proba_predictions >= classification_threshold).astype(int) # Convert probas to 0/1 based on threshold

    except Exception as e:
        print(f"Erro durante a predição: {e}")
        return None

    print("\n--- Relatório de Classificação ---")
    try:
        # Get full classification report as a dictionary to extract specific metrics
        report_dict = classification_report(y_test, predictions, zero_division=0, output_dict=True)
        print(classification_report(y_test, predictions, zero_division=0)) # Print human-readable report
    except Exception as e:
        print(f"Erro ao gerar relatório de classificação: {e}")
        report_dict = {} # Return empty dict on error to avoid crash later

    cm = confusion_matrix(y_test, predictions)
    print("\n--- Matriz de Confusão ---")
    print(cm)
    
    tn, fp, fn, tp = 0,0,0,0 # Initialize
    try:
        # Robust unpacking of confusion matrix
        if cm.size == 4: 
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1 and len(np.unique(y_test)) == 1 : 
             if y_test.iloc[0] == 0: tn = cm[0,0]
             else: tp = cm[0,0]
    except Exception: 
        print("Aviso: Não foi possível desempacotar totalmente a matriz de confusão.")

    # Prepare metrics to return to the caller (ui.py)
    metrics_to_return = {
        "accuracy": report_dict.get("accuracy", "N/A"), # Overall accuracy
        "f1_class1": report_dict.get("1", {}).get("f1-score", "N/A"), # F1 for fraud
        "precision_class1": report_dict.get("1", {}).get("precision", "N/A"), # Precision for fraud
        "recall_class1": report_dict.get("1", {}).get("recall", "N/A"), # Recall for fraud
        "support_class1": report_dict.get("1", {}).get("support", "N/A"),
        "f1_class0": report_dict.get("0", {}).get("f1-score", "N/A"), # F1 for non-fraud
        "precision_class0": report_dict.get("0", {}).get("precision", "N/A"),
        "recall_class0": report_dict.get("0", {}).get("recall", "N/A"),
        "support_class0": report_dict.get("0", {}).get("support", "N/A"),
        "confusion_matrix_class1": { # Details for fraud class
            "tp": int(tp), # True Positives for fraud
            "fn": int(fn), # False Negatives for fraud (missed frauds)
            "fp": int(fp), # False Positives for fraud (non-frauds flagged as fraud)
            "tn": int(tn)  # True Negatives (non-frauds correctly identified as non-fraud from fraud's perspective)
        },
        "training_duration_seconds": training_duration,
        "classification_threshold": classification_threshold # Also log the threshold used
    }
    
    print("\n--- Pipeline de Detecção de Fraudes Concluído (Execução Parametrizada) ---")
    return metrics_to_return # Return the dictionary of metrics

# This block allows detector.py to still be run directly for testing its own pipeline
if __name__ == "__main__":
    print("Executando detector.py diretamente (usando configurações padrão para teste)...")
    default_model_params = { # Example default parameters for direct run
        'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2,
        'min_samples_leaf': 1, 'random_state': 42, 'verbose': 0, 'n_jobs': -1 
    }
    results = run_fraud_detection_pipeline_with_params(
        use_real_data=True, # Default to synthetic for quick direct test
        apply_smote_to_train=True, 
        model_params_override=default_model_params,
        classification_threshold=0.5 # Default threshold for direct run
    )
    if results: # If results were returned (no major error)
        print("\nResultados da execução direta de detector.py:")
        # Loop through and print results nicely
        for key, value in results.items():
            if isinstance(value, dict): # If it's the confusion_matrix_class1 dict
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else: # For floats or other types
                print(f"  {key}: {value if isinstance(value, int) else f'{value:.4f}'}")