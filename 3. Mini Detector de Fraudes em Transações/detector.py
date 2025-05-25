# detector.py
# Main script for the fraud detection pipeline, now refactored to be callable.

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
from sklearn.model_selection import train_test_split # GridSearchCV removed for this direct param run
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# This is the function that will be called by run_experiments.py
def run_fraud_detection_pipeline_with_params(
    use_real_data=True,
    apply_smote_to_train=True,
    model_params_override=None,
    run_grid_search=False # Kept for future, but TUI will pass specific params
    ):
    """
    Executes the fraud detection pipeline with specified configurations
    and returns key performance metrics.

    Args:
        use_real_data (bool): True to load real data, False for synthetic.
        apply_smote_to_train (bool): True to apply SMOTE to the training data.
        model_params_override (dict): Dictionary of parameters for RandomForestClassifier.
                                      If None, uses internal defaults.
        run_grid_search (bool): If True, would run GridSearchCV (not implemented in this TUI path).

    Returns:
        dict: A dictionary containing performance metrics, e.g.,
              {'f1_class1': 0.83, 'precision_class1': 0.87, ...}
              Returns None if the pipeline fails before evaluation.
    """
    print("\n--- Executando Pipeline de Detecção de Fraudes com Parâmetros ---")

    X_final, y_final = None, None

    if use_real_data:
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
    else: # use_real_data is False
        print("Opção de Dados: Gerando dados sintéticos.")
        X_final, y_final = generate_synthetic_data_scratch(
            n_samples=20000, # Default for TUI, can be made configurable
            n_features=20,
            class_weights=[0.99, 0.01],
            random_state=42
        )
        print("(Engenharia de features geralmente pulada para dados sintéticos genéricos)")

    if X_final is None or y_final is None:
        print("--- Falha ao obter os dados. Pipeline encerrado. ---")
        return None

    print(f"Shape final de X antes do split: {X_final.shape}")
    print(f"Distribuição final de y antes do split:\n{y_final.value_counts(normalize=True)}")

    print("\nDividindo os dados em conjuntos de treino e teste...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final,
            test_size=0.25,
            random_state=42,
            stratify=y_final
        )
        print("Dados divididos com sucesso.")
    except Exception as e:
        print(f"Erro durante a divisão dos dados: {e}")
        return None

    X_train_processed = X_train.copy()
    y_train_processed = y_train.copy()

    if apply_smote_to_train:
        print("Aplicando SMOTE ao conjunto de treino...")
        X_train_processed, y_train_processed = augment_data_smote(X_train, y_train, random_state=42)
        if X_train_processed.shape[0] > X_train.shape[0]:
             print(f"Shape de X_train após SMOTE: {X_train_processed.shape}")
             print(f"Distribuição de y_train após SMOTE:\n{y_train_processed.value_counts(normalize=True)}")
    else:
        print("SMOTE não aplicado ao conjunto de treino.")

    # --- MODEL TRAINING ---
    # Use provided model_params_override or defaults
    current_model_params = {
        'n_estimators': 100, 'random_state': 42, 'verbose': 0, 'n_jobs': -1
    } # Basic defaults
    if model_params_override:
        current_model_params.update(model_params_override)
    
    # Adjust class_weight if not applying SMOTE and not overridden
    if not apply_smote_to_train and 'class_weight' not in current_model_params:
        current_model_params['class_weight'] = 'balanced'
        print("Usando class_weight='balanced' no modelo (SMOTE não aplicado ao treino e não especificado).")
    elif 'class_weight' in current_model_params and current_model_params['class_weight'] is None and not apply_smote_to_train :
        # If user explicitly set class_weight to None, but SMOTE is also off, they might want 'balanced'
        print("Aviso: class_weight é None e SMOTE não foi aplicado. Considere 'balanced' para desbalanceamento.")
    elif apply_smote_to_train and current_model_params.get('class_weight') == 'balanced':
        print("Aviso: SMOTE foi aplicado, class_weight='balanced' pode ser redundante.")


    model = RandomForestClassifier(**current_model_params)

    print(f"\nTreinando RandomForestClassifier com parâmetros: {current_model_params}...")
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
        predictions = model.predict(X_test)
    except Exception as e:
        print(f"Erro durante a predição: {e}")
        return None

    print("\n--- Relatório de Classificação ---")
    try:
        # Get full classification report as a dictionary
        report_dict = classification_report(y_test, predictions, zero_division=0, output_dict=True)
        print(classification_report(y_test, predictions, zero_division=0)) # Print for user
    except Exception as e:
        print(f"Erro ao gerar relatório de classificação: {e}")
        report_dict = {} # Empty dict on error

    cm = confusion_matrix(y_test, predictions)
    print("\n--- Matriz de Confusão ---")
    print(cm)
    
    tn, fp, fn, tp = 0,0,0,0
    try:
        if cm.size == 4: 
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1 and len(np.unique(y_test)) == 1 : 
             if y_test.iloc[0] == 0: tn = cm[0,0]
             else: tp = cm[0,0]
        # else: print("Aviso: Matriz de confusão com formato inesperado para desempacotamento detalhado.")
    except Exception: 
        print("Aviso: Não foi possível desempacotar totalmente a matriz de confusão.")

    # Prepare metrics to return, focusing on Class 1 (Fraude)
    metrics_to_return = {
        "accuracy": report_dict.get("accuracy", 0),
        "f1_class1": report_dict.get("1", {}).get("f1-score", 0),
        "precision_class1": report_dict.get("1", {}).get("precision", 0),
        "recall_class1": report_dict.get("1", {}).get("recall", 0),
        "support_class1": report_dict.get("1", {}).get("support", 0),
        "f1_class0": report_dict.get("0", {}).get("f1-score", 0),
        "precision_class0": report_dict.get("0", {}).get("precision", 0),
        "recall_class0": report_dict.get("0", {}).get("recall", 0),
        "support_class0": report_dict.get("0", {}).get("support", 0),
        "confusion_matrix_class1": { # Specific to class 1 (fraud) perspective
            "tp": int(tp), # True Positives for fraud
            "fn": int(fn), # False Negatives for fraud (missed frauds)
            "fp": int(fp), # False Positives for fraud (non-frauds flagged as fraud)
            "tn": int(tn)  # True Negatives for fraud (non-frauds correctly identified)
        },
        "training_duration_seconds": training_duration
    }
    
    print("\n--- Pipeline de Detecção de Fraudes Concluído (Execução Parametrizada) ---")
    return metrics_to_return

# This block allows detector.py to still be run directly for testing,
# using some default configurations.
if __name__ == "__main__":
    print("Executando detector.py diretamente (usando configurações padrão)...")
    # Example of default parameters for direct run
    default_model_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'verbose': 1, 
        'n_jobs': -1
    }
    results = run_fraud_detection_pipeline_with_params(
        use_real_data=True,       # Mude para False para testar com dados sintéticos
        apply_smote_to_train=True, # Mude para False se não quiser SMOTE
        model_params_override=default_model_params
    )
    if results:
        print("\nResultados da execução direta de detector.py:")
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value if isinstance(value, int) else f'{value:.4f}'}")

