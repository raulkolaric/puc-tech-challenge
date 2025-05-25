# run_experiments.py
# This script provides a Text-based User Interface (TUI) to configure and run
# the fraud detection pipeline, and store/view results.

import json
import os
from datetime import datetime

# We will import the refactored pipeline function from detector.py
# This assumes detector.py is in the same directory and has a callable function
# like `execute_pipeline(data_config, model_config, smote_config)`
# and returns a dictionary of metrics.
# We'll define the exact signature when we refactor detector.py
from detector import run_fraud_detection_pipeline_with_params # Placeholder name for refactored function

RESULTS_DB_FILE = "fraud_detection_results.json"
DEFAULT_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'verbose': 0, # Keep verbose low for TUI runs unless debugging
    'n_jobs': -1
}

def load_results():
    """Loads existing experiment results from the JSON database file."""
    if not os.path.exists(RESULTS_DB_FILE):
        return []
    try:
        with open(RESULTS_DB_FILE, 'r') as f:
            results = json.load(f)
        return results
    except json.JSONDecodeError:
        print(f"Warning: '{RESULTS_DB_FILE}' is corrupted or not valid JSON. Starting with an empty database.")
        return []
    except Exception as e:
        print(f"Error loading results: {e}")
        return []

def save_result(experiment_details):
    """Saves a new experiment result to the JSON database file."""
    results = load_results()
    results.append(experiment_details)
    try:
        with open(RESULTS_DB_FILE, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Result saved to {RESULTS_DB_FILE}")
    except Exception as e:
        print(f"Error saving result: {e}")

def get_data_choice_from_user():
    """Gets the user's choice for data source."""
    while True:
        print("\nEscolha a fonte de dados:")
        print("1. Dados Reais (creditcard.csv)")
        print("2. Dados Sintéticos (gerados)")
        choice = input("Sua escolha (1 ou 2): ")
        if choice == '1':
            return {'use_real_data': True, 'description': "Real CSV Data"}
        elif choice == '2':
            return {'use_real_data': False, 'description': "Synthetic Data"}
        else:
            print("Escolha inválida. Por favor, digite 1 ou 2.")

def get_smote_choice_from_user():
    """Gets the user's choice for applying SMOTE to the training data."""
    while True:
        choice = input("Aplicar SMOTE ao conjunto de treino? (s/n): ").lower()
        if choice == 's':
            return {'apply_smote': True, 'description': "SMOTE Applied to Train Set"}
        elif choice == 'n':
            return {'apply_smote': False, 'description': "SMOTE Not Applied"}
        else:
            print("Escolha inválida. Por favor, digite 's' ou 'n'.")

def get_model_params_from_user():
    """Gets RandomForestClassifier parameters from the user or uses defaults."""
    print("\nConfigurar parâmetros do RandomForestClassifier:")
    params = DEFAULT_MODEL_PARAMS.copy()
    
    use_defaults = input(f"Usar parâmetros padrão? ({params}) (s/n): ").lower()
    if use_defaults == 's':
        return params

    print("Por favor, insira os valores ou pressione Enter para usar o padrão.")

    def get_int_input(prompt, default_val):
        while True:
            val_str = input(f"{prompt} (padrão: {default_val}): ")
            if not val_str:
                return default_val
            try:
                return int(val_str)
            except ValueError:
                print("Entrada inválida. Por favor, insira um número inteiro.")
    
    def get_noneable_int_input(prompt, default_val):
        while True:
            val_str = input(f"{prompt} (padrão: {default_val}, 'None' para nenhum): ")
            if not val_str:
                return default_val
            if val_str.lower() == 'none':
                return None
            try:
                return int(val_str)
            except ValueError:
                print("Entrada inválida. Por favor, insira um número inteiro ou 'None'.")

    params['n_estimators'] = get_int_input("Número de árvores (n_estimators)", params['n_estimators'])
    params['max_depth'] = get_noneable_int_input("Profundidade máxima da árvore (max_depth)", params['max_depth'])
    params['min_samples_split'] = get_int_input("Mínimo de amostras para dividir um nó (min_samples_split)", params['min_samples_split'])
    params['min_samples_leaf'] = get_int_input("Mínimo de amostras em um nó folha (min_samples_leaf)", params['min_samples_leaf'])
    
    print(f"Parâmetros configurados: {params}")
    return params

def display_past_results(results_list):
    """Displays past experiment results."""
    if not results_list:
        print("\nNenhum resultado anterior encontrado.")
        return

    print("\n--- Resultados Anteriores ---")
    for i, res in enumerate(results_list):
        print(f"\nExperimento {i+1} ({res.get('timestamp', 'N/A')})")
        print(f"  Dados: {res.get('data_config', {}).get('description', 'N/A')}")
        print(f"  SMOTE: {res.get('smote_config', {}).get('description', 'N/A')}")
        print(f"  Parâmetros do Modelo: {res.get('model_params', 'N/A')}")
        metrics = res.get('metrics', {})
        print(f"  Métricas (Classe 1 - Fraude):")
        print(f"    F1-Score: {metrics.get('f1_class1', 'N/A'):.4f}")
        print(f"    Precisão: {metrics.get('precision_class1', 'N/A'):.4f}")
        print(f"    Recall:   {metrics.get('recall_class1', 'N/A'):.4f}")
        cm = metrics.get('confusion_matrix_class1', {})
        print(f"    Matriz de Confusão (Fraude): TP={cm.get('tp', 'N/A')}, FP={cm.get('fp', 'N/A')}, FN={cm.get('fn', 'N/A')}, TN={cm.get('tn', 'N/A')}")
        print("-" * 20)

def run_new_experiment():
    """Guides user through setting up and running a new experiment."""
    print("\n--- Configurando Novo Experimento ---")
    data_config = get_data_choice_from_user()
    smote_config = get_smote_choice_from_user()
    model_params_config = get_model_params_from_user()

    print("\nIniciando pipeline com as configurações escolhidas...")
    # This will call the refactored function from detector.py
    # It needs to accept these configurations and return metrics.
    try:
        # We pass the specific parameters the refactored pipeline will expect
        metrics = run_fraud_detection_pipeline_with_params(
            use_real_data=data_config['use_real_data'],
            apply_smote_to_train=smote_config['apply_smote'],
            model_params_override=model_params_config,
            run_grid_search=False # For now, TUI controls specific params, not GridSearchCV
        )
        
        if metrics:
            print("\n--- Resultados do Experimento Atual ---")
            print(f"  F1-Score (Fraude): {metrics.get('f1_class1', 'N/A'):.4f}")
            print(f"  Precisão (Fraude): {metrics.get('precision_class1', 'N/A'):.4f}")
            print(f"  Recall (Fraude):   {metrics.get('recall_class1', 'N/A'):.4f}")
            cm = metrics.get('confusion_matrix_class1', {})
            print(f"  Matriz de Confusão (Fraude): TP={cm.get('tp')}, FP={cm.get('fp')}, FN={cm.get('fn')}, TN={cm.get('tn')}")


            experiment_log = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_config": data_config,
                "smote_config": smote_config,
                "model_params": model_params_config,
                "metrics": metrics # Store the whole metrics dict returned by the pipeline
            }
            save_result(experiment_log)
        else:
            print("Pipeline executado, mas não retornou métricas.")

    except Exception as e:
        print(f"\nOcorreu um erro ao executar o pipeline: {e}")
        print("Verifique se 'detector.py' e suas dependências estão corretas.")


def main_menu():
    """Displays the main menu and handles user choices."""
    while True:
        print("\n--- Menu Principal do Detector de Fraudes ---")
        print("1. Executar Novo Experimento")
        print("2. Visualizar Resultados Anteriores")
        print("3. Sair")
        choice = input("Sua escolha: ")

        if choice == '1':
            run_new_experiment()
        elif choice == '2':
            results = load_results()
            display_past_results(results)
        elif choice == '3':
            print("Saindo...")
            break
        else:
            print("Escolha inválida. Tente novamente.")

if __name__ == "__main__":
    # Ensure detector.py can be imported. This might mean ensuring the project root
    # is in PYTHONPATH or that run_experiments.py is run from the project root.
    # If detector.py is in the same directory, direct import should work.
    # If detector.py is in a sub-directory, sys.path manipulation might be needed
    # or better, structure your project as a package.
    # For now, assuming detector.py is importable as is.
    main_menu()
