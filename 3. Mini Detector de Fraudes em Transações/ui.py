# ui.py
# Text UI to configure and run the fraud detection pipeline, and store/view results.

import json
import os
from datetime import datetime
# Assuming detector.py is in the same directory as ui.py
# and contains the refactored run_fraud_detection_pipeline_with_params function.
try:
    from detector import run_fraud_detection_pipeline_with_params
except ImportError:
    print("ERRO: Não foi possível importar 'run_fraud_detection_pipeline_with_params' de 'detector.py'.")
    print("Certifique-se de que 'detector.py' está no mesmo diretório que 'ui.py' e define esta função.")
    print("O programa será encerrado.")
    exit()


RESULTS_DB_FILE = "fraud_detection_results.json" # Salvo no mesmo diretório que ui.py
DEFAULT_MODEL_PARAMS = {
    'n_estimators': 150,
    'max_depth': 30,
    'min_samples_leaf': 1,    # Explicitly set to default, which is valid
    'min_samples_split': 5,   # Explicitly set to default, which is valid
    'random_state': 42,
    'verbose': 5,
    'n_jobs': -1
}

def load_results():
    """Carrega resultados de experimentos existentes do arquivo JSON."""
    if not os.path.exists(RESULTS_DB_FILE):
        return []
    try:
        with open(RESULTS_DB_FILE, 'r', encoding='utf-8') as f: # Added encoding
            results = json.load(f)
        return results
    except json.JSONDecodeError:
        print(f"Aviso: '{RESULTS_DB_FILE}' está corrompido ou não é um JSON válido. Começando com um banco de dados vazio.")
        return []
    except Exception as e:
        print(f"Erro ao carregar resultados: {e}")
        return []

def save_result(experiment_details):
    """Salva um novo resultado de experimento no arquivo JSON."""
    results = load_results()
    results.append(experiment_details)
    try:
        with open(RESULTS_DB_FILE, 'w', encoding='utf-8') as f: # Added encoding
            json.dump(results, f, indent=4, ensure_ascii=False) # ensure_ascii=False for Portuguese
        print(f"Resultado salvo em {RESULTS_DB_FILE}")
    except Exception as e:
        print(f"Erro ao salvar resultado: {e}")

def get_data_choice_from_user():
    """Obtém a escolha do usuário para a fonte de dados."""
    while True:
        print("\nEscolha a fonte de dados:")
        print("1. Dados Reais (creditcard.csv)")
        print("2. Dados Sintéticos (gerados)")
        choice = input("Sua escolha (1 ou 2): ")
        if choice == '1':
            return {'use_real_data': True, 'description': "Dados Reais do CSV"}
        elif choice == '2':
            return {'use_real_data': False, 'description': "Dados Sintéticos"}
        else:
            print("Escolha inválida. Por favor, digite 1 ou 2.")

def get_smote_choice_from_user():
    """Obtém a escolha do usuário para aplicar SMOTE ao conjunto de treino."""
    while True:
        choice = input("Aplicar SMOTE ao conjunto de treino? (s/n): ").lower()
        if choice == 's':
            return {'apply_smote': True, 'description': "SMOTE Aplicado ao Treino"}
        elif choice == 'n':
            return {'apply_smote': False, 'description': "SMOTE Não Aplicado"}
        else:
            print("Escolha inválida. Por favor, digite 's' ou 'n'.")

def get_threshold_from_user():
    """Obtém o limiar de classificação do usuário."""
    while True:
        val_str = input("Definir limiar de classificação (ex: 0.5, deixe em branco para 0.5): ")
        if not val_str: # User pressed Enter for default
            return 0.5 
        try:
            threshold = float(val_str)
            if 0.0 < threshold < 1.0: # Threshold should be between 0 and 1 exclusive
                return threshold
            else:
                print("Limiar inválido. Por favor, insira um número entre 0.0 e 1.0 (ex: 0.3 para 30%).")
        except ValueError:
            print("Entrada inválida. Por favor, insira um número decimal (ex: 0.4).")


def get_model_params_from_user():
    """Obtém os parâmetros do RandomForestClassifier do usuário ou usa os padrões."""
    print("\nConfigurar parâmetros do RandomForestClassifier:")
    params = DEFAULT_MODEL_PARAMS.copy() # Start with defaults
    
    print("Parâmetros atuais/padrão:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    use_defaults_choice = input(f"Usar estes parâmetros padrão? (s/n): ").lower()
    if use_defaults_choice == 's':
        print("Usando parâmetros padrão.")
        return params

    print("\nPor favor, insira os novos valores ou pressione Enter para manter o padrão atual para cada parâmetro.")

    def get_int_input(prompt_text, current_default):
        while True:
            val_str = input(f"{prompt_text} (atual/padrão: {current_default}): ")
            if not val_str: return current_default # User pressed Enter, keep default
            try: return int(val_str)
            except ValueError: print("Entrada inválida. Deve ser um número inteiro.")
    
    def get_noneable_int_input(prompt_text, current_default):
        while True:
            val_str = input(f"{prompt_text} (atual/padrão: {current_default}, digite 'None' para nenhum): ")
            if not val_str: return current_default # User pressed Enter, keep default
            if val_str.lower() == 'none': return None
            try: return int(val_str)
            except ValueError: print("Entrada inválida. Deve ser um inteiro ou 'None'.")

    params['n_estimators'] = get_int_input("Número de árvores (n_estimators)", params['n_estimators'])
    params['max_depth'] = get_noneable_int_input("Profundidade máxima (max_depth)", params['max_depth'])
    params['min_samples_split'] = get_int_input("Mínimo de amostras para dividir (min_samples_split)", params['min_samples_split'])
    params['min_samples_leaf'] = get_int_input("Mínimo de amostras por folha (min_samples_leaf)", params['min_samples_leaf'])
    # 'random_state', 'verbose', 'n_jobs' will keep their defaults unless you add prompts for them
    
    print(f"\nParâmetros configurados para este experimento: {params}")
    return params

def display_past_results(results_list):
    """Exibe resultados de experimentos anteriores."""
    if not results_list:
        print("\nNenhum resultado anterior encontrado.")
        return

    print("\n--- Resultados dos Experimentos Anteriores ---")
    # Display newest results first
    for i, res in enumerate(reversed(results_list)): 
        print(f"\nExperimento {len(results_list) - i} (Realizado em: {res.get('timestamp', 'N/A')})")
        print(f"  Configuração dos Dados: {res.get('data_config', {}).get('description', 'N/A')}")
        print(f"  SMOTE no Treino: {res.get('smote_config', {}).get('description', 'N/A')}")
        print(f"  Parâmetros do Modelo: {res.get('model_params', 'N/A')}")
        print(f"  Limiar de Classificação Usado: {res.get('classification_threshold', 'N/A')}")
        
        metrics = res.get('metrics', {})
        print(f"  Métricas (Classe 1 - Fraude):")
        f1_c1 = metrics.get('f1_class1', 'N/A')
        prec_c1 = metrics.get('precision_class1', 'N/A')
        rec_c1 = metrics.get('recall_class1', 'N/A')
        
        # Helper to format metrics, handling 'N/A' strings
        def format_metric(value):
            return value if isinstance(value, str) else f'{value:.4f}'

        print(f"    F1-Score: {format_metric(f1_c1)}")
        print(f"    Precisão: {format_metric(prec_c1)}")
        print(f"    Recall:   {format_metric(rec_c1)}")
        
        cm = metrics.get('confusion_matrix_class1', {})
        print(f"    Matriz de Confusão (Fraude): TP={cm.get('tp', 'N/A')}, FP={cm.get('fp', 'N/A')}, FN={cm.get('fn', 'N/A')}, TN={cm.get('tn', 'N/A')}")
        print("-" * 40)

def run_new_experiment():
    """Guia o usuário na configuração e execução de um novo experimento."""
    print("\n--- Configurando Novo Experimento ---")
    data_config = get_data_choice_from_user()
    smote_config = get_smote_choice_from_user()
    model_params_config = get_model_params_from_user()
    classification_threshold_config = get_threshold_from_user()

    print("\nIniciando pipeline com as configurações escolhidas...")
    print(f"  Usando dados: {data_config['description']}")
    print(f"  Aplicar SMOTE no treino: {'Sim' if smote_config['apply_smote'] else 'Não'}")
    print(f"  Parâmetros do modelo: {model_params_config}")
    print(f"  Limiar de classificação: {classification_threshold_config}")
    
    try:
        metrics = run_fraud_detection_pipeline_with_params(
            use_real_data=data_config['use_real_data'],
            apply_smote_to_train=smote_config['apply_smote'],
            model_params_override=model_params_config,
            classification_threshold=classification_threshold_config
        )
        
        if metrics:
            print("\n--- Resultados do Experimento Atual ---")
            f1_c1 = metrics.get('f1_class1', 'N/A')
            prec_c1 = metrics.get('precision_class1', 'N/A')
            rec_c1 = metrics.get('recall_class1', 'N/A')
            def format_metric(value): # Helper for consistent display
                 return value if isinstance(value, str) else f'{value:.4f}'

            print(f"  F1-Score (Fraude): {format_metric(f1_c1)}")
            print(f"  Precisão (Fraude): {format_metric(prec_c1)}")
            print(f"  Recall (Fraude):   {format_metric(rec_c1)}")
            cm = metrics.get('confusion_matrix_class1', {})
            print(f"  Matriz de Confusão (Fraude): TP={cm.get('tp')}, FP={cm.get('fp')}, FN={cm.get('fn')}, TN={cm.get('tn')}")
            print(f"  Acurácia Geral: {format_metric(metrics.get('accuracy', 'N/A'))}")
            print(f"  Duração do Treinamento: {metrics.get('training_duration_seconds', 'N/A'):.2f}s")
            print(f"  Limiar Utilizado: {metrics.get('classification_threshold', 'N/A')}")


            experiment_log = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_config": data_config,
                "smote_config": smote_config,
                "model_params": model_params_config,
                "classification_threshold": classification_threshold_config,
                "metrics": metrics 
            }
            save_result(experiment_log)
        else:
            print("Pipeline executado, mas não retornou métricas válidas.")

    except Exception as e:
        print(f"\nOcorreu um erro ao executar o pipeline: {e}")
        print("Verifique se 'detector.py' e suas dependências estão corretas e se o arquivo de dados está acessível.")


def main_menu():
    """Exibe o menu principal e lida com as escolhas do usuário."""
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
            print("Saindo do programa...")
            break
        else:
            print("Escolha inválida. Tente novamente.")

if __name__ == "__main__":
    main_menu()