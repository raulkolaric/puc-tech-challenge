# fraud_detection_script.py
# Este é o script principal para o pipeline de detecção de fraudes.

# --- Importações Essenciais ---
# Importar funções do nosso módulo data_input que está na pasta src
# Para que isso funcione, certifique-se de que a pasta 'src' está no mesmo nível
# ou em um nível acessível a partir de onde você executa este script,
# e que 'src' contém um arquivo __init__.py (mesmo que vazio).
from src.data_input import (load_csv_data,
                            generate_synthetic_data_scratch,
                            engineer_features_from_data,
                            augment_data_smote)

# Importar bibliotecas do scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Outras importações que possam ser úteis
import pandas as pd # Embora muitas manipulações de DataFrame estejam em data_input

def run_fraud_detection_pipeline():
    """
    Executa o pipeline completo de detecção de fraudes:
    1. Carrega/Gera dados
    2. Pré-processa (incluindo engenharia de features e opcionalmente SMOTE)
    3. Divide os dados
    4. Treina o modelo
    5. Avalia o modelo
    6. Exibe relatórios básicos
    """
    print("--- Iniciando Pipeline de Detecção de Fraudes ---")

    # --- Objetivo 1: Gerar/Carregar Dados & Objetivo 2: Pré-processamento (parte inicial) ---
    # Escolha UMA das opções de fonte de dados abaixo (descomente a desejada):

    # Opção A: Carregar dados reais de um CSV
    # Certifique-se que o arquivo CSV existe no caminho especificado e
    # que o nome da coluna alvo está correto.
    # --------------------------------------------------------------------
    # file_path_real_data = 'data/creditcard.csv' # Altere para o caminho do seu arquivo
    # target_col = 'Class'                        # Altere para o nome da sua coluna alvo
    # X_initial, y_initial = load_csv_data(file_path=file_path_real_data, target_column_name=target_col)
    
    # if X_initial is not None:
    #     # Engenharia de Features (parte do Pré-processamento)
    #     X_processed = engineer_features_from_data(X_initial)
        
    #     # Opcional: Aumento de Dados com SMOTE (parte do Pré-processamento)
    #     # É geralmente melhor aplicar SMOTE APENAS no conjunto de treino após o split,
    #     # mas para simplificar a demonstração inicial, podemos aplicar aqui.
    #     # Certifique-se que imblearn está instalado: pip install imbalanced-learn
    #     if X_processed is not None and y_initial is not None: # Verifica se X_processed e y_initial são válidos
    #         print("\n(Opção de aplicar SMOTE antes do split - para demonstração)")
    #         # X_final, y_final = augment_data_smote(X_processed, y_initial)
    #         # Se não for usar SMOTE ou se falhou:
    #         X_final = X_processed
    #         y_final = y_initial
    #     else:
    #         X_final, y_final = None, None # Propaga o erro se X_processed for None
    # else:
    #     X_final, y_final = None, None # Propaga o erro se X_initial for None
    # --------------------------------------------------------------------

    # Opção B: Gerar dados sintéticos (para desenvolvimento e teste rápido)
    # --------------------------------------------------------------------
    print("\nUsando Opção B: Gerando dados sintéticos.")
    X_final, y_final = generate_synthetic_data_scratch(
        n_samples=20000,        # Número de amostras
        n_features=20,          # Número de features
        class_weights=[0.99, 0.01] # 1% de "fraudes"
    )
    # A engenharia de features normalmente não se aplica a dados sintéticos genéricos
    # a menos que você crie features sintéticas com nomes específicos que sua função espera.
    # --------------------------------------------------------------------

    # Verificar se os dados foram carregados/gerados com sucesso
    if X_final is None or y_final is None:
        print("\n--- Falha ao obter os dados. Encerrando o pipeline. ---")
        return # Encerra a função

    print(f"\nForma final de X antes do split: {X_final.shape}")
    print(f"Distribuição final de y antes do split:\n{y_final.value_counts(normalize=True)}")

    # --- Pré-processamento (Parte 2: Divisão dos Dados) ---
    print("\nDividindo os dados em conjuntos de treino e teste...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final,
            test_size=0.25,       # 25% dos dados para teste
            random_state=42,      # Para reprodutibilidade
            stratify=y_final      # MUITO IMPORTANTE para dados desbalanceados como os de fraude
                                  # Garante que a proporção das classes seja similar nos sets de treino e teste.
        )
        print(f"Shape de X_train: {X_train.shape}, Shape de y_train: {y_train.shape}")
        print(f"Shape de X_test: {X_test.shape}, Shape de y_test: {y_test.shape}")
        print(f"Distribuição do alvo no treino:\n{y_train.value_counts(normalize=True)}")
        print(f"Distribuição do alvo no teste:\n{y_test.value_counts(normalize=True)}")
    except ValueError as e:
        print(f"Erro ao dividir os dados: {e}")
        print("Isso pode acontecer se uma classe tiver pouquíssimas amostras para permitir a estratificação.")
        print("Verifique a distribuição de y_final e o tamanho de test_size.")
        print("--- Encerrando o pipeline devido a erro no split. ---")
        return


    # --- Objetivo 3: Treinamento do Modelo ---
    print("\nTreinando o modelo RandomForestClassifier...")
    # class_weight='balanced' é uma forma de lidar com desbalanceamento diretamente no modelo.
    # Se você usou SMOTE para balancear X_final, y_final ANTES do split,
    # talvez não precise de class_weight='balanced' ou precise ajustar.
    # A melhor prática é aplicar SMOTE SOMENTE no X_train, y_train APÓS o split.
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
    
    try:
        model.fit(X_train, y_train)
        print("Modelo treinado com sucesso.")
    except Exception as e:
        print(f"Erro durante o treinamento do modelo: {e}")
        print("--- Encerrando o pipeline devido a erro no treinamento. ---")
        return

    # --- Objetivo 4: Avaliação do Modelo ---
    print("\nAvaliando o modelo no conjunto de teste...")
    try:
        predictions = model.predict(X_test)
        proba_predictions = model.predict_proba(X_test)[:, 1] # Probabilidades para a classe positiva (fraude)
    except Exception as e:
        print(f"Erro durante a predição: {e}")
        print("--- Encerrando o pipeline devido a erro na predição. ---")
        return

    print("\n--- Relatório de Classificação ---")
    # Handle potential warnings from classification_report for undefined metrics
    try:
        report = classification_report(y_test, predictions, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Erro ao gerar relatório de classificação: {e}")


    print("\n--- Matriz de Confusão ---")
    # Formato da Matriz de Confusão:
    # [[Verdadeiro Negativo (TN), Falso Positivo (FP)],
    #  [Falso Negativo (FN),  Verdadeiro Positivo (TP)]]
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    try:
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0],0,0,0) # Handle cases where cm might not be 2x2
        print(f"Verdadeiros Negativos (Não-Fraudes corretamente identificadas): {tn}")
        print(f"Falsos Positivos (Não-Fraudes incorretamente marcadas como Fraude) - Erro Tipo I: {fp}")
        print(f"Falsos Negativos (Fraudes não detectadas) - Erro Tipo II: {fn}")
        print(f"Verdadeiros Positivos (Fraudes corretamente identificadas): {tp}")
    except IndexError: # If confusion matrix is not as expected (e.g. only one class predicted)
        print("Não foi possível desempacotar a matriz de confusão. Verifique as predições.")


    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAcurácia Geral: {accuracy:.4f}")

    # --- Objetivo 5: Relatórios e Visualizações (Mais avançado) ---
    # Aqui você poderia adicionar código para:
    # - Salvar o relatório de classificação e a matriz de confusão em um arquivo.
    # - Gerar e salvar plots (ex: curva ROC, feature importances) usando matplotlib/seaborn.
    #   (Essas funções de plotagem poderiam estar em um `visualization_utils.py`)
    print("\n(Para relatórios mais avançados e visualizações, adicione código aqui ou chame funções de um módulo de utils)")

    print("\n--- Pipeline de Detecção de Fraudes Concluído ---")

# Este bloco garante que run_fraud_detection_pipeline() só é chamado quando o script é executado diretamente.
if __name__ == "__main__":
    run_fraud_detection_pipeline()