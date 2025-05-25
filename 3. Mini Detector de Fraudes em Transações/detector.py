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

# DEBUG ! 

# ... (todo o código das suas funções load_csv_data, generate_synthetic_data_scratch, 
#      engineer_features_from_data, augment_data_smote DEVE ESTAR ACIMA DESTA LINHA) ...

# --- Bloco para Testar este Módulo Diretamente (Opcional) ---
if __name__ == '__main__':
    print("\n--- Executando data_input.py diretamente para fins de teste ---")
    print("="*60)

    # --- Teste 1: Gerar dados sintéticos do zero ---
    print("\n--- Teste 1: generate_synthetic_data_scratch ---")
    X_sintetico, y_sintetico = generate_synthetic_data_scratch(
        n_samples=100,          # Menor número de amostras para teste rápido
        n_features=5,           # Menor número de features
        class_weights=[0.8, 0.2], # Mais balanceado para facilitar a visualização do SMOTE depois
        target_column_name='Alvo'
    )
    if X_sintetico is not None and y_sintetico is not None:
        print("\nDados sintéticos gerados com sucesso:")
        print("Primeiras 3 linhas de X_sintetico:\n", X_sintetico.head(3))
        print("\nPrimeiras 3 linhas de y_sintetico:\n", y_sintetico.head(3))
        print(f"\nDistribuição de y_sintetico:\n{y_sintetico.value_counts()}")
    else:
        print("Falha ao gerar dados sintéticos.")
    print("="*30)

    # --- Teste 2: Carregar dados de um CSV ---
    print("\n--- Teste 2: load_csv_data ---")
    # Ajuste este caminho se o seu 'data' folder estiver em um local diferente
    # em relação a 'src/'. Este caminho '../data/' sobe um nível a partir de 'src/'
    # e depois entra em 'data/'.
    caminho_arquivo_csv_real = '../data/creditcard.csv' 
    nome_coluna_alvo_real = 'Class'                    

    X_real, y_real = load_csv_data(file_path=caminho_arquivo_csv_real, target_column_name=nome_coluna_alvo_real)
    
    if X_real is not None and y_real is not None:
        print("\nDados reais carregados com sucesso:")
        print("Primeiras 3 linhas de X_real:\n", X_real.head(3))
        # ... (restante do bloco de teste como fornecido anteriormente, incluindo Teste 3 e Teste 4) ...
        # --- Teste 3: Engenharia de features (usando os dados carregados) ---
        print("\n--- Teste 3: engineer_features_from_data ---")
        X_engenheirado = engineer_features_from_data(X_real)
        if X_engenheirado is not None:
            print("\nEngenharia de features concluída:")
            print("Primeiras 3 linhas de X_engenheirado:\n", X_engenheirado.head(3))
            if len(X_engenheirado.columns) > len(X_real.columns):
                print("Novas features foram adicionadas com sucesso.")
            else:
                print("Nenhuma nova feature parece ter sido adicionada.")

        # --- Teste 4: Aumento de dados com SMOTE ---
        X_para_smote = X_engenheirado if X_engenheirado is not None else X_real
        if IMBLEARN_AVAILABLE and X_para_smote is not None and y_real is not None:
            print("\n--- Teste 4: augment_data_smote ---")
            X_aumentado, y_aumentado = augment_data_smote(X_para_smote, y_real)
            if X_aumentado is not None and y_aumentado is not None:
                if X_aumentado.shape[0] > X_para_smote.shape[0]:
                    print("\nSMOTE aplicado. Nova distribuição de y_aumentado:")
                    print(y_aumentado.value_counts())
                else:
                    print("\nSMOTE não alterou significativamente os dados.")
            else:
                print("SMOTE não retornou dados válidos.")
        elif not IMBLEARN_AVAILABLE:
            print("\nSMOTE não testado: biblioteca imbalanced-learn não está disponível.")
        else:
            print("\nSMOTE não testado: dados de entrada (X ou y) não estavam prontos.")
    else:
        print("\nFalha ao carregar dados reais do CSV, pulando testes dependentes.")
    print("="*30)

    print("\n--- Testes em data_input.py concluídos ---")
    print("="*60)