# detector.py
# Este é o script principal para o pipeline de detecção de fraudes.

import os # Para manipulação de caminhos de arquivo de forma robusta
import pandas as pd # Pode ser útil para inspecionar X_final, y_final

# Importar funções do nosso módulo data_input que está na pasta src
from src.data_input import (load_csv_data,
                            generate_synthetic_data_scratch,
                            engineer_features_from_data,
                            augment_data_smote) # Certifique-se que imblearn está instalado se for usar

# Importar bibliotecas do scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def run_fraud_detection_pipeline():
    """
    Executa o pipeline completo de detecção de fraudes.
    """
    print("--- Iniciando Pipeline de Detecção de Fraudes em detector.py ---")

    # --- Etapa 1: Carregar e Pré-processar Dados Iniciais ---
    # Usaremos dados reais para este exemplo.
    
    # Construindo o caminho para o arquivo de dados de forma robusta
    # __file__ é o caminho para o script atual (detector.py)
    # os.path.dirname(__file__) nos dá o diretório onde detector.py está
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path_real_data = os.path.join(script_dir, 'data', 'creditcard.csv')
    target_col = 'Class' # Nome da coluna alvo no creditcard.csv

    print(f"\nTentando carregar dados reais de: {file_path_real_data}")
    X_initial, y_initial = load_csv_data(file_path=file_path_real_data, target_column_name=target_col)

    if X_initial is None or y_initial is None:
        print("--- Falha ao carregar dados iniciais. Encerrando o pipeline. ---")
        return

    print("\nDados iniciais carregados com sucesso.")
    print(f"Shape de X_initial: {X_initial.shape}")
    print("Primeiras 3 linhas de X_initial:\n", X_initial.head(3))

    # Engenharia de Features
    print("\nAplicando engenharia de features...")
    X_processed = engineer_features_from_data(X_initial)

    if X_processed is None:
        print("--- Falha na engenharia de features. Encerrando o pipeline. ---")
        return
        
    print("\nEngenharia de features aplicada com sucesso.")
    print(f"Shape de X_processed após engenharia: {X_processed.shape}")
    print("Primeiras 3 linhas de X_processed:\n", X_processed.head(3))

    # Atribuindo X_final e y_final para o próximo passo
    # Aqui você decidiria se aplica SMOTE ou não. Por enquanto, vamos usar X_processed.
    X_final = X_processed
    y_final = y_initial # y não muda com a engenharia de features de X

    # Opcional: Aplicar SMOTE para lidar com desbalanceamento
    # Lembre-se: idealmente, SMOTE é aplicado APENAS no conjunto de treino APÓS o split.
    # Para uma primeira execução e simplificação, você pode testá-lo aqui ou pular.
    aplicar_smote_agora = False # Mude para True para testar SMOTE nesta etapa
    if aplicar_smote_agora:
        if augment_data_smote.IMBLEARN_AVAILABLE: # Verifica se a função importou SMOTE com sucesso
            print("\n(Opcional) Aplicando SMOTE aos dados ANTES do split...")
            X_final, y_final = augment_data_smote(X_final, y_final)
            print(f"Shape de X_final após SMOTE: {X_final.shape}")
            print(f"Distribuição de y_final após SMOTE:\n{y_final.value_counts(normalize=True)}")
        else:
            print("\nSMOTE não aplicado: biblioteca imbalanced-learn não disponível (conforme detectado em data_input.py).")


    # --- Etapa 2: Divisão dos Dados em Treino e Teste ---
    print("\nDividindo os dados em conjuntos de treino e teste...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final,
            test_size=0.25,
            random_state=42,
            stratify=y_final # Essencial para dados desbalanceados
        )
        print("Dados divididos com sucesso.")
        print(f"Shape de X_train: {X_train.shape}, Shape de y_train: {y_train.shape}")
        print(f"Shape de X_test: {X_test.shape}, Shape de y_test: {y_test.shape}")
        print(f"Distribuição do alvo no treino:\n{y_train.value_counts(normalize=True)}")
        print(f"Distribuição do alvo no teste:\n{y_test.value_counts(normalize=True)}")
    except ValueError as e:
        print(f"Erro ao dividir os dados: {e}")
        print("Isso pode acontecer se uma classe tiver pouquíssimas amostras. Verifique y_final.")
        print("--- Encerrando o pipeline. ---")
        return
    except Exception as e_split:
        print(f"Um erro inesperado ocorreu durante a divisão dos dados: {e_split}")
        print("--- Encerrando o pipeline. ---")
        return

    # --- Etapa 3: Treinamento do Modelo ---
    print("\nTreinando o modelo RandomForestClassifier...")
    # Se SMOTE não foi aplicado antes, class_weight='balanced' é uma boa opção.
    # Se SMOTE FOI aplicado aos dados completos (X_final, y_final) antes do split,
    # os dados já podem estar balanceados, e 'balanced' pode não ser necessário ou até prejudicial.
    # A melhor abordagem é aplicar SMOTE somente em X_train, y_train.
    
    # Vamos assumir que, se SMOTE foi aplicado, os dados já estão balanceados.
    # Se não aplicou SMOTE, `class_weight='balanced'` é uma boa escolha.
    model_params = {'random_state': 42, 'n_estimators': 100}
    if not aplicar_smote_agora: # Se SMOTE não foi aplicado globalmente
        model_params['class_weight'] = 'balanced'
        print("Usando class_weight='balanced' pois SMOTE não foi aplicado globalmente.")
    
    model = RandomForestClassifier(**model_params)
    
    try:
        model.fit(X_train, y_train)
        print("Modelo treinado com sucesso.")
    except Exception as e:
        print(f"Erro durante o treinamento do modelo: {e}")
        print("--- Encerrando o pipeline. ---")
        return

    # --- Etapa 4: Avaliação do Modelo ---
    print("\nAvaliando o modelo no conjunto de teste...")
    try:
        predictions = model.predict(X_test)
        # proba_predictions = model.predict_proba(X_test)[:, 1] # Para métricas como AUC-ROC
    except Exception as e:
        print(f"Erro durante a predição: {e}")
        print("--- Encerrando o pipeline. ---")
        return

    print("\n--- Relatório de Classificação ---")
    try:
        report = classification_report(y_test, predictions, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Erro ao gerar relatório de classificação: {e}")

    print("\n--- Matriz de Confusão ---")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    try:
        # Lógica mais robusta para desempacotar cm, especialmente se uma classe não for prevista
        if cm.size == 4: # Matriz 2x2 típica
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1 and y_test.nunique() == 1: # Apenas uma classe presente e prevista
             if y_test.iloc[0] == 0: tn = cm[0,0]; fp,fn,tp = 0,0,0
             else: tp = cm[0,0]; tn,fp,fn = 0,0,0
        else: # Caso inesperado, define como 0 para evitar crash no print
             tn, fp, fn, tp = (0,0,0,0) 
             print("Aviso: Matriz de confusão com formato inesperado. Detalhes abaixo podem não ser precisos.")

        print(f"Verdadeiros Negativos (Não-Fraudes OK): {tn}")
        print(f"Falsos Positivos (Não-Fraudes -> Fraude): {fp}")
        print(f"Falsos Negativos (Fraudes -> Não Fraude): {fn}")
        print(f"Verdadeiros Positivos (Fraudes OK): {tp}")
    except Exception as e_cm: 
        print(f"Não foi possível desempacotar a matriz de confusão: {e_cm}")

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAcurácia Geral: {accuracy:.4f}")

    # --- Etapa 5: Relatórios e Visualizações (Mais avançado) ---
    print("\n(Para relatórios mais avançados e visualizações, adicione código aqui)")

    print("\n--- Pipeline de Detecção de Fraudes Concluído ---")

# Este bloco garante que run_fraud_detection_pipeline() só é chamado 
# quando o script detector.py é executado diretamente.
if __name__ == "__main__":
    run_fraud_detection_pipeline()