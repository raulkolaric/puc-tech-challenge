# src/data_input.py
# Este arquivo contém funções para carregamento de dados, geração de dados sintéticos,
# engenharia de features (características) e aumento de dados (data augmentation).

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Para aumento de dados (SMOTE), você precisará da biblioteca imbalanced-learn
# Se não a tiver, instale-a: pip install imbalanced-learn

# caminho_absoluto_csv = r'C:\Users\raulk\Desktop\code\puc-tech\Desafio PUCTECH\3. Mini Detector de Fraudes em Transações\data\creditcard.csv'
# print(f"DEBUG: Tentando carregar com caminho absoluto: {caminho_absoluto_csv}") # Linha de depuração

# nome_coluna_alvo_real = 'Class'

           

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Aviso: A biblioteca imbalanced-learn não foi encontrada. A funcionalidade SMOTE não estará disponível.")
    print("Para instalá-la, execute: pip install imbalanced-learn")

print("data_input.py carregado: Contém funções para manipulação de dados.")

# --- 1. Função para Importar/Receber Dados CSV ---
def load_csv_data(file_path, target_column_name):
    # ... (código completo da função load_csv_data como fornecemos antes, com try-except, etc.) ...
    # ... (vou omitir o corpo da função aqui para economizar espaço, mas ele deve estar completo)
    """
    Carrega dados de um arquivo CSV especificado e separa as features (X) e o alvo (y).
    target_column_name:
    é a coluna que diz se é fraudulenta a transação
    Args:
        file_path (str): O caminho para o arquivo CSV.
        target_column_name (str): O nome da coluna da variável alvo.

    Returns:
        tuple: (pandas.DataFrame, pandas.Series) para features (X) e alvo (y),
               ou (None, None) se o carregamento falhar ou a coluna alvo não for encontrada.
    """
    print(f"\nTentando carregar dados de: {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Dados carregados com sucesso de {file_path}.")
        if target_column_name in df.columns:
            X = df.drop(target_column_name, axis=1)
            y = df[target_column_name]
            print(f"Features (X) e alvo (y: '{target_column_name}') separados.")
            print(f"Shape das Features (X): {X.shape}, Shape do Alvo (y): {y.shape}")
            return X, y
        else:
            print(f"Erro: Coluna alvo '{target_column_name}' não encontrada em {file_path}.")
            print(f"Colunas disponíveis: {df.columns.tolist()}")
            return None, None
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {file_path}. Por favor, verifique se o caminho está correto.")
        return None, None
    except Exception as e:
        print(f"Ocorreu um erro ao carregar ou processar os dados: {e}")
        return None, None

# --- 2. Função para Criar Dados do Zero (Sintéticos) ---
def generate_synthetic_data_scratch(n_samples=10000, n_features=29, n_informative=20,
                                    class_weights=[0.998, 0.002], target_column_name='Class', random_state=42):
    # ... (código completo da função generate_synthetic_data_scratch como fornecemos antes) ...
    """
    Gera um conjunto de dados sintético para classificação binária a partir do zero.
    # ... (Args e Returns)
    """
    print("\nGerando dados sintéticos do zero...")
    X_synth, y_synth = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative,
        n_redundant=max(0, n_features - n_informative - 2), 
        n_repeated=0, 
        n_classes=2,
        n_clusters_per_class=1, 
        weights=class_weights, 
        flip_y=0.01, 
        random_state=random_state
    )
    feature_names = [f'feature_sintetica_{i+1}' for i in range(X_synth.shape[1])]
    X_df = pd.DataFrame(X_synth, columns=feature_names)
    y_series = pd.Series(y_synth, name=target_column_name)
    print(f"Dados sintéticos gerados com shape X: {X_df.shape}, shape y: {y_series.shape}")
    class_dist = y_series.value_counts(normalize=True) * 100
    print(f"Distribuição das classes: \nClasse 0: {class_dist.get(0, 0):.2f}%\nClasse 1: {class_dist.get(1, 0):.2f}%")
    return X_df, y_series

# --- 3. Funções para Criar Dados Baseados em Dados de Entrada ---
# Opção 3A: Engenharia de Features
def engineer_features_from_data(X_input_df):
    # ... (código completo da função engineer_features_from_data como fornecemos antes) ...
    # ... (vou omitir o corpo da função aqui para economizar espaço)
    """
    Cria novas features a partir de um DataFrame de features existente.
    # ... (Args e Returns)
    """
    if X_input_df is None:
        print("DataFrame de entrada para engenharia de features é None. Pulando.")
        return None
    print("\nRealizando engenharia de novas features a partir dos dados de entrada...")
    X_engineered = X_input_df.copy()
    if 'Time' in X_engineered.columns and 'Amount' in X_engineered.columns:
        X_engineered['Amount_per_Time'] = X_engineered['Amount'] / (X_engineered['Time'] + 1e-6) 
        print("Feature criada: 'Amount_per_Time'")
    if 'Amount' in X_engineered.columns:
        if (X_engineered['Amount'] > 0).all():
            X_engineered['Log_Amount'] = np.log(X_engineered['Amount'])
            print("Feature criada: 'Log_Amount'")
        else:
            X_engineered['Log1p_Amount'] = np.log1p(X_engineered['Amount'])
            print("Feature criada: 'Log1p_Amount' (lida com valores zero)")
    if 'V1' in X_engineered.columns and 'V2' in X_engineered.columns:
        X_engineered['V1_x_V2'] = X_engineered['V1'] * X_engineered['V2']
        print("Feature criada: 'V1_x_V2'")
    print(f"Engenharia de features completa. Novo shape X: {X_engineered.shape}")
    return X_engineered

# Opção 3B: Aumento de Dados usando SMOTE (para classificação desbalanceada)
def augment_data_smote(X_input_df, y_input_series, random_state=42):
    # ... (código completo da função augment_data_smote como fornecemos antes) ...
    # ... (vou omitir o corpo da função aqui para economizar espaço)
    """
    Aumenta os dados usando SMOTE (Synthetic Minority Over-sampling Technique)
    # ... (Args e Returns)
    """
    if not IMBLEARN_AVAILABLE:
        print("SMOTE requer imbalanced-learn. Retornando dados originais.")
        return X_input_df, y_input_series
    if X_input_df is None or y_input_series is None:
        print("X ou y de entrada para SMOTE é None. Retornando dados originais.")
        return X_input_df, y_input_series
    print("\nTentando aumentar os dados usando SMOTE...")
    try:
        print("Distribuição de classes original:\n", y_input_series.value_counts(normalize=True) * 100)
        if len(y_input_series.value_counts()) < 2 or y_input_series.value_counts().min() < 2 :
            print("Não há amostras suficientes na classe minoritária ou apenas uma classe presente. Pulando SMOTE.")
            return X_input_df, y_input_series
        smote = SMOTE(random_state=random_state)
        X_smote, y_smote = smote.fit_resample(X_input_df, y_input_series)
        X_smote_df = pd.DataFrame(X_smote, columns=X_input_df.columns)
        y_smote_series = pd.Series(y_smote, name=y_input_series.name)
        print("SMOTE aplicado com sucesso.")
        print(f"Shape X após SMOTE: {X_smote_df.shape}, shape y após SMOTE: {y_smote_series.shape}")
        print("Nova distribuição de classes após SMOTE:\n", y_smote_series.value_counts(normalize=True) * 100)
        return X_smote_df, y_smote_series
    except Exception as e:
        print(f"Erro durante o SMOTE: {e}. Retornando dados originais.")
        return X_input_df, y_input_series

##debug
if __name__ == '__main__':
    print("\n--- Executando data_input.py diretamente para fins de teste ---")
    print("="*60)

    # --- Teste 1: Gerar dados sintéticos do zero ---
    print("\n--- Teste 1: generate_synthetic_data_scratch ---")
    X_sintetico, y_sintetico = generate_synthetic_data_scratch(
        n_samples=100,
        n_features=5,
        n_informative=3, # Corrected from previous error
        class_weights=[0.8, 0.2],
        target_column_name='Alvo',
        random_state=42
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
    
    # ***** USAR O CAMINHO ABSOLUTO QUE FUNCIONOU *****
    caminho_arquivo_csv_real = r'C:\Users\raulk\Desktop\code\puc-tech\Desafio PUCTECH\3. Mini Detector de Fraudes em Transações\data\creditcard.csv'
    nome_coluna_alvo_real = 'Class' # This is correct for creditcard.csv                   

    print(f"Teste 2: Tentando carregar com caminho: {caminho_arquivo_csv_real}")
    X_real, y_real = load_csv_data(file_path=caminho_arquivo_csv_real, target_column_name=nome_coluna_alvo_real)
    
    if X_real is not None and y_real is not None:
        print("\nDados reais carregados com sucesso (Teste 2):")
        print("Primeiras 3 linhas de X_real:\n", X_real.head(3))
        print(f"Shape de X_real: {X_real.shape}")

        # --- Teste 3: Engenharia de features ---
        # (O restante dos Testes 3 e 4 como estavam antes, pois agora X_real deveria carregar)
        print("\n--- Teste 3: engineer_features_from_data ---")
        X_engenheirado = engineer_features_from_data(X_real)
        if X_engenheirado is not None:
            print("\nEngenharia de features concluída:")
            print("Primeiras 3 linhas de X_engenheirado:\n", X_engenheirado.head(3))
            if len(X_engenheirado.columns) > len(X_real.columns):
                print("Novas features foram adicionadas com sucesso.")
            else:
                print("Nenhuma nova feature parece ter sido adicionada (verifique a lógica em engineer_features_from_data e se as colunas base existem).")

        # --- Teste 4: Aumento de dados com SMOTE ---
        X_para_smote = X_engenheirado if X_engenheirado is not None else X_real
        if IMBLEARN_AVAILABLE and X_para_smote is not None and y_real is not None: # y_real é o original aqui
            print("\n--- Teste 4: augment_data_smote ---")
            # Verifique se y_real é desbalanceado o suficiente para o SMOTE ser significativo
            if y_real.value_counts(normalize=True).min() < 0.4: # Exemplo: se a classe minoritária for < 40%
                X_aumentado, y_aumentado = augment_data_smote(X_para_smote, y_real)
                if X_aumentado is not None and y_aumentado is not None:
                    if X_aumentado.shape[0] > X_para_smote.shape[0]:
                        print("\nSMOTE aplicado. Nova distribuição de y_aumentado:")
                        print(y_aumentado.value_counts())
                    else:
                        print("\nSMOTE não alterou significativamente os dados (verifique a distribuição original de y_real e as condições dentro de augment_data_smote).")
                else:
                    print("SMOTE não retornou dados válidos.")
            else:
                print("\nSMOTE pulado: y_real não parece significativamente desbalanceado para este teste ou a classe minoritária tem poucas amostras.")
        elif not IMBLEARN_AVAILABLE:
            print("\nSMOTE não testado: biblioteca imbalanced-learn não está disponível.")
        else:
            print("\nSMOTE não testado: dados de entrada (X ou y) não estavam prontos.")
    else:
        print(f"\nFalha ao carregar dados reais do CSV para Teste 2 usando o caminho: {caminho_arquivo_csv_real}")
    print("="*30)

    print("\n--- Testes em data_input.py concluídos ---")
    print("="*60)