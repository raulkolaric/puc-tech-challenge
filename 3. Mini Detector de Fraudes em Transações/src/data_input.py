# src/data_input.py
# Este arquivo contém funções para carregamento de dados, geração de dados sintéticos,
# engenharia de features (características) e aumento de dados (data augmentation).

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Para aumento de dados (SMOTE), você precisará da biblioteca imbalanced-learn
# Se não a tiver, instale-a: pip install imbalanced-learn
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
        # print("Primeiras 5 linhas:\n", df.head()) # Debug
        # print("\nInformações dos Dados:") # Debug
        # df.info() # Debug

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
    """
    Gera um conjunto de dados sintético para classificação binária a partir do zero.

    Args:
        n_samples (int): Número total de amostras.
        n_features (int): Número total de features.
        n_informative (int): Número de features informativas.
        class_weights (list): Proporção de amostras para cada classe para desbalanceamento.
        target_column_name (str): Nome para a coluna alvo na Series retornada.
        random_state (int): Semente para geração de números aleatórios (para reprodutibilidade).

    Returns:
        tuple: (pandas.DataFrame, pandas.Series) para features (X) e alvo (y).
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
        flip_y=0.01, # Introduz uma pequena quantidade de ruído nos rótulos
        random_state=random_state
    )
    
    # Converte para DataFrame/Series do pandas para consistência com dados carregados
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
    """
    Cria novas features a partir de um DataFrame de features existente.

    Args:
        X_input_df (pandas.DataFrame): O DataFrame de features de entrada.

    Returns:
        pandas.DataFrame: DataFrame com features originais e as recém-criadas por engenharia,
                          ou None se a entrada for None.
    """
    if X_input_df is None:
        print("DataFrame de entrada para engenharia de features é None. Pulando.")
        return None
        
    print("\nRealizando engenharia de novas features a partir dos dados de entrada...")
    X_engineered = X_input_df.copy() # Trabalhe em uma cópia

    # === Adicione sua lógica específica de engenharia de features aqui ===
    # Exemplo: Criar um termo de interação se as colunas 'Time' e 'Amount' existirem
    # (Estes são comuns em alguns datasets de fraude, adapte para suas colunas reais)
    if 'Time' in X_engineered.columns and 'Amount' in X_engineered.columns:
        # Evitar divisão por zero se 'Amount' puder ser zero
        X_engineered['Amount_per_Time'] = X_engineered['Amount'] / (X_engineered['Time'] + 1e-6) 
        print("Feature criada: 'Amount_per_Time'")

    # Exemplo: Transformação logarítmica de 'Amount' se existir e for positiva
    if 'Amount' in X_engineered.columns:
        if (X_engineered['Amount'] > 0).all(): # Verificar se todos os valores são positivos
            X_engineered['Log_Amount'] = np.log(X_engineered['Amount'])
            print("Feature criada: 'Log_Amount'")
        else:
            # Lidar com casos com valores 0 ou negativos, ex: log1p ou pular
            X_engineered['Log1p_Amount'] = np.log1p(X_engineered['Amount']) # log(1+x) lida com 0
            print("Feature criada: 'Log1p_Amount' (lida com valores zero)")
            
    # Adicione mais de suas regras de engenharia de features...
    # Por exemplo, baseado no `creditcard.csv` do seu projeto que tem features V1, V2...:
    if 'V1' in X_engineered.columns and 'V2' in X_engineered.columns:
        X_engineered['V1_x_V2'] = X_engineered['V1'] * X_engineered['V2']
        print("Feature criada: 'V1_x_V2'")
    # === Fim da lógica específica de engenharia de features ===

    print(f"Engenharia de features completa. Novo shape X: {X_engineered.shape}")
    return X_engineered

# Opção 3B: Aumento de Dados usando SMOTE (para classificação desbalanceada)
def augment_data_smote(X_input_df, y_input_series, random_state=42):
    """
    Aumenta os dados usando SMOTE (Synthetic Minority Over-sampling Technique)
    para lidar com o desbalanceamento de classes. Útil quando a classe minoritária (ex: fraude) está sub-representada.

    Args:
        X_input_df (pandas.DataFrame): DataFrame de features original.
        y_input_series (pandas.Series): Series do alvo original.
        random_state (int): Semente para reprodutibilidade.

    Returns:
        tuple: (pandas.DataFrame, pandas.Series) para features aumentadas (X_smote) 
               e alvo (y_smote), ou os dados originais se o SMOTE falhar ou não estiver disponível.
    """
    if not IMBLEARN_AVAILABLE:
        print("SMOTE requer imbalanced-learn. Retornando dados originais.")
        return X_input_df, y_input_series
    if X_input_df is None or y_input_series is None:
        print("X ou y de entrada para SMOTE é None. Retornando dados originais.")
        return X_input_df, y_input_series

    print("\nTentando aumentar os dados usando SMOTE...")
    try:
        # Verificar distribuição de classes inicial
        print("Distribuição de classes original:\n", y_input_series.value_counts(normalize=True) * 100)
        
        # Garantir que há uma classe minoritária para superamostragem
        if len(y_input_series.value_counts()) < 2 or y_input_series.value_counts().min() < 2 : # SMOTE precisa de pelo menos 2 amostras na classe minoritária para algumas versões/configurações
            print("Não há amostras suficientes na classe minoritária ou apenas uma classe presente. Pulando SMOTE.")
            return X_input_df, y_input_series

        smote = SMOTE(random_state=random_state)
        X_smote, y_smote = smote.fit_resample(X_input_df, y_input_series)
        
        # Converter X_smote de volta para DataFrame se tornou um array NumPy, preservando nomes das colunas
        X_smote_df = pd.DataFrame(X_smote, columns=X_input_df.columns)
        y_smote_series = pd.Series(y_smote, name=y_input_series.name)

        print("SMOTE aplicado com sucesso.")
        print(f"Shape X após SMOTE: {X_smote_df.shape}, shape y após SMOTE: {y_smote_series.shape}")
        print("Nova distribuição de classes após SMOTE:\n", y_smote_series.value_counts(normalize=True) * 100)
        return X_smote_df, y_smote_series
    except Exception as e:
        print(f"Erro durante o SMOTE: {e}. Retornando dados originais.")
        return X_input_df, y_input_series