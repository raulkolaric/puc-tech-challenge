import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Carregamento dos Dados ---
url = "https://raw.githubusercontent.com/puc-tech/challenge/refs/heads/main/student_preferences_extended.csv"
df = pd.read_csv(url)

# --- 2. Inspeção Inicial dos Dados ---
print("--- Inspeção Inicial dos Dados ---")
print("\nPrimeiras 5 linhas do DataFrame:")
print(df.head())

print("\nInformações gerais do DataFrame:")
df.info()

print("\nEstatísticas descritivas (incluindo colunas categóricas):")
print(df.describe(include='all'))

print("\nContagem de valores ausentes por coluna:")
print(df.isnull().sum())

# --- 3. Limpeza e Preparação dos Dados ---
print("\n--- Limpeza e Preparação dos Dados ---")

# Remover colunas que geralmente não são usadas diretamente em análises quantitativas de preferência agregada
# ou são metadados.
# 'tempo_resposta' é um timestamp de submissão.
# 'comentario' é texto livre, requer NLP para análise quantitativa.
# 'id_aluno' é um identificador.
cols_to_drop_initial = ['tempo_resposta', 'comentario']
df_cleaned = df.drop(columns=[col for col in cols_to_drop_initial if col in df.columns])

# Para as colunas chave das análises de preferência solicitadas,
# vamos remover linhas onde esses dados estão ausentes para garantir a precisão dessas análises específicas.
key_preference_cols = ['linguagem_preferida', 'horario_estudo', 'formato_conteudo_principal']
df_cleaned.dropna(subset=key_preference_cols, inplace=True)
print(f"\nShape do DataFrame após remover NaNs das colunas chave de preferência: {df_cleaned.shape}")

# Para outras colunas numéricas que podem ser usadas em gráficos,
# podemos preencher NaNs com a média ou mediana.
# Exemplo: 'horas_estudo_dia', 'media_geral', 'satisfacao_curso', 'faltas_percentual'
numeric_cols_to_fill = ['horas_estudo_dia', 'media_geral', 'satisfacao_curso', 'faltas_percentual', 'idade', 'semestre']
for col in numeric_cols_to_fill:
    if col in df_cleaned.columns:
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True) # Usando mediana por ser menos sensível a outliers

# Para outras colunas categóricas, podemos preencher com 'Não Informado' ou a moda.
categorical_cols_to_fill = ['framework_preferido', 'formato_conteudo_secundario',
                              'ambiente_desenvolvimento', 'sistema_operacional', 'area_interesse']
for col in categorical_cols_to_fill:
    if col in df_cleaned.columns:
        df_cleaned[col].fillna('Não Informado', inplace=True)

# Booleanos: preencher com False (ou a moda) se fizer sentido contextual.
boolean_cols = ['estuda_em_grupo', 'usa_biblioteca', 'participa_monitoria', 'busca_estagio', 'prefere_backend', 'interesse_pesquisa']
for col in boolean_cols:
    if col in df_cleaned.columns:
        # Verificando se há NaNs e se a coluna é do tipo object antes de tentar preencher.
        # A conversão para bool pode ser feita depois se necessário e se os valores forem consistentes.
        if df_cleaned[col].isnull().any():
             df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True) # Preenche com a moda
        # Tentativa de converter para tipo booleano de forma segura
        try:
            # Mapeamento explícito se os valores não forem True/False literais
            if not pd.api.types.is_bool_dtype(df_cleaned[col]):
                # Supondo que a coluna pode ter strings 'True', 'False' ou outros valores
                # Este é um exemplo, pode precisar de ajuste baseado nos dados reais
                if df_cleaned[col].dtype == 'object':
                    map_dict = {'True': True, 'False': False, True: True, False: False}
                    # Aplicar o mapa apenas para valores que existem no mapa
                    df_cleaned[col] = df_cleaned[col].map(lambda x: map_dict.get(x, x if pd.isna(x) else df_cleaned[col].mode()[0])).astype(bool)
                else: # Se for numérico 0/1
                     df_cleaned[col] = df_cleaned[col].astype(bool)

        except Exception as e:
            print(f"Não foi possível converter a coluna {col} para booleano diretamente: {e}")


print("\nContagem de valores ausentes após tratamento geral:")
print(df_cleaned.isnull().sum())

print("\nPrimeiras 5 linhas do DataFrame limpo e preparado:")
print(df_cleaned.head())
df_cleaned.info() # Para verificar os tipos de dados após a limpeza


# --- 4. Configurações de Visualização ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.autolayout'] = True # Ajusta automaticamente o layout para evitar cortes


# --- 5. Análise de Preferências Solicitadas ---

# Insight 1: Total de respostas por Linguagem de Programação Preferida
print("\n--- Insight 1: Linguagem de Programação Preferida ---")
linguagem_counts = df_cleaned['linguagem_preferida'].value_counts()
print(linguagem_counts)

plt.figure()
sns.barplot(x=linguagem_counts.index, y=linguagem_counts.values, palette="viridis", hue=linguagem_counts.index, legend=False)
plt.title('Total de Alunos por Linguagem de Programação Preferida', fontsize=16)
plt.xlabel('Linguagem de Programação', fontsize=14)
plt.ylabel('Número de Alunos', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.show()

# Insight 2: Percentual de preferência por Horário de Estudo
print("\n--- Insight 2: Horário de Estudo Preferido ---")
horario_counts = df_cleaned['horario_estudo'].value_counts()
horario_percentages = df_cleaned['horario_estudo'].value_counts(normalize=True) * 100
print(horario_percentages)

plt.figure(figsize=(10,8)) # Tamanho específico para gráfico de pizza
plt.pie(horario_counts, labels=horario_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Percentual de Preferência por Horário de Estudo', fontsize=16)
plt.axis('equal')
plt.show()

# Insight 3: Formato de Conteúdo Principal Mais Popular
print("\n--- Insight 3: Formato de Conteúdo Principal Preferido ---")
formato_principal_counts = df_cleaned['formato_conteudo_principal'].value_counts()
print(formato_principal_counts)
print(f"O formato de conteúdo principal mais popular é: '{formato_principal_counts.idxmax()}' com {formato_principal_counts.max()} preferências.")

plt.figure()
sns.barplot(x=formato_principal_counts.index, y=formato_principal_counts.values, palette="coolwarm", hue=formato_principal_counts.index, legend=False)
plt.title('Preferência por Formato de Conteúdo Principal', fontsize=16)
plt.xlabel('Formato de Conteúdo', fontsize=14)
plt.ylabel('Número de Alunos', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.show()


# --- 6. Gráficos Adicionais Explorando Outras Colunas ---
print("\n--- Gráficos Adicionais ---")

# Gráfico 1: Distribuição da Satisfação com o Curso
if 'satisfacao_curso' in df_cleaned.columns:
    print("\nAnalisando 'satisfacao_curso'...")
    plt.figure()
    # Assegurar que a coluna é numérica para o histograma
    if pd.api.types.is_numeric_dtype(df_cleaned['satisfacao_curso']):
        sns.histplot(df_cleaned['satisfacao_curso'], kde=True, bins=5) # bins para agrupar os níveis de satisfação
        plt.title('Distribuição da Satisfação com o Curso (1 a 5)', fontsize=16)
        plt.xlabel('Nível de Satisfação', fontsize=14)
        plt.ylabel('Número de Alunos', fontsize=14)
        plt.show()
    else:
        print("'satisfacao_curso' não é numérica ou contém valores não convertidos.")
else:
    print("'satisfacao_curso' não encontrada.")

# Gráfico 2: Horas de Estudo por Dia vs. Média Geral
if 'horas_estudo_dia' in df_cleaned.columns and 'media_geral' in df_cleaned.columns:
    print("\nAnalisando 'horas_estudo_dia' vs 'media_geral'...")
    plt.figure()
    # Assegurar que as colunas são numéricas
    if pd.api.types.is_numeric_dtype(df_cleaned['horas_estudo_dia']) and pd.api.types.is_numeric_dtype(df_cleaned['media_geral']):
        sns.scatterplot(x='horas_estudo_dia', y='media_geral', data=df_cleaned, hue='satisfacao_curso', palette='coolwarm', alpha=0.7)
        plt.title('Horas de Estudo por Dia vs. Média Geral', fontsize=16)
        plt.xlabel('Horas de Estudo por Dia', fontsize=14)
        plt.ylabel('Média Geral', fontsize=14)
        plt.legend(title='Satisfação Curso')
        plt.show()
    else:
        print("'horas_estudo_dia' ou 'media_geral' não são numéricas ou contêm valores não convertidos.")
else:
    print("'horas_estudo_dia' ou 'media_geral' não encontradas.")


# Gráfico 3: Contagem de Preferência por Área de Interesse
if 'area_interesse' in df_cleaned.columns:
    print("\nAnalisando 'area_interesse'...")
    area_interesse_counts = df_cleaned['area_interesse'].value_counts().nlargest(10) # Top 10 para clareza
    plt.figure()
    sns.barplot(x=area_interesse_counts.index, y=area_interesse_counts.values, palette="cubehelix", hue=area_interesse_counts.index, legend=False)
    plt.title('Top 10 Áreas de Interesse dos Alunos', fontsize=16)
    plt.xlabel('Área de Interesse', fontsize=14)
    plt.ylabel('Número de Alunos', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.show()
else:
    print("'area_interesse' não encontrada.")

# Gráfico 4: Uso da Biblioteca
if 'usa_biblioteca' in df_cleaned.columns:
    print("\nAnalisando 'usa_biblioteca'...")
    # Certificar que a coluna é tratada como categoria para value_counts
    usa_biblioteca_counts = df_cleaned['usa_biblioteca'].astype(str).value_counts()
    plt.figure(figsize=(8,6))
    plt.pie(usa_biblioteca_counts, labels=usa_biblioteca_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
    plt.title('Alunos que Utilizam a Biblioteca', fontsize=16)
    plt.axis('equal')
    plt.show()
else:
    print("'usa_biblioteca' não encontrada.")


# Gráfico 5: Relação entre Semestre e Média Geral (Boxplot)
if 'semestre' in df_cleaned.columns and 'media_geral' in df_cleaned.columns:
    print("\nAnalisando 'semestre' vs 'media_geral'...")
    plt.figure(figsize=(14, 8))
    # Assegurar que 'semestre' seja tratado categoricamente para o boxplot se não for já
    if not pd.api.types.is_categorical_dtype(df_cleaned['semestre']) and not pd.api.types.is_string_dtype(df_cleaned['semestre']):
         df_cleaned['semestre_cat'] = df_cleaned['semestre'].astype(str) # Converter para string para tratar como categoria
         # Ordenar categorias de semestre se necessário
         semestre_order = sorted(df_cleaned['semestre_cat'].unique(), key=lambda x: int(x) if x.isdigit() else float('inf'))
         sns.boxplot(x='semestre_cat', y='media_geral', data=df_cleaned, palette="pastel", order=semestre_order)

    else: # Se já for categórico ou string
        semestre_order = sorted(df_cleaned['semestre'].unique(), key=lambda x: int(x) if isinstance(x, (str, int)) and str(x).isdigit() else float('inf'))
        sns.boxplot(x='semestre', y='media_geral', data=df_cleaned, palette="pastel", order=semestre_order)


#     plt.title('Distribuição da Média Geral por Semestre', fontsize=16)
#     plt.xlabel('Semestre', fontsize=14)
#     plt.ylabel('Média Geral', fontsize=14)
#     plt.xticks(rotation=45, ha='right')
#     plt.show()
#     if 'semestre_cat' in df_cleaned.columns: # Remover coluna auxiliar
#         df_cleaned.drop('semestre_cat', axis=1, inplace=True)
# else:
#     print("'semestre' ou 'media_geral' não encontradas.")

print("\n--- Fim da Análise ---")