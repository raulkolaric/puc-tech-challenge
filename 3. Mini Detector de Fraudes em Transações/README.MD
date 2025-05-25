Mini Detector de Fraudes em Transações (Credit Card Fraud Detection)
Este projeto é uma submissão para um desafio universitário, focado na detecção de fraudes em transações de cartão de crédito utilizando técnicas de Machine Learning. Ele inclui um pipeline para processamento de dados, treinamento de modelo, avaliação e um sistema de gerenciamento de experimentos via interface de texto (TUI).

Descrição do Projeto
O objetivo principal é construir um classificador capaz de distinguir transações fraudulentas de transações legítimas. O projeto explora:

Carregamento e geração de dados (reais e sintéticos).
Engenharia de features para melhorar a performance do modelo.
Técnicas para lidar com datasets desbalanceados (SMOTE).
Treinamento e avaliação de um modelo RandomForestClassifier.
Uma interface de texto para configurar e executar experimentos, registrando os resultados.
Estrutura de Pastas
├── data/ # Pasta para armazenar os datasets
│ └── creditcard.csv # Dataset principal (exemplo)
├── notebooks/ # (Opcional) Para Jupyter Notebooks de análise/apresentação
├── src/ # Módulos Python auxiliares
│ ├── init.py # Torna 'src' um pacote Python
│ └── data_input.py # Funções para carregamento, geração e pré-processamento de dados
├── venv_fraude/ # Ambiente virtual Python (ignorado pelo Git)
├── detector.py # Script principal do pipeline de Machine Learning
├── ui.py # Script para a interface de texto de gerenciamento de experimentos
├── fraud_detection_results.json # Banco de dados JSON para armazenar resultados dos experimentos
└── README.md # Este arquivo
Configuração (Setup)
Pré-requisitos
Python 3.7+ (recomendado 3.8+)
pip (gerenciador de pacotes Python)
Ambiente Virtual
Navegue até a pasta raiz deste projeto:
cd "3. Mini Detector de Fraudes em Transações"
Crie o ambiente virtual:

bash
Copy
Edit
python -m venv venv_fraude
Ative o ambiente virtual:

Windows (CMD/PowerShell)

bash
Copy
Edit
.\venv_fraude\Scripts\activate
macOS/Linux

bash
Copy
Edit
source venv_fraude/bin/activate
Você deverá ver (venv_fraude) no início do seu prompt do terminal.

Instalando Dependências
Se ainda não existir, gere o arquivo requirements.txt após instalar as bibliotecas necessárias:

bash
Copy
Edit
pip freeze > requirements.txt
Para instalar as dependências (se for rodar em outro ambiente):

bash
Copy
Edit
pip install -r requirements.txt
Principais bibliotecas utilizadas:

pandas

numpy

scikit-learn

imbalanced-learn

Como Executar
Certifique-se de que o ambiente virtual (venv_fraude) está ativo.

1. Executar a Interface de Texto (Recomendado)
bash
Copy
Edit
python ui.py
Siga as instruções no menu para:

Executar um novo experimento (escolhendo tipo de dados, uso de SMOTE, parâmetros do modelo e limiar de classificação).

Visualizar resultados de experimentos anteriores, salvos em fraud_detection_results.json.

2. Executar o Pipeline Diretamente
Execute o pipeline com configurações padrão definidas dentro do detector.py:

bash
Copy
Edit
python detector.py
3. Testar as Funções de Processamento de Dados
Para testar isoladamente as funções de carregamento, geração de dados e SMOTE:

bash
Copy
Edit
python src/data_input.py
Dados
Este projeto utiliza o dataset "Credit Card Fraud Detection", disponível no Kaggle.

O arquivo CSV (creditcard.csv) deve ser colocado na pasta data/.

As colunas esperadas são: Time, V1 a V28, Amount e a coluna alvo Class (onde 1 indica fraude).

Visão Geral dos Scripts
ui.py
Fornece uma interface de texto para o usuário.

Permite configurar os parâmetros de cada execução.

Chama o pipeline principal (detector.py).

Salva os resultados em fraud_detection_results.json.

Permite visualizar experimentos anteriores.

detector.py
Contém a função principal run_fraud_detection_pipeline_with_params().

Executa o pipeline completo: carregamento, pré-processamento, divisão treino/teste, SMOTE (opcional), treinamento (RandomForestClassifier), avaliação e geração de métricas.

src/data_input.py
Módulo com funções auxiliares:

load_csv_data: Carregamento de dados CSV.

generate_synthetic_data_scratch: Geração de dados sintéticos.

engineer_features_from_data: Engenharia de features.

augment_data_smote: Aplicação de SMOTE para balanceamento.

Experimentação
Utilize o ui.py para testar diferentes combinações de parâmetros:

Tipo de Dados: Real ou Sintético.

SMOTE: Ativar ou não no conjunto de treino.

Hiperparâmetros do Modelo: n_estimators, max_depth, min_samples_leaf, min_samples_split.

Limiar de Classificação: Ajustar (padrão 0.5) para otimizar o equilíbrio entre precisão e recall.

Os resultados, incluindo métricas como F1-score, precisão, recall e matriz de confusão, são salvos em fraud_detection_results.json.

TODO / Melhorias Futuras
✅ Integrar GridSearchCV na ui.py para busca automática de hiperparâmetros.

✅ Adicionar mais algoritmos de classificação para comparação.

✅ Implementar visualizações mais avançadas (curva ROC, importância das features).

✅ Expandir a engenharia de features com base na análise exploratória.
