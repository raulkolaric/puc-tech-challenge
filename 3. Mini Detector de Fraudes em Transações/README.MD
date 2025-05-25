
# 3. Mini Detector de Fraudes em Transações

# Mini Detector de Fraudes em Transações (Credit Card Fraud Detection)

Este projeto é uma submissão para um desafio universitário, focado na detecção de fraudes em transações de cartão de crédito utilizando técnicas de Machine Learning. Ele inclui um pipeline para processamento de dados, treinamento de modelo, avaliação e um sistema de gerenciamento de experimentos via interface de texto (TUI).
## TLDR (Importante)
O melhor modelo atingido nesse projeto é classificado como "bom" com base na comunidade de Machine learning. Não consegui rodar um Grid do tamanho que eu queria, devido ao tempo. Aqui está a melhor "run" com um f1 de 0.87:
```bash
Experimento 2 (Realizado em: 2025-05-25 10:56:08)
  Configuração dos Dados: Real CSV Data
  SMOTE no Treino: SMOTE Applied to Train Set
  Parâmetros do Modelo: {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42, 'verbose': 0, 'n_jobs': -1}
  Limiar de Classificação Usado: N/A

  Métricas (Classe 1 - Fraude):
    F1-Score:    0.8333
    Precisão:    0.8547
    Recall:      0.8130
    Matriz de Confusão (Fraude): 
    TP=100, FP=17, FN=23, TN=71062
```
Eu acho que daria para chegar ao (f1 0.9) com maior tempo de alteração dos parâmetros, porém devido a semana que foi dada essa atividade minhas obrigações não permitiram o aprofundamento maior no assunto.

Fiquei satisfeito com o score, outros treinamentos que eu fiz foram visando menor False Positives, porém o Recall ficava horroroso, por isso não incluí.

Embaixo vai uma descrição mais detalhada do projeto.
## 📄 Descrição do Projeto

O objetivo principal é construir um classificador capaz de distinguir transações fraudulentas de transações legítimas. O projeto explora:

- Carregamento e geração de dados (reais e sintéticos).
- Engenharia de features para melhorar a performance do modelo.
- Técnicas para lidar com datasets desbalanceados (SMOTE).
- Treinamento e avaliação de um modelo `RandomForestClassifier`.
- Uma interface de texto para configurar e executar experimentos, registrando os resultados.

## 📁 Estrutura de Pastas

```
├── data/ # Pasta para armazenar os datasets
│ └── creditcard.csv # Dataset principal (exemplo)
├── notebooks/ # (Opcional) Jupyter Notebooks de análise/apresentação
├── src/ # Módulos Python auxiliares
│ ├── init.py # Torna 'src' um pacote Python
│ └── data_input.py # Funções para carregamento, geração e pré-processamento de dados
├── venv_fraude/ # Ambiente virtual Python (ignorado pelo Git)
├── detector.py # Script principal do pipeline de Machine Learning
├── ui.py # Script para a interface de texto (TUI) de gerenciamento de experimentos
├── fraud_detection_results.json # Banco de dados JSON com resultados dos experimentos
└── README.md # Este arquivo
```

# ⚙️ Configuração (Setup)

### ✅ Pré-requisitos

- Python 3.7+ (recomendado 3.8+)
- pip (gerenciador de pacotes Python)

### 🔧 Configuração do Ambiente Virtual

Navegue até a pasta raiz deste projeto:
```bash
cd "3. Mini Detector de Fraudes em Transações"
```
Crie o ambiente virtual:
```bash
python -m venv venv_fraude
```
Ative o ambiente virtual:
```bash
Windows (CMD/PowerShell):
```
```bash
.\venv_fraude\Scripts\activate
```
macOS/Linux (bash/zsh):
```bash
source venv_fraude/bin/activate
```
**Você verá (venv_fraude) no início do seu terminal, indicando que o ambiente está ativo.**

## 📦 Instalando Dependências 
Gerar o arquivo requirements.txt (se ainda não existir):

Após instalar os pacotes necessários (pandas, numpy, scikit-learn, imbalanced-learn), execute:

```bash
pip freeze > requirements.txt
```
Instalar as dependências (em outro ambiente ou máquina):
```bash
pip install -r requirements.txt
```
Principais bibliotecas utilizadas:
```bash
pandas

numpy

scikit-learn

imbalanced-learn
```
# 🚀 Como Executar
Certifique-se de ativar seu ambiente virtual antes de executar qualquer comando.

### 1️⃣ Executar via Interface de Texto (Recomendado)
A interface permite configurar experimentos, escolher parâmetros e visualizar resultados.
```bash
python ui.py
```
No menu, você pode:

Executar um novo experimento (selecionar tipo de dados, aplicar SMOTE, definir hiperparâmetros e limiar).

Visualizar resultados anteriores salvos em fraud_detection_results.json.

### 2️⃣ Executar o Pipeline Diretamente
Executa o pipeline com parâmetros padrão definidos no script.

```bash
python detector.py
```

### 3️⃣ Testar Módulo de Dados
Para testar carregamento, geração de dados, engenharia de features e SMOTE:

```bash
python src/data_input.py
```

### 📊 Dados
Este projeto utiliza o dataset "Credit Card Fraud Detection", disponível no Kaggle.

Coloque o arquivo CSV (creditcard.csv) dentro da pasta data/.

As colunas esperadas são:
```bash
Time, V1, V2, ..., V28, Amount, Class
A coluna Class indica se a transação é fraudulenta (1) ou legítima (0).
```
### 🔍 Visão Geral dos Scripts
**ui.py**
```bash
-Interface de texto (TUI) para o usuário.

-Permite configurar parâmetros dos experimentos.

-Executa o pipeline (detector.py).

-Salva resultados em fraud_detection_results.json.

-Permite visualizar experimentos passados.
```
**detector.py**

Função principal: run_fraud_detection_pipeline_with_params().
```bash

-Executa todo o pipeline de ML:

-Carregamento e pré-processamento dos dados (via data_input.py).

-Divisão treino/teste.

-Aplicação de SMOTE (opcional).

-Treinamento do modelo (RandomForestClassifier).

-Avaliação do modelo.

-Retorno de métricas (F1-Score, Precisão, Recall, Matriz de Confusão).
```
**src/data_input.py**

Funções utilitárias:
```bash
Função	Descrição
load_csv_data()	Carrega dados CSV.
generate_synthetic_data_scratch()	Gera dados sintéticos.
engineer_features_from_data()	Aplica engenharia de features.
augment_data_smote()	Aplica SMOTE para balanceamento de classes.
```
## 🧪 Experimentando
Use ui.py para testar diferentes configurações:
```bash
Tipo de Dados: Real ou Sintético.

SMOTE: Aplicar ou não.

Hiperparâmetros:

n_estimators

max_depth

min_samples_leaf

min_samples_split

Limiar de Classificação: Ajustar o threshold (padrão 0.5) para melhorar o trade-off entre precisão e recall.
```
### ✅ Os resultados são armazenados em fraud_detection_results.json com:

-Configurações usadas.

-Métricas (F1-Score, Precisão, Recall da classe fraudulenta).

-Matriz de Confusão.

## 🚧 TODO / Melhorias Futuras
🔍 Integrar GridSearchCV na ui.py para busca automática de hiperparâmetros.

🤖 Adicionar outros algoritmos de classificação além do RandomForest.

📈 Implementar visualizações avançadas (ex.: curva ROC, importância de features).

🔬 Expandir a engenharia de features baseada em análises exploratórias.

