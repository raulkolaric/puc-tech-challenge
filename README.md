# PUC-Tech Challenge Projects

## 📝 Descrição Geral

Este repositório agrega os projetos desenvolvidos como parte do desafio PUC-Tech. Inclui análises de dados, aplicações de console e projetos de Machine Learning, cada um demonstrando diferentes habilidades e tecnologias.

---

## 📂 Projetos Incluídos

### 1. Análise de Preferências de Estudantes
   - **Descrição:** Um Jupyter Notebook que realiza a carga, limpeza, preparação, análise exploratória e visualização de dados de um conjunto de dados sobre preferências de estudantes (provavelmente `student_preferences_extended.csv`).
   - **Principais Etapas:**
      - Carregamento e inspeção inicial dos dados.
      - Limpeza e tratamento de valores ausentes e tipos de dados.
      - Geração de insights visuais sobre:
         - Linguagens de programação preferidas.
         - Horários de estudo.
         - Formatos de conteúdo.
         - Satisfação com o curso, áreas de interesse, e mais.
   - **Tecnologias:** Python, Pandas, Matplotlib, Seaborn.
   - **Como Executar:**
      1. Abra o notebook (ex: `analise_preferencias_estudantes.ipynb`) em um ambiente Jupyter (Colab, VS Code, Jupyter Lab).
      2. Certifique-se de que as bibliotecas `pandas`, `matplotlib`, e `seaborn` estão instaladas. Se necessário, instale-as (ex: `!pip install pandas matplotlib seaborn` em uma célula do Colab).
      3. Execute as células do notebook em ordem sequencial.

### 2. Sistema de Gerenciamento de Estoque de Farmácia
   - **Descrição:** Uma aplicação de console em Python, apresentada e executável através de um Jupyter Notebook, para gerenciar o estoque de uma farmácia.
   - **Principais Funcionalidades:**
      - Adicionar, listar, atualizar e deletar medicamentos (operações CRUD).
      - Processar pedidos, incluindo verificação de receita e aplicação de descontos.
      - Monitorar estoque com avisos de nível crítico.
      - Interface de usuário via console com feedback colorido para melhor usabilidade.
      - IDs únicos para medicamentos e tratamento robusto de erros de entrada.
   - **Tecnologias:** Python (utilizando classes e funcionalidades padrão).
   - **Como Executar:**
      1. Abra o notebook (ex: `sistema_farmacia.ipynb`) em um ambiente Jupyter (Colab, VS Code, Jupyter Lab).
      2. Execute as células do notebook em ordem sequencial. A última célula iniciará o menu interativo do sistema na área de saída da célula.

### 3. Mini Detector de Fraudes em Transações
   - **Descrição:** Um projeto de Machine Learning focado na detecção de fraudes em transações de cartão de crédito. Este projeto implementa um pipeline completo que abrange desde o carregamento e pré-processamento de dados (incluindo engenharia de features e tratamento de desbalanceamento com SMOTE), até o treinamento de um modelo `RandomForestClassifier`, avaliação de sua performance (alcançando F1-score de 0.87 nos melhores experimentos) e um sistema de gerenciamento de experimentos.
   - **Destaques:**
      - Pipeline de Machine Learning de ponta a ponta.
      - Interface de Texto (TUI) (`ui.py`) para configuração e execução de experimentos, com resultados salvos em `fraud_detection_results.json`.
      - Módulos dedicados para entrada de dados (`src/data_input.py`) e execução do pipeline principal (`detector.py`).
   - **Tecnologias:** Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn.
   - **Como Executar:**
      1. Navegue até a pasta `3. Mini Detector de Fraudes em Transações/`.
      2. Siga as instruções no `README.MD` específico desta pasta para configurar o ambiente virtual (`venv_fraude/`) e instalar as dependências listadas em `requirements.txt`.
      3. Execute a interface de gerenciamento de experimentos com `python ui.py` (recomendado) ou o pipeline de detecção diretamente com `python detector.py`.

---

## 🚀 Como Começar (Geral)

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/raulkolaric/puc-tech-challenge.git](https://github.com/raulkolaric/puc-tech-challenge.git)
    cd puc-tech-challenge
    ```
2.  **Navegue até a pasta do projeto desejado** ou abra diretamente os arquivos `.ipynb` no seu ambiente Jupyter preferido.
3.  **Siga as instruções específicas de cada projeto** listadas acima para executar e interagir com as aplicações/análises.

---

## 🛠️ Requisitos Gerais

* Python 3.7+
* Ambiente Jupyter (Google Colab, Jupyter Notebook, Jupyter Lab, VS Code com extensões Python/Jupyter)
* Bibliotecas Python específicas mencionadas em cada projeto (consulte os `requirements.txt` dentro das pastas dos projetos, se disponíveis, ou as seções de tecnologia acima).

---
