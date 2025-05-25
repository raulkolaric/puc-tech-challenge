# Mini Detector de Fraudes em Transações (Credit Card Fraud Detection)

Este projeto é uma submissão para um desafio universitário, focado na detecção de fraudes em transações de cartão de crédito utilizando técnicas de Machine Learning. Ele inclui um pipeline para processamento de dados, treinamento de modelo, avaliação e um sistema de gerenciamento de experimentos via interface de texto (TUI).

## Descrição do Projeto

O objetivo principal é construir um classificador capaz de distinguir transações fraudulentas de transações legítimas. O projeto explora:

- Carregamento e geração de dados (reais e sintéticos).
- Engenharia de features para melhorar a performance do modelo.
- Técnicas para lidar com datasets desbalanceados (SMOTE).
- Treinamento e avaliação de um modelo `RandomForestClassifier`.
- Uma interface de texto para configurar e executar experimentos, registrando os resultados.

## Estrutura de Pastas
```
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
```
## Configuração (Setup)

### Pré-requisitos

- Python 3.7+ (recomendado 3.8+)
- pip (gerenciador de pacotes Python)

Passo 3: Ative o ambiente virtual
A ativação é diferente dependendo do seu sistema operacional e terminal:

Para Windows (CMD/Prompt de Comando):
Bash

.\venv_fraude\Scripts\activate.bat
Para Windows (PowerShell):
Bash

.\venv_fraude\Scripts\Activate.ps1
(Se encontrar um erro de política de execução no PowerShell, pode ser necessário executar Set-ExecutionPolicy RemoteSigned -Scope CurrentUser uma vez.)
Para macOS ou Linux (terminal bash/zsh), ou Git Bash no Windows:
Bash

source venv_fraude/bin/activate
Após a ativação, você deverá ver (venv_fraude) no início do seu prompt do terminal.

Instalando Dependências
Com o ambiente virtual (venv_fraude) ativo, instale as bibliotecas Python necessárias.

Passo 1: Crie o arquivo requirements.txt (se ainda não o fez)
Se você já instalou os pacotes manualmente (pandas, numpy, scikit-learn, imbalanced-learn) enquanto o venv_fraude estava ativo, gere o arquivo de requisitos:

Bash

pip freeze > requirements.txt
(Este comando lista todos os pacotes instalados no ambiente ativo e salva essa lista em requirements.txt.)

Passo 2: Instale as dependências usando o arquivo requirements.txt
Se você está configurando o projeto em um novo local ou se outra pessoa precisa configurá-lo, e você tem o arquivo requirements.txt:

Bash

pip install -r requirements.txt
Isso instalará todas as bibliotecas listadas no arquivo. As principais incluem: pandas, numpy, scikit-learn, imbalanced-learn.

Como Executar
Certifique-se de que seu ambiente virtual (venv_fraude) está ativo antes de executar os scripts. Todos os comandos devem ser executados a partir da pasta raiz deste projeto (3. Mini Detector de Fraudes em Transações/).

1. Executando Experimentos via Interface de Texto (Recomendado)
Esta é a forma principal de interagir com o projeto. Para iniciar a interface de texto:

Bash

python ui.py
Siga as instruções no menu para:

Executar um novo experimento (escolhendo tipo de dados, SMOTE, parâmetros do modelo, limiar de classificação).
Visualizar resultados de experimentos anteriores salvos em fraud_detection_results.json.
2. Executando o Pipeline Diretamente (para teste do detector.py)
O script detector.py pode ser executado diretamente para um teste rápido do pipeline com configurações padrão (definidas dentro do seu bloco if __name__ == "__main__":).

Bash

python detector.py
3. Testando as Funções de Entrada de Dados (para teste do data_input.py)
Para testar as funções de carregamento, geração de dados, engenharia de features e SMOTE de forma isolada, execute o script data_input.py diretamente:

Bash

python src/data_input.py

TODO / Melhorias Futuras
✅ Integrar GridSearchCV na ui.py para busca automática de hiperparâmetros.

✅ Adicionar mais algoritmos de classificação para comparação.

✅ Implementar visualizações mais avançadas (curva ROC, importância das features).

✅ Expandir a engenharia de features com base na análise exploratória.

