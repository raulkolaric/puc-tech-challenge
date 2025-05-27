# Sistema de Gerenciamento de Estoque de Farmácia
Esta parte do repositório contém um projeto em Python que implementa um Sistema de Gerenciamento de Estoque para uma Farmácia. A aplicação é baseada em console, construída com uma abordagem orientada a objetos, e é apresentada aqui em um formato de Jupyter Notebook para facilitar a visualização, execução e compreensão do código.

O sistema permite gerenciar medicamentos (adicionar, listar, atualizar quantidade, deletar), processar pedidos de clientes (incluindo verificação de receita e descontos) e monitorar o inventário com avisos de estoque crítico.

## 🚀 Principais Funcionalidades

* **Gerenciamento Completo de Medicamentos (CRUD):** Adição, listagem detalhada, atualização de quantidade e remoção de medicamentos.
* **Interface de Usuário Interativa:** Menu de console claro e com feedback colorido para melhor experiência do usuário.
* **Controle de Estoque:** IDs únicos para medicamentos, aviso dinâmico de estoque crítico/esgotado.
* **Processamento de Pedidos:** Simulação de vendas com verificação de receita, cálculo de totais e aplicação de descontos (ex: por CPF).
* **Validação de Entradas:** Múltiplas verificações para prevenir erros de digitação do usuário.
* **Estrutura Organizada:** Uso de classes (`Medicamento`, `Cores`) para melhor organização e legibilidade do código.
* **Dados Iniciais:** O sistema é pré-carregado com alguns medicamentos para demonstração imediata.

## 🛠️ Tecnologias Utilizadas

* **Python 3**
* **Jupyter Notebook / Google Colab:** Para apresentação e execução interativa do código.
    * O script utiliza apenas funcionalidades padrão do Python e não requer a instalação de bibliotecas externas (como pandas, numpy, etc., que não são usadas neste projeto específico).

## ⚙️ Como Usar / Executar o Notebook

1.  **Clone o repositório ou baixe o arquivo `.ipynb`** para o seu ambiente local.
    ```bash
    git clone [https://github.com/raulkolaric/puc-tech-challenge.git]
    cd [puc-tech-challenge]
    ```
2.  **Abra o arquivo do notebook** (ex: `farmacia_gerenciamento.ipynb`) em um ambiente que suporte Jupyter Notebooks:
    * Google Colab (fazendo upload do arquivo).
    * VS Code (com a extensão Python e Jupyter instaladas).
    * Jupyter Lab ou Jupyter Notebook instalado localmente.
3.  **Execute as Células em Ordem:**
    * Comece executando a primeira célula de código que define as configurações iniciais (variáveis globais, classe `Cores`).
    * Em seguida, execute a célula que define a classe `Medicamento`.
    * Prossiga executando as células que definem as funções de gerenciamento de estoque, processamento de pedidos e menu.
    * A última célula de código contém a função `main()` e o bloco `if __name__ == "__main__":` que inicializa o estoque com dados de exemplo e inicia o programa.
4.  **Interaja com o Menu:** Após executar a última célula, o menu do sistema de farmácia aparecerá na área de saída. Siga as instruções do menu para testar as funcionalidades.

## 📖 Estrutura do Notebook

O notebook é dividido em seções lógicas:
* **Introdução:** Apresentação do projeto.
* **Configurações Iniciais:** Definição de variáveis globais e classes de utilidade.
* **Definição da Classe `Medicamento`:** Modelo para os objetos de medicamento.
* **Funções de Gerenciamento de Estoque:** Código para CRUD de medicamentos.
* **Funções de Processamento de Pedidos e Resumo:** Lógica de vendas e relatórios.
* **Menu Principal e Execução:** Interface do usuário e ponto de entrada do programa.

Cada seção de código é precedida por uma célula Markdown com uma breve explicação formal do seu propósito.

## 🔮 Próximos Passos (Sugestões)

* Implementação de persistência de dados (ex: salvar/carregar estoque de um arquivo CSV ou JSON).
* Funcionalidades de busca e filtragem mais avançadas.
* Interface gráfica do usuário (GUI).

---

Sinta-se à vontade para adaptar e expandir este README conforme necessário!