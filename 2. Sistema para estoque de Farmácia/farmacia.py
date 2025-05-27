'''
Raul Kolaric 

O script foi feito com class, e eu adicionei algumas features extras:
-menu UI reciclado;
-Receita de medicamento ;
-Classificação de genérico ;
-ID único númerico crescente ;
-Cores de texto para melhorar a UX ;
-Diversas proteções contra erro de usúario ;
-Desconto por CPF (pra ficar fiel | Não tem a formula do CPF pra não ficar chato testar) ;
-Aviso dinamico de estoque crítico! ;

Foram adicionados previamente 5 remédios na memória para melhorar a visualização do programa!
'''

estoque = {}
proximo_id_medicamento = 1
LIMITE_ESTOQUE_CRITICO = 5

class Cores: 
    RESET = '\033[0m'
    AZUL = '\033[94m'
    VERDE = '\033[92m'
    AMARELO = '\033[93m'
    VERMELHO = '\033[91m'
    MAGENTA = '\033[95m'
    CIANO = '\033[96m'
    BRANCO = '\033[97m'

    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Medicamento:  #Inicializa um novo objeto Medicamento.
    def __init__(self, nome, preco, receita, generico, quantidade=0, id=None):
        self.nome = nome   #nome (str)
        self.preco = preco  #preco (float)
        self.receita = receita  #receita (bool)
        self.generico = generico    #generico (bool)
        self.quantidade = quantidade    #quantidade (int). Pode ser 0.
        self.id = id    #Pra ser mais pratico botar 2 remedios iguais do mesmo preço, e listar eles
#Funcções extras no final do código (main, menu, estoque_crítico)
# Opção número 1
def listar_estoque():
    """
    Lista todos os medicamentos no estoque, numerados, com seus atributos
    formatados e espaçados.
    """
    print(f"\n{Cores.BOLD}{Cores.CIANO}--- LISTA COMPLETA DO ESTOQUE ---{Cores.RESET}\n")
    if not estoque:
        print(f"{Cores.AMARELO}Estoque vazio. Não há medicamentos para listar.{Cores.RESET}")
        return

    numero_item = 1
    for medicamento_obj in estoque.values():
        # LÓGICA MODIFICADA AQUI PARA O AVISO DE ESGOTADO/CRÍTICO
        if medicamento_obj.quantidade == 0:
            nome_display = f"{Cores.BOLD}{Cores.VERMELHO}{medicamento_obj.nome.upper()}       ESGOTADO{Cores.RESET}"
        elif medicamento_obj.quantidade > 0 and medicamento_obj.quantidade <= LIMITE_ESTOQUE_CRITICO:
            nome_display = f"{Cores.VERMELHO}{medicamento_obj.nome.upper()}       ESTOQUE CRÍTICO{Cores.RESET}"
        else:
            nome_display = f"{Cores.AMARELO}{medicamento_obj.nome.upper()}{Cores.RESET}"
        # FIM DA LÓGICA MODIFICADA

        print(f"{numero_item}. {nome_display}")
        print(f"      Preço: R${medicamento_obj.preco:.2f}") # Note: regular spaces for alignment
        cor_qtd_listar = Cores.VERMELHO if medicamento_obj.quantidade <= LIMITE_ESTOQUE_CRITICO else Cores.VERDE
        print(f"      Quantidade: {cor_qtd_listar}{medicamento_obj.quantidade}{Cores.RESET}") # Note: regular spaces for alignment
        print(f"      Exige Receita: {'Sim' if medicamento_obj.receita else 'Não'}") # Note: regular spaces for alignment
        print(f"      É Genérico: {'Sim' if medicamento_obj.generico else 'Não'}") # Note: regular spaces for alignment
        print(f"      ID Único: {medicamento_obj.id}") # Note: regular spaces for alignment
        print(f"{Cores.CIANO}---------------------------------{Cores.RESET}")

        numero_item += 1
    print(f"{Cores.BOLD}{Cores.CIANO}--- FIM DA LISTA DE ESTOQUE ---{Cores.RESET}")
    
#Opção número 2
def adicionar_medicamento():
    global proximo_id_medicamento
    #Input nome medicamento
    print(f"\n{Cores.BOLD}{Cores.VERDE}--- ADICIONAR NOVO MEDICAMENTO ---{Cores.RESET}")

    nome_medicamento = input("Digite o nome do medicamento (ou digite '0' para cancelar): ").strip()
    
    if nome_medicamento == '0':
        print(f"{Cores.AZUL}Operação de adição de medicamento cancelada.{Cores.RESET}")
        return 

    if not nome_medicamento:
        print(f"{Cores.VERMELHO}Nome do medicamento não pode ser vazio.{Cores.RESET}")
        return
    #Input preço medicamento 
    while True: 
        try:
            print(f"Digite o preço de {Cores.AMARELO}{nome_medicamento}{Cores.RESET} (ex: 15.75): ", end="")
            preco_str = input().strip()
            preco_medicamento = float(preco_str)    #conversão float
            if preco_medicamento < 0:
                print(f"{Cores.VERMELHO}O preço não pode ser negativo. Tente novamente.{Cores.RESET}")
                continue 
            break 
        except ValueError:
            print(f"{Cores.VERMELHO}Preço inválido. Por favor, digite um número.{Cores.RESET}")
    #Input bool receita
    while True:
        print(f"Exige receita médica para {Cores.AMARELO}{nome_medicamento}{Cores.RESET}? (s/n): ", end="")
        receita_str = input().lower().strip()
        if receita_str == 's':
            receita_medicamento = True
            break
        elif receita_str == 'n':
            receita_medicamento = False
            break
        else:
            print(f"{Cores.VERMELHO}Resposta inválida. Por favor, digite 's' para sim ou 'n' para não.{Cores.RESET}")
    #Input bool genérico
    while True:
        generico_str = input(f"É um medicamento genérico? (s/n): ").lower().strip()
        if generico_str == 's':
            generico_medicamento = True
            break
        elif generico_str == 'n':
            generico_medicamento = False
            break
        else:
            print(f"{Cores.VERMELHO}Resposta inválida. Por favor, digite 's' para sim ou 'n' para não.{Cores.RESET}")
    #Input quantidade inicial medicamento
    while True:
        try:
            print(f"Digite a quantidade inicial de {Cores.AMARELO}{nome_medicamento}{Cores.RESET}: ", end="")
            quantidade_str = input().strip()
            quantidade_medicamento = int(quantidade_str)
            if quantidade_medicamento < 0: #deixei que pode adicionar com 0.
                print(f"{Cores.VERMELHO}A quantidade inicial não pode ser negativa. Tente novamente.{Cores.RESET}")
                continue
            break
        except ValueError:
            print(f"{Cores.VERMELHO}Quantidade inválida. Por favor, digite um número inteiro.{Cores.RESET}")
    
    try:
        novo_medicamento_obj = Medicamento(nome_medicamento, preco_medicamento, receita_medicamento, generico_medicamento, quantidade_medicamento, proximo_id_medicamento)
        estoque[proximo_id_medicamento] = novo_medicamento_obj  #criação do obj
        print(f"\nMedicamento {Cores.AMARELO}'{nome_medicamento}'{Cores.RESET} adicionado com {Cores.VERDE} ID {proximo_id_medicamento} {Cores.RESET}e {Cores.MAGENTA}{quantidade_medicamento} unidades.{Cores.RESET}")
        proximo_id_medicamento += 1 #atualização do id pro próximo medicamento
        
    except ValueError as e:
        print(f"{Cores.VERMELHO}Erro ao criar o medicamento: {e}. Não foi adicionado ao estoque.{Cores.RESET}")

#Opção número 3
def atualizar_estoque():
    """
    Permite ao usuário adicionar uma quantidade a um medicamento existente
    escolhendo-o por ID. A quantidade fornecida será SOMADA à quantidade atual.
    """
    print(f"\n{Cores.BOLD}{Cores.AZUL}--- ADICIONAR QUANTIDADE AO ESTOQUE POR ID ---{Cores.RESET}")
    
    if not estoque:
        print(f"{Cores.AMARELO}Estoque vazio. Não há medicamentos para adicionar quantidade.{Cores.RESET}")
        return

    print("Medicamentos disponíveis para adicionar quantidade:")    #Display de medicamentos
    for id_medicamento, medicamento_obj in estoque.items():
        if medicamento_obj.quantidade <= LIMITE_ESTOQUE_CRITICO: 
            print(f"     {id_medicamento}. {Cores.AMARELO}{medicamento_obj.nome}{Cores.RESET} (Qtd atual: {Cores.VERMELHO}{medicamento_obj.quantidade}{Cores.RESET})")
        else:
            print(f"     {id_medicamento}. {Cores.AMARELO}{medicamento_obj.nome}{Cores.RESET} (Qtd atual: {Cores.MAGENTA}{medicamento_obj.quantidade}{Cores.RESET})")

    print(f"{Cores.BOLD}{Cores.AZUL}---------------------------------------------{Cores.RESET}")
    while True: 
        try:#
            print(f"Digite o {Cores.BOLD}ID{Cores.RESET} do medicamento para adicionar quantidade (ou '{Cores.AMARELO}0{Cores.RESET}' para cancelar): ", end="")
            id_escolhido_str = input().strip()
            #escape
            if id_escolhido_str == '0':
                print(f"{Cores.AZUL}Operação de adição de quantidade cancelada.{Cores.RESET}")
                return 

            id_escolhido = int(id_escolhido_str)
            
            if id_escolhido in estoque:
                medicamento_para_atualizar = estoque[id_escolhido]
                print(f"\nVocê escolheu: {medicamento_para_atualizar.nome} (Qtd atual: {medicamento_para_atualizar.quantidade})")
          #após selecionado o medicamento
                while True: 
                    try:
                        print(f"Digite a quantidade DE {Cores.AMARELO}{medicamento_para_atualizar.nome}{Cores.RESET} para ser SOMADA ao estoque: ", end="")
                        qtd_a_adicionar_str = input().strip()
                        qtd_a_adicionar = int(qtd_a_adicionar_str)
                        
                        if qtd_a_adicionar >= 0: 
                            medicamento_para_atualizar.quantidade += qtd_a_adicionar 
                            print(f"{Cores.VERDE}Quantidade de '{medicamento_para_atualizar.nome}' atualizada para {medicamento_para_atualizar.quantidade} unidades.{Cores.RESET}")
                            return 
                        else:
                            print(f"{Cores.AMARELO}A quantidade a ser somada não pode ser negativa. Tente novamente.{Cores.RESET}")
                    except ValueError:
                        print(f"{Cores.VERMELHO}Quantidade inválida. Por favor, digite um número inteiro.{Cores.RESET}")
            else:
                print(f"{Cores.AMARELO}ID '{id_escolhido_str}' não encontrado no estoque. Por favor, digite um ID existente.{Cores.RESET}")
        except ValueError:
            print(f"{Cores.VERMELHO}Entrada inválida para o ID. Por favor, digite um número.{Cores.RESET}")

#Opção número 4
def deletar_medicamento():
    """
    Permite ao usuário deletar uma entrada de medicamento do estoque,
    escolhendo-a por ID.
    """
    print(f"\n{Cores.BOLD}{Cores.VERMELHO}--- DELETAR MEDICAMENTO POR ID ---{Cores.RESET}")
    
    if not estoque:
        print(f"{Cores.AMARELO}Estoque vazio. Não há medicamentos para deletar.{Cores.RESET}")
        return
    #Display de todos os medicamentos
    print("Medicamentos disponíveis para deleção:")
    for id_medicamento, medicamento_obj in estoque.items():
        print(f"     {id_medicamento}. {Cores.AMARELO}{medicamento_obj.nome}{Cores.RESET} (Qtd: {medicamento_obj.quantidade})")
    print(f"{Cores.BOLD}{Cores.VERMELHO}----------------------------------{Cores.RESET}")

    while True: 
        try:
            print(f"Digite o {Cores.BOLD}ID{Cores.RESET} do medicamento a ser deletado (ou '{Cores.AMARELO}0{Cores.RESET}' para cancelar): ", end="")
            id_deletar_str = input().strip()
            
            if id_deletar_str == '0':
                print(f"{Cores.AZUL}Operação de deleção de medicamento cancelada.{Cores.RESET}")
                return 

            id_deletar = int(id_deletar_str)
            
            if id_deletar in estoque:
                nome_medicamento_deletado = estoque[id_deletar].nome 
                del estoque[id_deletar] 
                print(f"{Cores.VERDE}Medicamento '{nome_medicamento_deletado}' (ID: {id_deletar}) removido com sucesso do estoque.{Cores.RESET}")
                break 
            else:
                print(f"{Cores.AMARELO}ID '{id_deletar_str}' não encontrado no estoque. Por favor, digite um ID existente.{Cores.RESET}")
        except ValueError:
            print(f"{Cores.VERMELHO}Entrada inválida para o ID. Por favor, digite um número inteiro.{Cores.RESET}")

#Opção número 5
def processar_pedidos():
    print(f"\n{Cores.BOLD}{Cores.CIANO}--- PROCESSAR PEDIDOS ---{Cores.RESET}")

    if not estoque:
        print(f"{Cores.AMARELO}Estoque vazio. Não é possível processar pedidos.{Cores.RESET}")
        return

    pedido_atual = [] 
    continuar_adicionando_itens = True
    #Tabela para visualização
    while continuar_adicionando_itens:
        print(f"\n{Cores.BOLD}{Cores.AZUL}--- Medicamentos Disponíveis ---{Cores.RESET}")
        print(f"  {'ID':<4} | {'Nome do Medicamento':<30} | {'Preço':<10} | {'Qtd':<3} | {'Info':<15}")
        print(f"  {'-'*4} | {'-'*30} | {'-'*10} | {'-'*3} | {'-'*15}")

        for id_med, med_obj in estoque.items():
            cor_qtd = Cores.VERMELHO if med_obj.quantidade <= LIMITE_ESTOQUE_CRITICO else Cores.VERDE
            id_field = f"{Cores.BRANCO}{id_med:<4}{Cores.RESET}"
            nome_field = f"{Cores.AMARELO}{med_obj.nome:<30.30}{Cores.RESET}"
            preco_text_val = f"R$ {med_obj.preco:.2f}"
            preco_field = f"{preco_text_val:>10}"
            qtd_numero_str = f"{med_obj.quantidade:>3}" 
            qtd_colorido_str = f"{cor_qtd}{qtd_numero_str}{Cores.RESET}"
            qtd_field = qtd_colorido_str 

            receita_text_display = "Req. Receita" if med_obj.receita else "" 
            info_text_padded = f"{receita_text_display:<15.15}" 
            info_field = f"{Cores.MAGENTA}{info_text_padded}{Cores.RESET}"
            print(f"  {id_field} | {nome_field} | {preco_field} | {qtd_field} | {info_field}")
            
        print(f"{Cores.BOLD}{Cores.AZUL}{'-'*76}{Cores.RESET}") 
        id_escolhido_str = input(f"Digite o ID do medicamento para adicionar ao pedido (ou '0' para finalizar): ").strip()
        #escape
        if id_escolhido_str == '0':
            continuar_adicionando_itens = False
            continue

        try:
            id_escolhido = int(id_escolhido_str)
            if id_escolhido not in estoque:
                print(f"{Cores.VERMELHO}ID inválido ou não encontrado. Tente novamente.{Cores.RESET}")
                continue

            medicamento_selecionado = estoque[id_escolhido]

            if medicamento_selecionado.quantidade == 0:
                print(f"{Cores.VERMELHO}O medicamento '{medicamento_selecionado.nome}' está esgotado.{Cores.RESET}")
                continue
            
            if medicamento_selecionado.receita:
                print(f"{Cores.MAGENTA}Atenção: O medicamento '{Cores.AMARELO}{medicamento_selecionado.nome}{Cores.RESET}' {Cores.MAGENTA}exige receita médica.{Cores.RESET}")
                while True:
                    confirmacao_receita = input("O cliente apresentou a receita? (s/n): ").lower().strip() # Plain input
                    if confirmacao_receita in ['s', 'n']:
                        break
                    print(f"{Cores.VERMELHO}Resposta inválida. Por favor, digite 's' ou 'n'.{Cores.RESET}")
                
                if confirmacao_receita == 'n':
                    print(f"{Cores.AMARELO}Venda de '{medicamento_selecionado.nome}' não pode ser processada sem a apresentação da receita.{Cores.RESET}")
                    continue 

            while True:
                try:
                    print(f"Digite a quantidade desejada de {Cores.AMARELO}{medicamento_selecionado.nome}{Cores.RESET} (Disponível: {medicamento_selecionado.quantidade}): ", end="")
                    qtd_desejada_str = input().strip()
                    qtd_desejada = int(qtd_desejada_str)

                    if qtd_desejada <= 0:
                        print(f"{Cores.VERMELHO}A quantidade deve ser um número positivo.{Cores.RESET}")
                        continue 

                    quantidade_ja_no_carrinho = 0
                    item_existente_no_carrinho = None
                    for item_carrinho in pedido_atual:
                        if item_carrinho['id'] == id_escolhido:
                            item_existente_no_carrinho = item_carrinho
                            quantidade_ja_no_carrinho = item_carrinho['quantidade_pedida']
                            break
                    
                    if (quantidade_ja_no_carrinho + qtd_desejada) > medicamento_selecionado.quantidade:
                        disponivel_para_adicionar = medicamento_selecionado.quantidade - quantidade_ja_no_carrinho
                        print(f"{Cores.VERMELHO}Erro: Estoque insuficiente para adicionar {qtd_desejada} unidade(s).{Cores.RESET}")
                        print(f"  Você já tem {quantidade_ja_no_carrinho} no carrinho. Estoque total: {medicamento_selecionado.quantidade}.")
                        if disponivel_para_adicionar > 0:
                             print(f"  Você pode adicionar no máximo mais {Cores.VERDE}{disponivel_para_adicionar}{Cores.RESET} unidade(s).")
                        else:
                            print(f"  {Cores.AMARELO}Não há mais unidades disponíveis para adicionar deste item ao carrinho.{Cores.RESET}")
                        continue 
                    
                    if item_existente_no_carrinho:
                        item_existente_no_carrinho['quantidade_pedida'] += qtd_desejada
                        print(f"{Cores.VERDE}Quantidade atualizada para '{medicamento_selecionado.nome}' no pedido: {item_existente_no_carrinho['quantidade_pedida']} unidade(s).{Cores.RESET}")
                    else:
                        pedido_atual.append({
                            'id': id_escolhido,
                            'nome': medicamento_selecionado.nome,
                            'quantidade_pedida': qtd_desejada,
                            'preco_unitario': medicamento_selecionado.preco
                        })
                        print(f"{Cores.VERDE}{qtd_desejada} unidade(s) de '{medicamento_selecionado.nome}' adicionada(s) ao pedido.{Cores.RESET}")
                    break 
                except ValueError:
                    print(f"{Cores.VERMELHO}Quantidade inválida. Por favor, digite um número inteiro.{Cores.RESET}")
        except ValueError:
            print(f"{Cores.VERMELHO}ID inválido. Por favor, digite um número.{Cores.RESET}")
        except Exception as e: 
            print(f"{Cores.VERMELHO}Ocorreu um erro inesperado: {e}{Cores.RESET}")

    if not pedido_atual:
        print(f"{Cores.AMARELO}Nenhum item no pedido. Processo de venda cancelado.{Cores.RESET}")
        return

    print(f"\n{Cores.BOLD}{Cores.VERDE}--- RESUMO DO PEDIDO ---{Cores.RESET}")
    valor_total_bruto = 0.0
    print(f"  {'Item':<5} | {'Nome do Medicamento':<30.30} | {'Qtd':>4} | {'Preço Unit.':>12} | {'Subtotal':>10}")
    print(f"  {'-'*5} | {'-'*30} | {'-'*4} | {'-'*12} | {'-'*10}")

    for i, item in enumerate(pedido_atual):
        subtotal_item = item['preco_unitario'] * item['quantidade_pedida']
        item_num_f = f"{i+1:<5}"
        nome_f = f"{item['nome']:<30.30}" 
        qtd_f = f"{item['quantidade_pedida']:>4}"
        preco_unit_text = f"R$ {item['preco_unitario']:.2f}"
        preco_unit_f = f"{preco_unit_text:>12}" 
        subtotal_text = f"R$ {subtotal_item:.2f}"
        subtotal_f = f"{subtotal_text:>10}"     
        print(f"  {item_num_f} | {nome_f} | {qtd_f} | {preco_unit_f} | {subtotal_f}")
        valor_total_bruto += subtotal_item
    
    print(f"{Cores.BOLD}{'-'*75}{Cores.RESET}")
    print(f"{Cores.BOLD}Valor Total Bruto: R${valor_total_bruto:.2f}{Cores.RESET}")

    desconto_aplicado = 0.0
    cpf_cliente = None 
    valor_final_a_pagar = valor_total_bruto

    while True:
        informar_cpf = input("Deseja informar o CPF para obter 20% de desconto? (s/n): ").lower().strip() # Plain input
        if informar_cpf == 's':
            cpf_cliente = input("Digite o CPF do cliente: ").strip() # Plain input
            if cpf_cliente: 
                desconto_aplicado = valor_total_bruto * 0.20
                valor_final_a_pagar = valor_total_bruto - desconto_aplicado
                print(f"{Cores.VERDE}Desconto de 20% (R${desconto_aplicado:.2f}) aplicado.{Cores.RESET}")
            else:
                print(f"{Cores.AMARELO}CPF não informado. Nenhum desconto será aplicado.{Cores.RESET}")
                cpf_cliente = None 
            break
        elif informar_cpf == 'n':
            print(f"{Cores.AMARELO}Nenhum desconto aplicado.{Cores.RESET}")
            break
        else:
            print(f"{Cores.VERMELHO}Opção inválida. Por favor, digite 's' ou 'n'.{Cores.RESET}")

    print(f"{Cores.BOLD}{Cores.VERDE}Valor Final a Pagar: R${valor_final_a_pagar:.2f}{Cores.RESET}")

    while True:
        print(f"Confirmar venda e atualizar estoque ({Cores.BOLD}{Cores.VERDE}s{Cores.RESET}/{Cores.BOLD}{Cores.VERMELHO}n{Cores.RESET})? ", end="")
        confirmar_venda = input().lower().strip()
        if confirmar_venda == 's':
            for item in pedido_atual:
                estoque[item['id']].quantidade -= item['quantidade_pedida']
            print(f"{Cores.VERDE}{Cores.BOLD}Venda processada com sucesso! Estoque atualizado.{Cores.RESET}")
            if cpf_cliente:
                print(f"CPF do cliente registrado: {cpf_cliente}")
            break
        elif confirmar_venda == 'n':
            print(f"{Cores.AMARELO}Venda cancelada pelo operador. Estoque não foi alterado.{Cores.RESET}")
            break
        else:
            print(f"{Cores.VERMELHO}Opção inválida. Digite 's' ou 'n'.{Cores.RESET}")

#Opção número 6
def exibir_resumo():
    print(f"\n{Cores.BOLD}{Cores.MAGENTA}--- RESUMO DO ESTOQUE ---{Cores.RESET}")
    if not estoque:
        print(f"{Cores.AMARELO}Estoque vazio.{Cores.RESET}")
        return

    total_medicamentos = len(estoque)
    total_unidades = 0
    valor_total_estoque = 0.0
    itens_em_critico = 0
    nomes_criticos = []

    for med_id, med_obj in estoque.items():
        total_unidades += med_obj.quantidade
        valor_total_estoque += med_obj.quantidade * med_obj.preco
        if med_obj.quantidade <= LIMITE_ESTOQUE_CRITICO:
            itens_em_critico +=1
            nomes_criticos.append(f"{med_obj.nome} (Qtd: {med_obj.quantidade})")
    
    print(f"Total de Tipos de Medicamentos (IDs diferentes): {Cores.CIANO}{total_medicamentos}{Cores.RESET}")
    print(f"Total de Unidades de Medicamentos no Estoque: {Cores.CIANO}{total_unidades}{Cores.RESET}")
    print(f"Valor Total Estimado do Estoque: {Cores.VERDE}R${valor_total_estoque:.2f}{Cores.RESET}")
    
    if itens_em_critico > 0:
        print(f"Medicamentos em Estoque Crítico ou Esgotado ({Cores.VERMELHO}{itens_em_critico}{Cores.RESET} tipo(s)):")
        for nome_crit in nomes_criticos:
            print(f"  - {Cores.AMARELO}{nome_crit}{Cores.RESET}")
    else:
        print(f"{Cores.VERDE}Nenhum medicamento em estoque crítico.{Cores.RESET}")
    print(f"{Cores.BOLD}{Cores.MAGENTA}--------------------------{Cores.RESET}")


def main():
    while True:
        menu()
        opcao = input("Escolha uma opção: ").strip()

        if opcao == '1':
            listar_estoque() 
        elif opcao == '2':
            adicionar_medicamento() 
        elif opcao == '3':
            atualizar_estoque()
        elif opcao == '4':
            deletar_medicamento()
        elif opcao == '5':
            processar_pedidos() 
        elif opcao == '6':
            exibir_resumo()
        elif opcao == '7':
            print("Saindo do programa. Obrigado!")
            break
        else:
            print(f"{Cores.VERMELHO}Opção inválida. Por favor, escolha uma opção de 1 a 7.{Cores.RESET}")


def menu():

    print(f"\n{Cores.BOLD}{Cores.CIANO}--- MENU DE OPÇÕES ---{Cores.RESET}")

    tem_critico, qtd_criticos = estoque_critico()
    if tem_critico:
        print(f"{Cores.BOLD}{Cores.VERMELHO}AVISO: {qtd_criticos} ITENS EM ESTOQUE CRÍTICO/ESGOTADO!{Cores.RESET}")
        print(f"{Cores.BOLD}{Cores.VERMELHO}1. Listar Estoque Completo{Cores.RESET}")
    else:
        # Se não houver itens críticos, a opção de listar é exibida normalmente
        print(f"{Cores.AZUL}1. Listar Estoque Completo{Cores.RESET}")
    
    print(f"{Cores.AZUL}2. Adicionar Novo Medicamento{Cores.RESET}")
    print(f"{Cores.AZUL}3. Atualizar Quantidade por ID{Cores.RESET}")
    print(f"{Cores.AZUL}4. Deletar Entrada de Medicamento por ID{Cores.RESET}")
    print(f"{Cores.AZUL}5. Processar Pedidos por ID{Cores.RESET}")
    print(f"{Cores.AZUL}6. Exibir Resumo do Estoque{Cores.RESET}")
    print(f"{Cores.BOLD}{Cores.AMARELO}7. Sair{Cores.RESET}")
    print(f"{Cores.BOLD}{Cores.CIANO}---------------------------------------------{Cores.RESET}")


def estoque_critico():   #Verificação de Quantidade de medicamento, pra ver se esta acima do nível crítico.

    if not estoque: # Se o estoque está vazio, não há nada crítico
        return False, 0

    itens_criticos = 0
    for medicamento_obj in estoque.values():
        if medicamento_obj.quantidade <= LIMITE_ESTOQUE_CRITICO:
            itens_criticos += 1
    
    return itens_criticos > 0, itens_criticos


if __name__ == "__main__":

    med1 = Medicamento("Paracetamol 500mg", 12.50, False, True, 50, proximo_id_medicamento)
    estoque[proximo_id_medicamento] = med1
    proximo_id_medicamento += 1

    med5 = Medicamento("Clonazepam 2mg", 32.00, True, False, 0, proximo_id_medicamento)
    estoque[proximo_id_medicamento] = med5
    proximo_id_medicamento += 1

    med2 = Medicamento("Dipirona Gotas 20ml", 11.90, False, True, 25, proximo_id_medicamento)
    estoque[proximo_id_medicamento] = med2
    proximo_id_medicamento += 1

    med3 = Medicamento("Amoxicilina 875mg", 45.99, True, False, 10, proximo_id_medicamento)
    estoque[proximo_id_medicamento] = med3
    proximo_id_medicamento += 1
    
    med4 = Medicamento("Ibuprofeno 400mg", 8.75, False, True, 5, proximo_id_medicamento)
    estoque[proximo_id_medicamento] = med4
    proximo_id_medicamento += 1
    
    med5 = Medicamento("BOPKOF-D-GFDK-FD 1000mg", 8.75, False, True, 5, proximo_id_medicamento)
    estoque[proximo_id_medicamento] = med4
    proximo_id_medicamento += 1

    main()