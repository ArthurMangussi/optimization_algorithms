'''
Decoder do artigo
'''

import sys
sys.path.append('//Users//arthurdantasmangussi//@Mestrado ITA//Códigos')
from kruskal_clean import GrafoKruskal
import numpy as np 
from greedy_heuristic import GreedyAlgorithm


class Decoder:
    '''
    Decoder do artigo
    '''

    @staticmethod
    def define_hubs(random_key_chromosome: list, n: int, p: int) :
        """
        Função que define quais vértice são hubs, baseando-se no vetor de chaves aleatórias (cromossomo).
        Parâmetros:
        -> random_key_chromosome: vetor de chaves aleatórias
        -> n: número de vértices do grafo
        -> p: número de vértices que serão classificados como hubs.
        """
        chromosome_and_index = [(random_key_chromosome[i], i + 1) for i in range(n)]
        chromosome_and_index.sort(
            key=lambda x: x[0]
        )  # ordenando os genes do cromossomo com base nas chaves aleatórias
        hubs = chromosome_and_index[:p]
        non_hubs = chromosome_and_index[p:]

        #print(f"Nós hubs: {[hub[1] for hub in hubs]}\nNós não-hubs: {[n_hub[1] for n_hub in non_hubs]}")
        return [hub[1] for hub in hubs], [n_hub[1] for n_hub in non_hubs]

    @staticmethod
     
    def assign_non_hubs_to_hubs(random_key_chromosome: list, n: int, p: int, hnodes: list):
        '''
        Função que atribui os nós não-hubs para os hubs, baseando-se na parte do cromossomo com n-p genes.
            Parâmetros:
            -> random_key_chromosome: vetor de chaves aleatórias
            -> n: número de vértices do grafo
            -> p: número de vértices que serão classificados como hubs
            -> hubs_nodes: uma lista contendo os nós hubs.
        '''

        rho = 1/p
        non_hubs = [non_hub for non_hub in range(1,n+1) if non_hub not in hnodes]
        ranges = []
        non_hubs_output = []

        for k in range(p):
            lower = k * rho
            upper = (k+1) * rho
            ranges.append([lower, upper])

        t = [(hnodes[hub], ranges[hub]) for hub in range(len(hnodes))]
            

        for non_hub_node in range(n - p):
            for interval in range(len(t)):
                lower_bound = t[interval][1][0]
                upper_bound = t[interval][1][1]
                if lower_bound <= random_key_chromosome[n : n + (n - p)][non_hub_node] < upper_bound:
                    #print(f'Rk = {cromossomo_artigo[n : n + (n - p)][non_hub_node]} está no intervalo {t[interval]}')
                    non_hubs_output.append((t[interval][0], non_hubs[non_hub_node]))
                    break 

        return non_hubs_output
    
    @staticmethod
     
    def add_edge(list_of_nodes: list) -> list:
        """
        Função que adiciona todas as arestas entre os hubs.
        Parâmetros:
        list_of_nodes: lista contendo todos os nós que precisam ter arestas adicionadas.
        """
        all_edges = []
        for i in range(len(list_of_nodes)):
            for j in range(i + 1, len(list_of_nodes)):
                all_edges.append((list_of_nodes[i], list_of_nodes[j]))
        return all_edges
    
    @staticmethod
     
    def arcs_hubs_for_building_tree(random_key_chromosome: list, list_hubs: list, p: int):
        """
        Função para determinar, com base nas chaves aleatórias do cromossomo, quais serão as p-1 arestas que
        formarão a árvore de hubs.
        Parâmetros
            random_key_chromosome: lista que contém o cromossomo com as chaves aleatórias
            list_hubs: lista contendo as chaves aleatórias contidas na parte p(p-1)/2 do cromossomo inicial
            p: quantidade de hubs que precisam ser abertos
        """
        list_edges_of_hubs = Decoder.add_edge(list_hubs)

        # Adicionar as arestas com valor das chaves aleatórias
        random_keys_with_edges = []
        for edge, random_key in zip(
            list_edges_of_hubs, random_key_chromosome[-int((p * (p - 1)) / 2) :]
        ):
            random_keys_with_edges.append([edge, random_key])

        random_keys_with_edges.sort(key=lambda x: x[1])

        hubs_mapped = GreedyAlgorithm.create_map(list_hubs, p)
        
        g = GrafoKruskal(p) # inicializando o grafo com p vértices

        for hub in range(len(random_keys_with_edges)):
            g.adiciona_borda(random_keys_with_edges[hub][0][0], random_keys_with_edges[hub][0][1], random_keys_with_edges[hub][1]) # cria aresta entre hubs com as chaves aleatórias
       

        graph_mapped = GreedyAlgorithm.change_for_kruskal(hubs_mapped, g.grafo) 

        g.grafo = graph_mapped

        retorno = g.kruskal_algoritmo() # Algoritmo de Kruskal

        tree = GreedyAlgorithm.reverse_solution(hubs_mapped, retorno)
        #print('Árvore geradora mínima = ', tree)

        sol = [(nodes[0], nodes[1]) for nodes in tree]
        return sorted(sol)
    
    @staticmethod
     
    def chromosome_solution(*args):
        "Função para concatenar todas as arestas da solução do decoder."
        edges_concatenated = []
        for arg in args:
            edges_concatenated += arg
        #print(f"Solução: {sorted(edges_concatenated)}")
        return sorted(edges_concatenated)
    
    @staticmethod
     
    def find_item_in_tuple(iterator: int, list_of_all_edges: list) -> int:
        """
        Função para encontrar qual hub o nó não-hub está ligado.
        Parâmetros
        iterator: é um iterador para correr o vetor de tuplas
        list_of_all_edges: é uma lista de tuplas que representa a solução do THLP.
        """
        for edges in range(len(list_of_all_edges)):
            if iterator in list_of_all_edges[edges]:
                return list_of_all_edges[edges][0]
            
    @staticmethod
     
    def building_representation_of_solution(
    list_of_hubs: list, decoder_solution: list, n: int
) -> list:
        """
        Função que constrói a representação da solução. A solução será representada por um vetor, o qual, se o elemento n for hub, terá
        o o indíce do hub no vetor. Caso contrário, terá o indíce de qual hub atende o nó não-hub
        Parâmetros:
        list_of_hubs: lista contendo quais nós são hubs
        decoder_solution: é uma lista de tuplas proveniente do decoder
        n: é um inteiro que representa o total de nós do grafo.
        """
        result = []
        for node in range(1, n + 1):
            if node in list_of_hubs:
                result.append(node)
            else:
                result.append(Decoder.find_item_in_tuple(node, decoder_solution))

        for teste in result:
            if teste == None:
                print('oi')
        return result
    
    @staticmethod
     
    def create_binary_matrix_of_hubs(
    num_of_hubs: int, list_of_hubs_nodes: list, list_of_nodes_in_tree: list
) -> np.array:
        """
        Função que cria uma matriz binária para as arestas da árvore dos hubs, onde o arco (r,s) está na árvore, o valor na matriz é 1. 0, caso contrário.
        Importante: a primeira linha e primeira coluna da matriz é reservado para os nós hubs. Portanto, a matriz binária será A[1:, 1:].
        Parâmetros:
        num_of_hubs: é um inteiro para representar o número de hubs
        list_of_hubs_nodes: é uma lista que contém todos os nós hubs
        list_of_nodes_in_tree: é uma lista que contém todas as arestas de hubs que formam a árvore

        """
        list_of_hubs_nodes = sorted(list_of_hubs_nodes)
        list_of_nodes_in_tree += [(tup[1], tup[0]) for tup in list_of_nodes_in_tree]

        if len(list_of_hubs_nodes) == num_of_hubs:
            list_of_hubs_nodes.insert(0, 0)
        # Matriz com valores inf com shape pxp
        A = np.full((num_of_hubs + 1, num_of_hubs + 1), np.inf)
        A[0, :] = list_of_hubs_nodes
        A[:, 0] = list_of_hubs_nodes

        for i in range(1, len(A[0])):
            for j in range(1, len(A[1])):
                if i == j:
                    A[i][j] = 0
                else:
                    if (int(A[i][0]), int(A[0][j])) in list_of_nodes_in_tree:
                        A[i][j] = 1
                    else:
                        A[i][j] = 0

        #matrix_binary = np.triu(A) + np.triu(A, k=1).T
        return A
    
    
    @staticmethod
     
    def all_edges_in_solution(solution_list: list) -> list:
        """
        Função que concatena as tuplas de nós do grafo. Isso é necessário, pois não é um grafo orientado, todavia (1,5) e (5,1) possuem mesmo fluxo e custo.
        Parâmetros:
        solution_list: é uma lista da solução das arestas gerada por meio do decoder
        """
        reversed_solution = [(node[1], node[0]) for node in solution_list]
        return sorted(solution_list + reversed_solution)
        
    
    @staticmethod
     
    def find_index(value_to_find: int, list_in_table_binary: list)-> int:
        '''
        Função que encontra o index para mapear os hubs em cT
        Parâmetros:
        value_to_find --> o valor que deseja encontra o index
        list_in_table_binary --> uma lista que atribui um contador nas posição dos hubs
        '''
        if value_to_find in list_in_table_binary:
            return list_in_table_binary.index(value_to_find)

    @staticmethod
     
    def create_matrix_for_fitness(npoints:int, phubs: int, list_hubs: list, tree:list, A_cost: np.array)-> np.array:
        '''
            Função que calcula a matriz de referência para otimizar o cálculo da função-objetivo.
            Parâmetros:
            npoints --> número de nós da instância
            phubs --> número de hubs
            list_hubs --> lista contendo os nós hubs
            tree --> é a árvore de hubs
            A_cost --> matriz de custo entre todos os nós da instância
            '''
        cT = np.full((phubs,phubs), np.inf)
        matrix_ref = Decoder.create_binary_matrix_of_hubs(phubs, list_hubs, tree) # Cria uma matriz binária hub por hub. onde 1 é se existe a aresta entre hubs. 0, caso contrário
        matrix_ref = matrix_ref[1:,1:]

        h = [None] * npoints  # Create a list of length n with None as placeholders
        counter = 0
        for i in range(npoints):
            if i in np.array(list_hubs)-1:
                h[i] = counter 
                counter += 1
        
        for i in range(phubs):
            for j in range(phubs):
                if i != j:
                    if matrix_ref[i][j] > 0:
                        cT[i][j] = A_cost[Decoder.find_index(i, h)][Decoder.find_index(j, h)]
                    else: 
                        pass
                else:
                    cT[i][j] = 0

        for k in range(phubs):
            for i in range(phubs):
                for j in range(phubs):
                    if cT[i][k] + cT[k][j] < cT[i][j]:
                        cT[i][j] = cT[i][k] + cT[k][j]

        return cT,h

    @staticmethod
     
    def fitness_function(n: int, alpha: float, assig: list, h:list, A_cost: np.array, A_flows:np.array, A_fitness: np.array)-> float:
        '''
        Função objetivo do THLP, descrito no artigo de Pessoa et al (2017).
        Parâmetros:
        n         -> número de vértices da instância
        assig     -> vetor que representa a solução
        A_cost    -> matriz de custo
        A_flows   -> matriz de fluxo
        A_fitness -> matriz que otimiza o cálculo do custo entre hubs.
        '''
        z = 0
        for i in range(n):
            for j in range(n):
                custo = A_cost[i][assig[i]-1] + A_cost[assig[j]-1][j] + alpha * (A_fitness[h[assig[i]-1]][h[assig[j]-1]])
                z += custo * A_flows[i][j]
        return z 
    
    @staticmethod
     
    def decoder(chromossome: list, n:int, p:int, alpha: float, matrix_of_costs: np.array, matrix_of_flows: np.array)-> float:
        '''
        Decoder para as chaves aleatórias com um cromossomo com N elementos, onde N = (n) + (n-p) + (p(p-1)/2) elementos.
        Parâmetros:
            chromossome --> cromossomo com chaves aleatórias
            n ------------> número de nó da instância
            p ------------> número de hubs
            alpha --------> fator de desconto dos fluxos.
        '''

        hubs_nodes, non_hubs_nodes = Decoder.define_hubs(chromossome, n, p)
        #print("-" * 40)
        # Arcos de não-hubs para hubs
        arcs_hub = Decoder.assign_non_hubs_to_hubs(chromossome, n, p, hubs_nodes)
        #print("-" * 40)
        tree_of_hubs = Decoder.arcs_hubs_for_building_tree(chromossome, hubs_nodes, p)
        #print("-" * 40)
        solution = Decoder.chromosome_solution(arcs_hub, tree_of_hubs)

        matrix_for_fitness, h = Decoder.create_matrix_for_fitness(n, p, hubs_nodes, tree_of_hubs, matrix_of_costs)
        representation = Decoder.building_representation_of_solution(hubs_nodes, solution, n)
        
        fitness_value = Decoder.fitness_function(n=n, alpha=alpha, assig=representation, h=h, A_cost=matrix_of_costs, A_flows=matrix_of_flows, A_fitness=matrix_for_fitness)
        return fitness_value, solution, representation
    
    @staticmethod
     
    def greedy_fitness_function(n:int, p:int, hubs_nodes:list ,tree:list, A_cost ,A_flows, alpha:float, sol: list):
        matrix_for_fitness, h = Decoder.create_matrix_for_fitness(n, p, hubs_nodes, tree, A_cost)
        representation = Decoder.building_representation_of_solution(hubs_nodes, sol, n)
        
        fitness_value = Decoder.fitness_function(n=n, alpha=alpha, assig=representation, h=h, A_cost=A_cost, A_flows=A_flows, A_fitness=matrix_for_fitness)

        return fitness_value, representation
        