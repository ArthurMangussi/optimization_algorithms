'''
Funções para construção de uma heurística gulosa para THLP.
'''
import sys
sys.path.append('//Users//arthurdantasmangussi//@Mestrado ITA//Códigos')
from kruskal_clean import GrafoKruskal

import random
import numpy as np

class GreedyAlgorithm():
    '''
    Heurística Gulosa
    '''
    @staticmethod
    def create_map(hubs:list, p:int)-> dict:
        '''
        Função que cria um dicionário para mapear os hubs para utilizar o Algoritmo de Kruskal.
        Parâmetros:
            hubs : é uma lista contendo os vértices hubs
        '''
        iterator = 0
        mapa = {}
        while iterator < p:
            mapa.update({ hubs[iterator] : iterator})
            iterator += 1

        return mapa
    @staticmethod
    def change_for_kruskal(mdict: dict, graph: list)-> list:
        '''
        Função que altera os valores do nós hubs, para uma lista sequencial. Esse passo é necessário 
        para evitar o erro "list out of range" no Algoritmo de Kruskal.
        Parâmetros: 
        mdict = dicionário contendo os hubs e valores ordenados
        graph = grafo entre hubs.
        '''
        return [[mdict.get(key, key) if isinstance(key, int) else key for key in sublist] for sublist in graph]

    @staticmethod
    def reverse_dict(input_dict):
        """
        Reverses keys with values in a dictionary.

        Args:
            input_dict (dict): The input dictionary to reverse.

        Returns:
            dict: The reversed dictionary with keys and values swapped.
        """
        # Create an empty dictionary to store the reversed key-value pairs
        reversed_dict = {}

        # Iterate through the input dictionary
        for key, value in input_dict.items():
            # Swap the key and value in the reversed dictionary
            reversed_dict[value] = key

        return reversed_dict

    @staticmethod
    def reverse_solution(mdict: dict, graph:list)->list:
        return [[GreedyAlgorithm.reverse_dict(mdict).get(value, value) if isinstance(value, int) else value for value in sublist] for sublist in graph]
    
    @staticmethod
    def add_iterator_graph(graph:list):
        mapped = []
        iterator = 0
        for i in graph:
            mapped.append((i, iterator))
            iterator += 1
        return mapped
    
    @staticmethod
    def greedy_solution_thlp(n: int, p:int, A_cost: np.array)-> list:
        '''
        Heurística Construtiva para o Tree Hub Location Problem. A heurística escolhe aleatoriamente p hubs. Depois, cria uma árvore geradora mínima 
        entre hubs por meio do Algoritmo de Kruskal. Por fim, atribui o não-hub ao hub de menor custo.

        Parâmetros:
        n -------> número de nós das instâncias
        p -------> número de hubs da instância
        A_cost --> matriz de custo 
        '''

        #Escolher hubs aleatoriamente
        hubs = random.sample(range(1, n), p)
        #print(f'Hubs escolhidos = {hubs}')

        all_edges = []
        for i in range(len(hubs)):
            for j in range(i + 1, len(hubs)):
                all_edges.append((hubs[i], hubs[j]))

        hubs_mapped = GreedyAlgorithm.create_map(hubs,p)

        g = GrafoKruskal(p) # inicializando o grafo com p vértices

        for edge in range(len(all_edges)):
            g.adiciona_borda(all_edges[edge][0], all_edges[edge][1], A_cost[all_edges[edge][0]-1][all_edges[edge][1]-1]) # cria aresta entre hubs
            
        # Unir o último vértice com o primeiro
        #g.adiciona_borda(hubs[-1], hubs[0], A_cost[hubs[-1]-1][hubs[0]-1])


        # Salvar  posição 
        my_list = GreedyAlgorithm.add_iterator_graph(g.grafo)
        my_list = sorted(my_list, key = lambda x : x[0][2])
        representa = [x[1] for x in my_list]
            
        graph_mapped = GreedyAlgorithm.change_for_kruskal(hubs_mapped, g.grafo) 

        g.grafo = graph_mapped

        retorno = g.kruskal_algoritmo() # Algoritmo de Kruskal

        tree = GreedyAlgorithm.reverse_solution(hubs_mapped, retorno)
        #print('Tree = ', tree)

        non_hubs = [non_hub for non_hub in range(1,n+1) if non_hub not in hubs] # todos nos nós não hubs que precisam ser alocados com hubs

        list_of_all_options = []
        for k in range(len(non_hubs)):
            for j in range(len(hubs)):
                #print(f'Hub = {hubs[j]} --> Não hub = {non_hubs[k]}; custo = {matrix_cost[hubs[j]-1][non_hubs[k]-1]}')
                list_of_all_options.append((hubs[j], non_hubs[k], A_cost[hubs[j]-1][non_hubs[k]-1])) # opções de arestas entre não-hub e hub

        iterator = 0
        edges = []

        while iterator < len(non_hubs):
            step = list_of_all_options[p*iterator : p*(iterator+1)] # para p hubs, há p opções de arestas
            options = sorted(step, key= lambda x : x[2])
            edges.append(options[0]) # menor custo determinará qual hub o não-hub será alocado

            iterator += 1
            

        sol = [(nodes[0], nodes[1]) for nodes in edges + tree]

        tree = [(tup[0], tup[1]) for tup in tree]
        return hubs, tree, sorted(sol), representa
    
    