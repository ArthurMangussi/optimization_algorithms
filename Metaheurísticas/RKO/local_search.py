"""
Busca Locais
"""

import numpy as np
from decoder import Decoder
import random 

class LocalSearch:
    """Busca local"""

    def __init__(
        self,
        nHubs: int,
        nodes: int,
        alpha_discount: float,
        A_cost: np.array,
        A_flows: np.array,
    ) -> None:
        self.nHubs = nHubs
        self.nodes = nodes
        self.alpha_discount = alpha_discount
        self.A_cost = A_cost
        self.A_flows = A_flows
        

    # ----------------------------------------------------------------------------------------------
    def change_hub(
        self,
        chromossome: list,
        current_fo: float,
        current_solution: list,
        current_assigment: list,
    ):
        """
        Busca Local nas chaves aleatórias. Tal busca altera um hub por um ponto não hub

        Parâmetros:
        nHubs --------------> número de Hubs
        nodes --------------> número de pontos da instância
        alpha_discount -----> valor alpha de desconto para o cálculo da FO
        A_cost -------------> matriz de custo
        A_flows ------------> matriz de fluxos
        neighboor ----------> uma lista contendo as chaves aleatórias do vizinho da solução corrente
        current_fo ---------> valor atual da função-objetivo
        current_solution ---> solução atual da instância
        current_assigment --> representação da solução atual
        """
        # Local Search - Troca Hub
        best_chromossome = chromossome.copy()

        for i in range(self.nHubs):
            for j in range(self.nHubs, self.nodes + 1):
                best_chromossome[i], best_chromossome[j] = best_chromossome[j], best_chromossome[i]

                (
                    fitnesse_value_neighboor,
                    solution_neighboor,
                    assigment_neighboor,
                ) = Decoder.decoder(
                    best_chromossome,
                    self.nodes,
                    self.nHubs,
                    self.alpha_discount,
                    self.A_cost,
                    self.A_flows,
                )

                # Atualiza a solução corrente se melhorou e continua a busca a partir do vizinho
                if fitnesse_value_neighboor < current_fo:
                    current_fo = fitnesse_value_neighboor
                    current_solution = solution_neighboor.copy()
                    current_assigment = assigment_neighboor.copy()
                    #print(f"FOUND NEW BEST FO = {current_fo} trocando {i} por {j}")
                    

                else:  # retorna para a solução corrente
                    best_chromossome[j], best_chromossome[i] = best_chromossome[i], best_chromossome[j]
                    

        return current_fo, current_solution, current_assigment, best_chromossome

    # ----------------------------------------------------------------------------------------------
    def change_tree(
        self,
        chromossome: list,
        current_fo: float,
        current_solution: list,
        current_assigment: list,
    ):
        """
        Busca Local nas chaves aleatórias. Tal busca altera um arco i por um arco j não selecionado na árvore. Importante: o Decoder verifica se não forma ciclos.

        Parâmetros:
        nHubs --------------> número de Hubs
        nodes --------------> número de pontos da instância
        alpha_discount -----> valor alpha de desconto para o cálculo da FO
        A_cost -------------> matriz de custo
        A_flows ------------> matriz de fluxos
        neighboor ----------> uma lista contendo as chaves aleatórias do vizinho da solução corrente
        current_fo ---------> valor atual da função-objetivo
        current_solution ---> solução atual da instância
        current_assigment --> representação da solução atual
        """
        start = self.nodes + (self.nodes - self.nHubs)
        end = start + self.nHubs

        best_chromossome = chromossome.copy()

        # Local Search - Troca Tree
        for i in range(start, end - 1):
            for j in range(end - 1, len(best_chromossome)):
                best_chromossome[i], best_chromossome[j] = best_chromossome[j], best_chromossome[i]

                (
                    fitnesse_value_neighboor,
                    solution_neighboor,
                    assigment_neighboor,
                ) = Decoder.decoder(
                    best_chromossome,
                    self.nodes,
                    self.nHubs,
                    self.alpha_discount,
                    self.A_cost,
                    self.A_flows,
                )
                

                # Atualiza a solução corrente se melhorou e continua a busca a partir do vizinho
                if fitnesse_value_neighboor < current_fo:
                    current_fo = fitnesse_value_neighboor
                    current_solution = solution_neighboor.copy()
                    current_assigment = assigment_neighboor.copy()
                    

                else:  # retorna para a solução corrente
                    best_chromossome[j], best_chromossome[i] = best_chromossome[i], best_chromossome[j]
                    

        return current_fo, current_solution, current_assigment, best_chromossome

    # ----------------------------------------------------------------------------------------------
    def change_assigment(
        self,
        chromossome: list,
        current_fo: float,
        current_solution: list,
        current_assigment: list,
    ):
        """
        Busca Local nas chaves aleatórias. Tal busca altera a atribuição de cada ponto não-hub ao hub.

        Parâmetros:
        nHubs --------------> número de Hubs
        nodes --------------> número de pontos da instância
        alpha_discount -----> valor alpha de desconto para o cálculo da FO
        A_cost -------------> matriz de custo
        A_flows ------------> matriz de fluxos
        neighboor ----------> uma lista contendo as chaves aleatórias do vizinho da solução corrente
        current_fo ---------> valor atual da função-objetivo
        current_solution ---> solução atual da instância
        current_assigment --> representação da solução atual
        """

        best_chromossome = chromossome.copy()

        # Definir o fator
        factor = 1.0 / self.nHubs

        for i in range(
            self.nodes, self.nodes + (self.nodes - self.nHubs)
        ):  # troca a alocação de cada ponto não hub
            for j in range(self.nHubs):  # gera uma nova chave aleatória no intervalo j
                lowerbound = j * factor
                upperbound = (j + 1) * factor
                upperbound = min(1, upperbound)
                lowerbound = max(0, lowerbound)
                random_key = np.random.uniform(lowerbound, upperbound)
                chromossome[i] = random_key

                (
                    fitnesse_value_neighboor,
                    solution_neighboor,
                    assigment_neighboor,
                ) = Decoder.decoder(
                    best_chromossome,
                    self.nodes,
                    self.nHubs,
                    self.alpha_discount,
                    self.A_cost,
                    self.A_flows,
                )
                

                if fitnesse_value_neighboor < current_fo:
                    current_fo = fitnesse_value_neighboor
                    current_solution = solution_neighboor.copy()
                    current_assigment = assigment_neighboor.copy()
                    

                else:  # retorna para a solução corrente
                    best_chromossome[j], best_chromossome[i] = best_chromossome[i], best_chromossome[j]
                    

        return current_fo, current_solution, current_assigment, best_chromossome
    
    # ----------------------------------------------------------------------------------------------
    def LS(self,current_fo_value: float, current_chromossome:list, current_solution:list, current_assigment:list):
        '''
        Buscal Local nas chaves aleatórias.

        Parâmetros:
        current_fo_value -----> valor atual da função-objetivo 
        current_chromossome --> cromossomo atual contendo chaves aleatórias
        current_solution -----> solução atual da instância
        current_assigment ----> representação atual da solução
        '''
        
        number_of_local_search = 3

        # predefined number of neighborhood moves
        options_neighborhood = [i for i in range(number_of_local_search)]
        options_neighborhood_assistant = [i for i in range(number_of_local_search)]

        while options_neighborhood:

            # Current objective function value
            current_fo = current_fo_value
            chromossome_initial = current_chromossome.copy()

            # Randomly choose a neighboorhood
            pos = random.randint(0, len(options_neighborhood)-1)
            k = options_neighborhood[pos]
            
            match k:
                case 0:
                    fo_ls, ls_solution, ls_assigment, ls_chromossome = self.change_hub(chromossome_initial, current_fo_value, current_solution, current_assigment)
                     
                case 1:
                    fo_ls, ls_solution, ls_assigment, ls_chromossome = self.change_assigment(chromossome_initial, current_fo_value, current_solution, current_assigment)
                     
                case 2:
                    fo_ls, ls_solution, ls_assigment, ls_chromossome = self.change_tree(chromossome_initial, current_fo_value, current_solution, current_assigment)
                     

            if fo_ls < current_fo:
                options_neighborhood = options_neighborhood_assistant[:].copy()
                current_fo_value = fo_ls

            else:
                options_neighborhood.pop(pos)

        #ls_chromossome = [min(1, estouro)  for estouro in ls_chromossome if estouro > 1]
        return fo_ls, ls_solution, ls_assigment, ls_chromossome
            
class Perturbation:
    """
    Perturbação da solução corrente
    """
    @staticmethod
    def shake_solution(intensity, chromossome)-> list:
        """
        Função que perturba a soluçnao corrente, gerando um vizinho
        """       
        # Shake the current solution
        s_neighboor = chromossome.copy()

        for _ in range(intensity):
            shaking_type = random.randint(1, 4)
            # print(shaking_type)
            i = random.randint(0, len(s_neighboor) - 1)

            if shaking_type == 1:
                s_neighboor[i] = 1 - s_neighboor[i]  # invert value
                # print(f'alterou a posição {i+1} --> {s_neighboor}')

            elif shaking_type == 2:
                # Swap two positions
                j = random.randint(0, len(s_neighboor) - 1)
                s_neighboor[i], s_neighboor[j] = s_neighboor[j], s_neighboor[i]
                # print(f'alterou {s_neighboor[i]} por {s_neighboor[j]} --> {s_neighboor}')

            elif shaking_type == 3:
                # Change to random value
                s_neighboor[i] = random.random()
                # print(f'{s_neighboor}')

            elif shaking_type == 4:
                # Swap with neighbor
                i = random.randint(0, len(s_neighboor) - 2)
                s_neighboor[i], s_neighboor[i + 1] = s_neighboor[i + 1], s_neighboor[i]
                # print(s_neighboor)

        return s_neighboor