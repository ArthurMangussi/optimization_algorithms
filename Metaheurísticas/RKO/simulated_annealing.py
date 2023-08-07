"""
Simulated Annealing for Tree Hub Location Problem
"""
import random
import numpy as np
from decoder import Decoder
import datetime
from local_search import Perturbation

class SimulatedAnnealing:
    """
    Metaheurística Simulated Annealing para Tree Hub Location Problem 
    """

    def __init__(self, n, p, alpha_discount, matrix_cost, matrix_flows) -> None:
        self.n = n
        self.p = p
        self.alpha_discount = alpha_discount
        self.matrix_cost = matrix_cost
        self.matrix_flows = matrix_flows

    def Run(self, T_initial, T_stop, max_iter, alpha, optimal, stop_time):
        """
        Parâmetros:
        T_inital --> temperatura inicial
        T_stop ----> temperatura mínima para parada do algoritmo
        max_iter --> número máximo de interações da metaheurística
        alpha -----> fator de resfriamento
        optimal ---> valor ótimo para a instância 
        """
        n = self.n
        p = self.p
        alpha_discount = self.alpha_discount
        matrix_cost = self.matrix_cost
        matrix_flows = self.matrix_flows

        sBest_value = 99999999999

        for i in range(1000):

            # Random initial chromossome 
            cromosso_aleatorio = [
                random.random() for _ in range(int((n + (n - p) + ((p * (p - 1)) / 2))))
            ]
            fitnesse_value, solution, assigment = Decoder.decoder(
                cromosso_aleatorio, n, p, alpha_discount, matrix_cost, matrix_flows
            )

            if fitnesse_value < sBest_value:
                sBest_value = fitnesse_value
                sBest_solution = solution
                sBest_assigment = assigment
                best_chromossome = cromosso_aleatorio

        # print('Melhor função-objetivo da população inicial = ', sBest_value)
        #print('Melhor Solução', sBest_solution)
        # print('Representação da melhor solução', sBest_assigment)
        # print('Melhor cromossomo', best_chromossome)

        temperature = T_initial

        while temperature > T_stop:
            iterT = 0
            while iterT < max_iter:

                # Shake the current solution
                intensity = int(len(best_chromossome) * 0.05)
                s_neighboor = Perturbation.shake_solution(intensity, best_chromossome)

                (
                    fitnesse_value_neighboor,
                    solution_neighboor,
                    assigment_neighboor,
                ) = Decoder.decoder(
                    s_neighboor, n, p, alpha_discount, matrix_cost, matrix_flows
                )

                delta = fitnesse_value_neighboor - sBest_value

                if delta < 0:
                    sBest_value = fitnesse_value_neighboor
                    sBest_solution = solution_neighboor
                    sBest_assigment = assigment_neighboor
                    best_chromossome = s_neighboor
                    # print(f'Found best solution at T = {temperature} --> FO = {sBest_value}')

                else:
                    x = random.random()
                    if x < np.exp(-delta / temperature):
                        sBest_value = fitnesse_value_neighboor
                        sBest_solution = solution_neighboor
                        sBest_assigment = assigment_neighboor
                        best_chromossome = s_neighboor

                iterT += 1
            temperature *= alpha
            print(f"\nT = {temperature:.4f}    FO = {sBest_value:.1f}")

            
            if int(sBest_value) <= int(optimal) or datetime.datetime.now() >= stop_time:
                break

        print("Melhor valor de função-objetivo = ", sBest_value)
        print("Melhor solução = ", sBest_solution)
        return sBest_value, sBest_solution
