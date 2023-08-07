"Variable Neighborhood Search"

from encoder import Encoder
import numpy as np
import random
from decoder import Decoder 
from myusefuls import Useful
from greedy_heuristic import GreedyAlgorithm
from local_search import LocalSearch, Perturbation

class VNS:
    "Variable"
    
    def __init__(self, n, p, alpha_discount, matrix_cost, matrix_flows) -> None:
        self.n = n
        self.p = p
        self.alpha_discount = alpha_discount
        self.matrix_cost = matrix_cost
        self.matrix_flows = matrix_flows

    # ----------------------------------------------------------------------------------------------
    def Run(self, beta_min: float, optimal: float):

        global sBest_value, sBest_solution, sBest_assigment, best_chromossome

        temp_ofv = 99999999999
        temp_sol = []

        # Run the search process until stop criterion
        ITER, iterMelhora, n_iter = 0, 0, 0

        # Create initial solution with random keys
        sBest_value = 99999999999

        for _ in range(1000):

            # Construtive Heuristic
            hubs, tree_of_hubs, solution, representation_tree = GreedyAlgorithm.greedy_solution_thlp(self.n,self.p,self.matrix_cost)
            fo, assig = Decoder.greedy_fitness_function(self.n,self.p,hubs,tree_of_hubs, self.matrix_cost,self.matrix_flows, self.alpha_discount, solution)

            # Back to random keys
            obj_encoder = obj_encoder = Encoder(hubs,assig,representation_tree, self.n, self.p, self.alpha_discount, self.matrix_cost, self.matrix_flows)

            chromossome_encoder = obj_encoder.get_complete_chromosome()

            # Decoder the current solution
            fo_decoder, solution, repre = Decoder.decoder(chromossome_encoder, self.n, self.p, self.alpha_discount, self.matrix_cost, self.matrix_flows)

            VNS.update_best_solution(fo_decoder, solution, repre, chromossome_encoder)

        print(f"Iter = {ITER} \t melhorFO = {sBest_value}")

        # Parâmetros da VNS
        k = 1
        r = 5
        max_iter = 1000

        while n_iter < max_iter:

            k = 1
            while k <= r:
                ITER += 1
                
                beta = np.random.uniform(k* beta_min, (k+1)*beta_min)
                intensity = int(self.n * beta)

                # Gera uma solução vizinha s' considerando a k-ésima estrutura de vizinhança
                neighboor = Perturbation.shake_solution(intensity, best_chromossome)

                # Calcular FO da solução vizinha
                fitnesse_value_neighboor,solution_neighboor,assigment_neighboor = Decoder.decoder(neighboor, self.n, self.p,self.alpha_discount,self.matrix_cost, self.matrix_flows)

                # Busca local em s' para econtrar s"
                search = LocalSearch(self.p, self.n, self.alpha_discount, self.matrix_cost, self.matrix_flows)
                new_best_value, new_best_solution, new_best_assigment, new_best_chromossome = search.LS(fitnesse_value_neighboor, neighboor, solution_neighboor, assigment_neighboor)

                
                delta = new_best_value - sBest_value

                if delta < 0 :
                    k = 1
                    iterMelhora = ITER
                    VNS.update_best_solution(new_best_value, new_best_solution, new_best_assigment, new_best_chromossome)
                
                else: 
                    k += 1 # próxima estrutura de vizinhança

                
                if ITER - iterMelhora > 1000:
                    if new_best_value < temp_ofv:
                        temp_ofv = new_best_value
                        temp_sol = new_best_solution.copy()
                    random_chromossome = [
                            random.random()
                            for _ in range(
                                int(
                                    (self.n + (self.n - self.p) + ((self.p * (self.p - 1)) / 2))
                                )
                            )
                        ]
                    
                    (random_chromossome_fo,
                    random_chromossome_solution,
                    random_chromossome_repre,
                    ) = Decoder.decoder(
                            random_chromossome,
                            self.n,
                            self.p,
                            self.alpha_discount,
                            self.matrix_cost,
                            self.matrix_flows,)
                    
                    # Update the best solution found to random solution
                    sBest_value = random_chromossome_fo
                    sBest_solution = random_chromossome_solution.copy()
                    sBest_assigment = random_chromossome_repre.copy()
                    best_chromossome = random_chromossome.copy()
                    iterMelhora = ITER

                if int(sBest_value) <= int(optimal):
                    n_iter = max_iter
                    temp_ofv = sBest_value
                    temp_sol = sBest_solution.copy()
                    break

            print(f"Iter = {n_iter} \t FO = {fitnesse_value_neighboor:.2f} \t FOVizinho = {new_best_value:.2f} \t melhorFO = {sBest_value:.2f}")
            n_iter += 1
        return temp_ofv, temp_sol

    # ----------------------------------------------------------------------------------------------
    def update_best_solution(current_ofv:float, current_solution:list, current_assigment:list, current_chromossome:list):
        global sBest_value, sBest_solution, sBest_assigment, best_chromossome

        # save the best solution found
        if current_ofv < sBest_value:
            sBest_value = current_ofv
            sBest_solution = current_solution.copy()
            sBest_assigment = current_assigment.copy()
            best_chromossome = current_chromossome.copy()