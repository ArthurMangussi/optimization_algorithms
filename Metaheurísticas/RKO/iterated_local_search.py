"Metaheurística Iterated Local Search (ILS)"

from encoder import Encoder
import numpy as np
import random
from decoder import Decoder
from greedy_heuristic import GreedyAlgorithm
from local_search import LocalSearch, Perturbation
import datetime


class ILS:
    """"""

    def __init__(self, n, p, alpha_discount, matrix_cost, matrix_flows) -> None:
        self.n = n
        self.p = p
        self.alpha_discount = alpha_discount
        self.matrix_cost = matrix_cost
        self.matrix_flows = matrix_flows

    # ----------------------------------------------------------------------------------------------
    def Run(self, beta_min: float, beta_max: float, optimal: float, stop_time):
        """
        Metaheurística Busca Local Iterativa
        """
        global new_best_value, new_best_solution, new_best_assigment, new_best_chromossome

        temp_ofv = 99999999999
        temp_sol = []

        # Run the search process until stop criterion
        ITER, iterMelhora = 0, 0

        # Create initial solution with random keys
        sBest_value = 99999999999

        for _ in range(1000):
            # Construtive Heuristic
            (
                hubs,
                tree_of_hubs,
                solution,
                representation_tree,
            ) = GreedyAlgorithm.greedy_solution_thlp(self.n, self.p, self.matrix_cost)
            fo, assig = Decoder.greedy_fitness_function(
                self.n,
                self.p,
                hubs,
                tree_of_hubs,
                self.matrix_cost,
                self.matrix_flows,
                self.alpha_discount,
                solution,
            )

            # Back to random keys
            obj_encoder = Encoder(
                hubs,
                assig,
                representation_tree,
                self.n,
                self.p,
                self.alpha_discount,
                self.matrix_cost,
                self.matrix_flows,
            )

            chromossome_encoder = obj_encoder.get_complete_chromosome()

            # Decoder the current solution
            fo_decoder, solution, repre = Decoder.decoder(
                chromossome_encoder,
                self.n,
                self.p,
                self.alpha_discount,
                self.matrix_cost,
                self.matrix_flows,
            )

            if fo_decoder < sBest_value:
                sBest_value = fo_decoder
                sBest_solution = solution.copy()
                sBest_assigment = repre.copy()
                best_chromossome = chromossome_encoder.copy()

        search = LocalSearch(
            self.p, self.n, self.alpha_discount, self.matrix_cost, self.matrix_flows
        )

        # Apply Local Search
        (
            new_best_value,
            new_best_solution,
            new_best_assigment,
            new_best_chromossome,
        ) = search.LS(sBest_value, best_chromossome, sBest_solution, sBest_assigment)

        print(
            f"Iter = {ITER} \t FO = {new_best_value:.2f} \t melhorFO = {sBest_value:.2f}"
        )

        while True:
            ITER += 1

            beta = np.random.uniform(beta_min, beta_max)
            intensity = int(self.n * beta)

            # Shake the current solution
            s_neighboor = Perturbation.shake_solution(intensity, new_best_chromossome)

            (
                fitnesse_value_neighboor,
                solution_neighboor,
                assigment_neighboor,
            ) = Decoder.decoder(
                s_neighboor,
                self.n,
                self.p,
                self.alpha_discount,
                self.matrix_cost,
                self.matrix_flows,
            )

            # Busca local na solução perturbada
            (
                best_neighboor_value,
                best_neighboor_solution,
                best_neighboor_assigment,
                best_neighboor_chromossome,
            ) = search.LS(
                fitnesse_value_neighboor,
                s_neighboor,
                solution_neighboor,
                assigment_neighboor,
            )

            print(
                f"Iter = {ITER} \t FO = {fitnesse_value_neighboor:.2f} \t FoVizinho = {best_neighboor_value:.2f}\t MelhorFO = {new_best_value:.2f}"
            )

            delta = best_neighboor_value - new_best_value

            if delta < 0:
                ILS.update_best_solution(
                    best_neighboor_value,
                    best_neighboor_solution,
                    best_neighboor_assigment,
                    best_neighboor_chromossome,
                )
                iterMelhora = ITER

            if ITER - iterMelhora > 800:  # 800 soluções sem melhoria
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
                (
                    random_chromossome_fo,
                    random_chromossome_solution,
                    random_chromossome_repre,
                ) = Decoder.decoder(
                    random_chromossome,
                    self.n,
                    self.p,
                    self.alpha_discount,
                    self.matrix_cost,
                    self.matrix_flows,
                )
                iterMelhora = ITER

                # Update the best solution found to random solution
                new_best_value = random_chromossome_fo
                new_best_solution = random_chromossome_solution.copy()
                new_best_assigment = random_chromossome_repre.copy()
                new_best_chromossome = random_chromossome.copy()

            if int(new_best_value) <= int(optimal):
                temp_ofv = new_best_value
                temp_sol = new_best_solution.copy()
                break

            if datetime.datetime.now() >= stop_time:
                break

        # return new_best_value, new_best_solution
        return temp_ofv, temp_sol

    # ----------------------------------------------------------------------------------------------
    def update_best_solution(
        current_ofv: float,
        current_solution: list,
        current_assigment: list,
        current_chromossome: list,
    ):
        "Função que atualiza a melhor solução encontrada durante a execução do algoritmo"
        global new_best_value, new_best_solution, new_best_assigment, new_best_chromossome

        # save the best solution found
        if current_ofv < new_best_value:
            new_best_value = current_ofv
            new_best_solution = current_solution.copy()
            new_best_assigment = current_assigment.copy()
            new_best_chromossome = current_chromossome.copy()
