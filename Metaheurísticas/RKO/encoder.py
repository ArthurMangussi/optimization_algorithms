"""
Classe de Encoder
"""
import numpy as np

from myusefuls import Useful


class Encoder:
    """Classe de Encoder"""

    def __init__(
        self,
        hubs: list,
        assig: list,
        representation_tree: list,
        n,
        p,
        alpha_discount,
        matrix_cost,
        matrix_flows
    ) -> None:
        self.hubs = hubs
        self.non_hubs = None
        self.assig = assig
        self.rho = None
        self.representation_tree = representation_tree
        self.n = n
        self.p = p
        self.alpha_discount = alpha_discount
        self.matrix_cost = matrix_cost
        self.matrix_flows = matrix_flows

    # ----------------------------------------------------------------------------------------------
    def find_interval(
        self,
        list_ranges: list,
        representation_solution: list,
        list_non_hubs: list,
        nhub: int,
    ):
        """
        Função que encontra o intervalo para gerar a chave aleatória, baseando-se no intervalo
        [0,1] em p partes e consultando qual hub, o nó não hub está alocado.
        Parâmetros:
        list_ranges: lista de tuplas, as quais são compostas por (hub, intervalo)
        representation_solution : é a representação da solução do THLP
        list_non_hubs: é uma lista com os nós não hubs
        nhub : é um inteiro que representa o nó não-hub
        """
        for interval in range(len(list_ranges)):
            if (
                representation_solution[list_non_hubs[nhub] - 1]
                in list_ranges[interval]
            ):
                lower = list_ranges[interval][1][0]
                upper = list_ranges[interval][1][1]
                lower = max(0, lower)
                upper = min(1, upper)
                rk = np.random.uniform(lower, upper)
        return rk

    # ----------------------------------------------------------------------------------------------
    def get_initial_chromosome(self) -> np.array:
        """Meteodo..."""

        self.non_hubs = [
            non_hub for non_hub in range(1, self.n + 1) if non_hub not in self.hubs
        ]
        nodes = self.hubs + self.non_hubs

        beta = 1 / self.n
        step = beta / 2

        c = np.arange(0, 1 + beta, beta)

        initial_chromossome = np.zeros(self.n)
        for i in range(len(c) - 1):
            lowerbound = (c[i] + c[i + 1]) / 2 - step  # limite inferior do intervalo
            upperbound = (c[i] + c[i + 1]) / 2 + step  # limite superior do intervalo
            lowerbound = max(0, lowerbound)
            upperbound = min(1, upperbound)
            random_key = np.random.uniform(
                lowerbound, upperbound
            )  # chave aleatória com base
            # no intervalo
            initial_chromossome[
                nodes[i] - 1
            ] = random_key  # Indexar pela posição na solução

        return initial_chromossome

    # ----------------------------------------------------------------------------------------------
    def get_middle_chromosome(self) -> list:
        """metodo"""
        self.rho = 1 / self.p

        ranges = []
        for k in range(self.p):
            lower = k * self.rho
            upper = (k + 1) * self.rho
            ranges.append([lower, upper])

        t = [(self.hubs[hub], ranges[hub]) for hub in range(len(self.hubs))]

        return [
            self.find_interval(t, self.assig, self.non_hubs, non_hub)
            for non_hub in range(len(self.non_hubs))
        ]

    # ----------------------------------------------------------------------------------------------
    def get_last_chromosome(self) -> np.array:
        """metodo"""
        gamma = round((1 / (self.p * (self.p - 1) / 2)),10)
        b = np.arange(0, 1 + gamma, gamma)
        
        #b = np.arange(0, 1 + rho, rho)
        
        last_choromossome = np.zeros(int(self.p * (self.p - 1) / 2))

        for i in range(len(b)-1):
            lowerbound = max(((b[i] + b[i+1])/2) - (self.rho /2),0) # limite inferior do intervalo

            upperbound = min((b[i] + b[i+1])/2 + self.rho /2, 1) # limite superior do intervalo

            random_key = round(np.random.uniform(lowerbound, upperbound),10) # chave aleatória com base no intervalo

            if i < len(self.representation_tree):
                last_choromossome[self.representation_tree[i]] = random_key # Indexar pela posição na solução

        return last_choromossome

    # ----------------------------------------------------------------------------------------------
    def get_complete_chromosome(self) -> list:
        """metodo"""
        return (
            list(self.get_initial_chromosome())
            + self.get_middle_chromosome()
            + list(self.get_last_chromosome())
        )
