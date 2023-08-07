from myusefuls import Useful
from time import perf_counter
import datetime

from simulated_annealing import SimulatedAnnealing
from iterated_local_search import ILS
from variable_neighborhood_search import VNS

path_instances = "//Users/arthurdantasmangussi/@Mestrado ITA/PO-205/Instâncias"
results = []

for i in range(2):
    all_instances = Useful.RunScenario()
    name = all_instances[i][0]
    runs = all_instances[i][1]
    optimal = all_instances[i][2]

    n, p, alpha_discount, N, matrix_cost, matrix_flows = Useful.read_data(
        path_instances, name
    )

    # Parâmetros iniciais da metaheurística

    # T_initial = 10000000
    # alpha = 0.99
    # max_iter = 50
    # T_stop = 0.0001

    betaMin = 0.20
    betaMax = 0.20

    # Execução da metaheurística
    run = 0
    while run < runs:
        stop_time = datetime.datetime.now() + datetime.timedelta(seconds=3000)
        initial_time = perf_counter()  # tempo inicial de execução

        # Metaheurística escolhida
        # MH = SimulatedAnnealing(n= n, p=p, alpha_discount=alpha_discount, matrix_cost=matrix_cost, matrix_flows=matrix_flows)
        # best_value, best_solution = SimulatedAnnealing.Run(MH, T_initial=T_initial, T_stop=T_stop, max_iter=max_iter, alpha=alpha, optimal=optimal, stop_time = stop_time)

        MH = VNS(
            n=n,
            p=p,
            alpha_discount=alpha_discount,
            matrix_cost=matrix_cost,
            matrix_flows=matrix_flows,
        )
        best_value, best_solution = VNS.Run(MH, betaMin, optimal)

        end_time = perf_counter()  # tempo final de execução

        t_algorithm = end_time - initial_time
        print("Instância: ", name)
        print("Tempo de execução: ", t_algorithm)

        results.append([name, best_value, best_solution, t_algorithm])

        run += 1

file = "resultadosVNS"
Useful.SaveResults(results, file)
