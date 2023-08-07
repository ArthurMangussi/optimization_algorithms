'''
Modelo exato para o problema THLP.
'''
from pulp import *
from Decoder import Decoder

path_instances = "C:\\Users\Admin\Desktop\@Mestrado ITA\PO-205\Instâncias"

ap_10 = os.listdir(path_instances)[0]
ap_20 = os.listdir(path_instances)[1]
ap_25 = os.listdir(path_instances)[2]

n, matrix_cost, matrix_flows = Decoder.generate_matrix_of_distances(path_instances, ap_10)
p = 3


problem = LpProblem('THLP', LpMinimize)


x = LpVariable.dicts("x", [(i, j) for i in range(1, n + 1) for j in range(1, n + 1)], cat = 'Binary')
y = LpVariable.dicts("y", [(k, i, j) for k in range(1, n + 1) for i in range(1, n + 1) for j in range(1, n + 1)], cat = 'Continuous', lowBound = 0)
z = LpVariable.dicts("z", [(i, j) for i in range(1, n + 1) for j in range(1, n + 1)], cat = 'Binary')

problem += lpSum((matrix_cost[i - 1, j - 1] * sum(matrix_flows[i - 1, :]) + matrix_cost[j - 1, i - 1] * sum(matrix_flows[:, i - 1])) * z[(i, j)] for i in range(1, n + 1) for j in range(1, n + 1)) + \
           lpSum( 0.5 * matrix_cost[i - 1, j - 1] * y[(k, i, j)] for k in range(1, n + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i != j)

# Restrição em (4)
problem += LpConstraint(e = lpSum(z[(i, i)] for i in range(1, n + 1)) - p, sense = LpConstraintEQ, rhs = 0)

# Restrição em (5)
for i in range(1, n + 1):
    problem += LpConstraint(e = lpSum(z[(i, j)] for j in range(1, n + 1)), sense = LpConstraintEQ, rhs = 1)

# Restrições (6) e (7)
for i in range(1, n + 1):
    for j in range(1, n + 1):
        if i < j:
            problem += LpConstraint(e = z[(i, j)] + x[(i, j)] - z[(j, j)], sense = LpConstraintLE, rhs = 0)
            problem += LpConstraint(e = z[(j, i)] + x[(i, j)] - z[(i, i)], sense = LpConstraintLE, rhs = 0)

# Restrição (8)
for k in range(1, n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1): 
            if i < j:
                problem += LpConstraint(e = y[(k, i, j)] + y[(k, j ,i)] - sum(matrix_flows[k - 1, :]) * x[(i, j)], sense = LpConstraintLE, rhs = 0)

# Restrição (9)
for k in range(1, n + 1):
    for j in range(1, n + 1):
        if k != j:
            problem += LpConstraint(e = sum(matrix_flows[k - 1, :]) * z[(k, j)] + lpSum(y[(k, i, j)] for i in range(1, n + 1) if i != j) - lpSum(y[(k, j, i)] for i in range(1, n + 1) if i != j) - lpSum(matrix_flows[k - 1, i - 1] * z[(i, j)] for i in range(1, n + 1)), rhs = 0 )

# Restrição (10)
problem += LpConstraint(e = lpSum(x[(i, j)] for i in range(1, n + 1) for j in range(1, n + 1)) + 1 - p, rhs = 0)

 # Resolve o problema
problem.solve()

if LpSolutionOptimal == problem.sol_status:
    solution_status = 'Ótima'
elif LpSolutionIntegerFeasible == problem.sol_status:
    solution_status = 'Factível'

print(solution_status)
print(f"Função objetivo: {problem.objective.value()}")

A = []

for i in range(1, n + 1):
    for j in range(1, n + 1):
        if 0 != value(x[(i, j)]):
            A.append((i, j))
            print(f"x_{i, j} = {value(x[(i, j)])}")

for k in range(1, n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if value(y[(k, i, j)]) != None and value(y[(k, i, j)]) != 0:
                    print(f"y_{k, i, j} = {value(y[(k, i, j)])}")

for i in range(1, n + 1):
    for j in range(1, n + 1):
        if value(z[(i, j)]) == 1:
            if i != j:
                A.append((i, j))
            print(f"z_{i, j} = {value(z[(i, j)])}")

print(A)