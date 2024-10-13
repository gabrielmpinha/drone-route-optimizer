from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.core.variable import Real, Integer
import numpy as np
from pymoo.visualization.scatter import Scatter

# Define 3D coordinates for cities
locacoes_cidades = {
    "Cidade1": (0, 0, 0),
    "Cidade2": (10, 10, 0),
    "Cidade3": (20, 20, 0),
    "Cidade4": (5, 5, 0),
    "Cidade5": (18, 18, 0),
    "Cidade6": (11, 11, 0),
    "Cidade7": (19, 19, 0),
    "Cidade8": (3, 3, 0),
    "Cidade9": (14, 14, 0),
    "Cidade10": (12, 12, 0),
    "Cidade11": (24, 1, 0),
    "Cidade12": (9, 9, 0),
}

initial_pos = (7, 7, 0)

# Lista de nomes de cidades para referência
nomes_cidades = list(locacoes_cidades.keys())

class DroneOptimizationProblem(Problem):
    
    def __init__(self):
        super().__init__(
            n_var=len(nomes_cidades) + 1,  # Número de variáveis: [cidade, aceleração]
            n_obj=2,  # Número de objetivos: [tempo de entrega, consumo de energia]
            n_constr=0,  # Sem restrições
            xl=[0] * len(nomes_cidades) + [0.1],  # Limites inferiores: cidades e aceleração
            xu=[len(nomes_cidades) - 1] * len(nomes_cidades) + [50],  # Limites superiores: cidades e aceleração
            type_var=np.array([Integer, Real]) # tipos das variaveis
        )

    def _evaluate(self, X, out, *args, **kwargs):
        aceleracao = X[:, -1]
        tempo_total = []
        for i in range(X.shape[0]):
            ordem_indices = np.argsort(X[i, :-1])
            ordem_cidades = [nomes_cidades[j] for j in ordem_indices]
            
            tempo = 0
            primeira_cidade = ordem_cidades[0]
            distancia_inicial = np.linalg.norm(np.array(locacoes_cidades[primeira_cidade]) - np.array(initial_pos))
            tempo += distancia_inicial / aceleracao[i]
            
            for k in range(1, len(ordem_cidades)):
                cidade_anterior = ordem_cidades[k - 1]
                cidade_atual = ordem_cidades[k]
                distancia = np.linalg.norm(np.array(locacoes_cidades[cidade_atual]) - np.array(locacoes_cidades[cidade_anterior])) 
                tempo += distancia / aceleracao[i]
            
            tempo_total.append(tempo)
        
            consumo_energia = aceleracao
        
        out["F"] = np.column_stack([tempo_total, consumo_energia])

# Configurar o algoritmo NSGA-II
algorithm = NSGA2(
    pop_size=200,
    eliminate_duplicates=True
)

# Resolver o problema
problem = DroneOptimizationProblem()
res = minimize(problem, algorithm, ('n_gen', 200), verbose=True)

# Exibir os resultados
print("Soluções Não-Dominadas:")
for i in range(len(res.X)):
    ordem_indices = np.argsort(res.X[i, :-1])
    ordem_cidades = [nomes_cidades[j] for j in ordem_indices]
    tempo = res.F[i, 0]
    energia = res.F[i, 1]
    aceleracao = res.X[i, -1]
    print(f"Ordem das cidades: {ordem_cidades}, Aceleração: {aceleracao:.2f}, Tempo: {tempo:.2f}, Energia: {energia:.2f}")

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="green", edgecolor="red")
plot.show()