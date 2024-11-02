from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.core.crossover import Crossover
import math

class PMXCrossover(Crossover):

    def __init__(self, prob=1.0):
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        n_matings, n_parents, n_var = X.shape
        Y = np.full_like(X, 0, dtype=X.dtype)

        for p in range(n_parents):
                if np.random.random() < self.prob:
                    if n_parents > 1:
                        parent1, parent2 = X[0, p, :-1], X[1, p, :-1]
                        accel1, accel2 = X[0, p, -1], X[1, p, -1]
                        child1, child2 = self.pmx(parent1, parent2)
                        Y[0, p, :-1], Y[1, p, :-1] = child1, child2
                        Y[0, p, -1], Y[1, p, -1] = accel1, accel2
                    else:
                        parent1, parent2 = X[0, p, :-1], X[1, p, :-1]
                        accel1, accel2 = X[0, p, -1], X[1, p, -1]
                        child1, child2 = self.pmx(parent1, parent2)
                        Y[0, p, :-1], Y[1, p, :-1] = child1, child2
                        Y[0, p, -1], Y[1, p, -1] = accel1, accel2
                else:
                    Y[0, p, :], Y[1, p, :] = X[0, p, :], X[1, p, :]
                    Y[0, p, -1], Y[1, p, -1] = X[0, p, -1], X[1, p, -1]

        return Y
       
    def pmx(self, parent1, parent2):
        size = len(parent1)
        cx_point1, cx_point2 = sorted(np.random.choice(range(size), 2, replace=False))
        child1, child2 = np.full(size, -1), np.full(size, -1)

        # Copy the segment between the crossover points
        child1[cx_point1:cx_point2+1] = parent1[cx_point1:cx_point2+1]
        child2[cx_point1:cx_point2+1] = parent2[cx_point1:cx_point2+1]

        # Create mappings
        mapping1 = {parent1[i]: parent2[i] for i in range(cx_point1, cx_point2 + 1)}
        mapping2 = {parent2[i]: parent1[i] for i in range(cx_point1, cx_point2 + 1)}

        # Fill the remaining positions
        self.fill_remaining(child1, parent2, mapping1, cx_point1, cx_point2)
        self.fill_remaining(child2, parent1, mapping2, cx_point1, cx_point2)

        return np.array(child1), np.array(child2)

    def fill_remaining(self, child, parent, mapping, cx_point1, cx_point2):
        size = len(parent)
        for i in range(size):
            if i >= cx_point1 and i <= cx_point2:
                continue
            gene = parent[i]
            while gene in mapping:
                gene = mapping[gene]
            child[i] = gene

class CustomPermutationRandomSampling(PermutationRandomSampling):

    def _do(self, problem, n_samples, **kwargs):
        n_var = problem.n_var - 1  # Exclude the last column (acceleration)
        samples = np.full((n_samples, problem.n_var), int)
        
        for i in range(n_samples):
            # Generate a permutation of city indices
            permutation = np.random.permutation(n_var)
            # Assign the permutation to the sample
            samples[i, :-1] = permutation
            # Assign a random acceleration value to the last position
            samples[i, -1] = np.random.randint(problem.xl[-1], problem.xu[-1])
        
        return samples

# Coordenadas das cidades
locacoes_cidades_a= {
    "Cidade1": (1, 15, 0),
    "Cidade2": (10, 0, 0),
    "Cidade3": (20, 1, 0),
    "Cidade4": (17, 18, 0),
    "Cidade5": (18, 17, 0),
    "Cidade6": (11, 9, 0),
    "Cidade7": (19, 19, 0),
    "Cidade8": (3, 1, 0),
    "Cidade9": (14, 27, 0),
    "Cidade10": (12, 2, 0),
    "Cidade11": (24, 1, 0),
    "Cidade12": (9, 20, 0),
    "Cidade13": (7, 7, 0),
    "Cidade14": (8, 7, 0),
}

locacoes_cidades = {
    "Cidade1": (5, 13, 0),
    "Cidade2": (28, 16, 0),
    "Cidade3": (14, 16, 0),
    "Cidade4": (28, 18, 0),
    "Cidade5": (28, 16, 0),
    "Cidade6": (27, 14, 0),
    "Cidade7": (16, 9, 0),
    "Cidade8": (25, 20, 0),
    "Cidade9": (28, 8, 0),
    "Cidade10": (14, 11, 0),
    "Cidade11": (27, 4, 0),
    "Cidade12": (24, 25, 0),
    "Cidade13": (16, 23, 0),
    "Cidade14": (1, 4, 0),
    "Cidade15": (6, 5, 0),
    "Cidade16": (19, 17, 0),
    "Cidade17": (11, 16, 0),
    "Cidade18": (25, 14, 0),
    "Cidade19": (1, 15, 0),
    "Cidade20": (5, 26, 0),
    "Cidade21": (4, 2, 0),
    "Cidade22": (24, 13, 0),
    "Cidade23": (4, 1, 0),
    "Cidade24": (27, 7, 0),
    "Cidade25": (30, 27, 0),
    "Cidade26": (26, 29, 0),
    "Cidade27": (1, 30, 0),
    "Cidade28": (5, 7, 0),
    "Cidade29": (10, 0, 0),
    "Cidade30": (21, 11, 0),
    "Cidade31": (1, 29, 0),
    "Cidade32": (19, 17, 0),
    "Cidade33": (15, 21, 0),
    "Cidade34": (14, 0, 0),
    "Cidade35": (9, 20, 0),
    "Cidade36": (11, 2, 0),
    "Cidade37": (11, 20, 0),
    "Cidade38": (9, 22, 0),
    "Cidade39": (9, 17, 0),
    "Cidade40": (4, 26, 0),
    "Cidade41": (29, 26, 0),
    "Cidade42": (8, 14, 0),
    "Cidade43": (16, 2, 0),
    "Cidade44": (14, 21, 0),
    "Cidade45": (13, 27, 0),
    "Cidade46": (10, 3, 0),
    "Cidade47": (6, 17, 0),
    "Cidade48": (14, 30, 0),
    "Cidade49": (24, 27, 0),
    "Cidade50": (10, 15, 0),
}

initial_pos = (7, 7, 0)

# Lista de nomes de cidades para referência
nomes_cidades = list(locacoes_cidades.keys())

altura_decolagem = 10  # Altura que o drone vai subir na transição

peso_drone = 1.5


def calcular_decolagem(massa, aceleracao, altura_voo, altura_cidade):
    g = 9.81  # Aceleração da gravidade em m/s^2

    # Altura total
    altura_total = altura_voo - altura_cidade

    aceleracao = (aceleracao/100) - g

    if aceleracao <= 0:
        raise ValueError("A aceleração resultante deve ser maior que zero.")

    # Tempo de subida MRUV (considerando aceleração constante)
    tempo_subida = ((2 * altura_total / aceleracao))** 0.5

    momento_inercia, velocidade_angular = rot_drone()
    energia_rotacional = ((momento_inercia * (velocidade_angular ** 2))/2)

    energia_gravitacional = massa * g * altura_total
    potencia_elevacao = energia_gravitacional / tempo_subida # potencia para elevar um drone a altura total em uma unidade de tempo
    
    # Energia gasta considerando velocidade angular constante
    energia_gasta = (energia_rotacional + potencia_elevacao) * tempo_subida

    return energia_gasta, tempo_subida

def calcular_pouso(massa, aceleracao, altura_voo, altura_cidade):
    g = 9.81  # Aceleração da gravidade em m/s^2

    # Altura total
    altura_total = altura_voo - altura_cidade

    # Aceleração durante a descida
    aceleracao = (aceleracao / 100) + g

    if aceleracao <= 0:
        raise ValueError("A aceleração resultante deve ser maior que zero.")

    # Tempo de descida MRUV (considerando aceleração constante)
    tempo_descida = ((2 * altura_total / aceleracao))** 0.5

    momento_inercia, velocidade_angular = rot_drone()
    energia_rotacional = ((momento_inercia * (velocidade_angular ** 2)) / 2)

    energia_gravitacional = massa * g * altura_total
    potencia_descida = energia_gravitacional / tempo_descida  # Potência para descer o drone a altura total em uma unidade de tempo
    
    # Energia gasta considerando velocidade angular constante
    energia_gasta = (energia_rotacional + potencia_descida) * tempo_descida

    return energia_gasta, tempo_descida

def calcular_deslocamento(massa, aceleracao, distancia):
    g = 9.81  # Aceleração da gravidade em m/s^2

    aceleracao = (aceleracao / 100) - g 

    distancia = distancia*1000

    if aceleracao <= 0:
        raise ValueError("A aceleração resultante deve ser maior que zero.")

    # Tempo de deslocamento (considerando aceleração constante)
    tempo_deslocamento = ((2 * distancia / aceleracao))** 0.5

    momento_inercia, velocidade_angular = rot_drone()
    energia_rotacional = ((momento_inercia * (velocidade_angular ** 2)) / 2)

    energia_gravitacional = massa * g * altura_decolagem 

    if (tempo_deslocamento != 0):
        potencia_deslocamento = energia_gravitacional / tempo_deslocamento  # Potência para manter o drone em voo contra a gravidade
    else:
        potencia_deslocamento = 0
    
    # Energia gasta considerando velocidade angular constante
    energia_gasta = (energia_rotacional + potencia_deslocamento) * tempo_deslocamento

    return energia_gasta, tempo_deslocamento


def rot_drone(massa_pa = 1, comprimento_pa = 0.5, rpm = 3000):
    # Cálculo do momento de inércia
    momento_inercia = (1/3) * massa_pa * (comprimento_pa ** 2)
    # Cálculo da velocidade angular
    velocidade_angular = (2 * math.pi * rpm) / 60
    return momento_inercia, momento_inercia

# Função para calcular a distância entre duas posições, inicialmente ignorando altura
def calcular_distancia(pos1 = (0,0,0), pos2 = (0,0,0)):
    posx = abs(pos1[0] - pos2[0])
    posy = abs(pos1[1] - pos2[1])
    distancia = math.sqrt(posx**2 + posy**2)
    return distancia

class DroneOptimizationProblem(Problem):
    
    def __init__(self):
        super().__init__(
            n_var=len(nomes_cidades) + 1,  # Número de variáveis: [cidade, aceleração]
            n_obj=2,  # Número de objetivos: [tempo de entrega, consumo de energia]
            xl=[0] * len(nomes_cidades) + [1050],  # Limites inferiores: cidades e aceleração
            xu=[len(nomes_cidades) - 1] * len(nomes_cidades) + [3000],  # Limites superiores: cidades e aceleração
            vtype=int,
            n_ieq_constr=1
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # X é a matriz de soluções, onde cada linha é uma solução e cada coluna é uma variável de decisão, a última coluna sendo a aceleração
        # Extrai a aceleraçao de todas as soluções e cria uma lista para colocar o tempo e energia de cada solução
        aceleracao = X[:, -1]
        tempo_total = []
        consumo_energia_total = []
        constraints_violadas = []
        
        for i in range(X.shape[0]):
            ordem_indices = X[i, :-1] # descodifica os valores encontrados na solução nos indices das cidades
            ordem_cidades = [nomes_cidades[j] for j in ordem_indices]
            
            tempo = 0
            consumo_energia = 0
            primeira_cidade = ordem_cidades[0]
            distancia_inicial = calcular_distancia(initial_pos, locacoes_cidades[primeira_cidade])

            # Decolando do ponto inicial
            energia_decolagem, tempo_decolagem = calcular_decolagem(massa = peso_drone, aceleracao=aceleracao[i], altura_voo=altura_decolagem, altura_cidade=initial_pos[2])
            consumo_energia += energia_decolagem
            tempo += tempo_decolagem
            # Pousando na primeira cidade
            energia_pouso, tempo_pouso = calcular_pouso(massa = peso_drone, aceleracao=aceleracao[i], altura_voo=altura_decolagem, altura_cidade=locacoes_cidades[primeira_cidade][2])
            consumo_energia += energia_pouso
            tempo += tempo_pouso
            # Deslocando entre as cidades
            energia_deslocamento, tempo_deslocamento = calcular_deslocamento(massa = peso_drone, aceleracao=aceleracao[i], distancia=distancia_inicial)
            consumo_energia += energia_deslocamento
            tempo += tempo_deslocamento
            
            for k in range(1, len(ordem_cidades)):
                cidade_anterior = ordem_cidades[k - 1]
                cidade_atual = ordem_cidades[k]
                distancia = calcular_distancia(locacoes_cidades[cidade_anterior], locacoes_cidades[cidade_atual]) 

                # Decolando da cidadade anterior
                energia_decolagem, tempo_decolagem = calcular_decolagem(massa = peso_drone, aceleracao=aceleracao[i], altura_voo=altura_decolagem, altura_cidade=locacoes_cidades[cidade_anterior][2])
                consumo_energia += energia_decolagem
                tempo += tempo_decolagem
                # Pousando na cidade atual
                energia_pouso, tempo_pouso = calcular_pouso(massa = peso_drone, aceleracao=aceleracao[i], altura_voo=altura_decolagem, altura_cidade=locacoes_cidades[cidade_atual][2])
                consumo_energia += energia_pouso
                tempo += tempo_pouso
                # Deslocando entre as cidades
                energia_deslocamento, tempo_deslocamento = calcular_deslocamento(massa = peso_drone, aceleracao=aceleracao[i], distancia=distancia)
                consumo_energia += energia_deslocamento
                tempo += tempo_deslocamento
            
            # TODO: RETORNAR PARA LOCAL INICIAL

            tempo_total.append(tempo) # Armazena o tempo da solução
            consumo_energia_total.append(consumo_energia) # Armazena o consumo da solução
            constraints = 0
            if sorted(set(ordem_cidades)) != sorted(set(nomes_cidades)):
                print("Cidades faltando")
                constraints += 1
            constraints_violadas.append(constraints)

        out["F"] = np.column_stack([tempo_total, consumo_energia_total])
        out["G"] = np.column_stack([constraints_violadas])

# Configurar o algoritmo NSGA-II
algorithm = NSGA2(
    pop_size=50, # tamanho de soluções em cada geraçao
    eliminate_duplicates=True,
    sampling=CustomPermutationRandomSampling(),
    crossover=PMXCrossover(prob=0.9),
    mutation=PM(eta=20, prob=0.0, vtype=int, repair=RoundingRepair())
)

# Resolver o problema
problem = DroneOptimizationProblem()
res = minimize(problem, algorithm, ('n_gen', 50), verbose=True) # Quantidade de gerações que o alg roda
# res.X array onde cada linha é uma solução, e cada coluna é uma variável de decisão, a ultima coluna sendo a aceleração
# res.F array onde cada linha é uma solução, e cada coluna é um objetivo, a primeira coluna sendo o tempo e a segunda a energia

# Exibir os resultados
print("Soluções:")
for i in range(len(res.X)):
    print("Solução ", i, ":")
    ordem_indices = res.X[i, :-1] # descodifica os valores encontrados na solução nos indices das cidades
    ordem_cidades = [nomes_cidades[j] for j in ordem_indices] # mapeia as cidades de acordo com os indices
    tempo = res.F[i, 0]
    energia = res.F[i, 1]
    aceleracao = res.X[i, -1]/100
    print(f"Ordem das cidades: {ordem_cidades}, Aceleração: {aceleracao:.2f}, Tempo: {tempo:.2f}, Energia: {energia:.2f}")
    print(f"Constraints violadas: {res.G[i]}")
    #print(res.X[i]) 

def plot(res):
    tempo = res.F[:, 0]
    energia = res.F[:, 1]
    aceleracao = res.X[:, -1]/100

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tempo, energia, c=aceleracao, cmap='viridis', edgecolor='red')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Aceleração')

    plt.xlabel('Tempo')
    plt.ylabel('Energia')
    plt.title('Tempo x Energia x Aceleração')

    plt.show()

plot(res)

def plot_route(solution, locacoes_cidades, initial_pos):
    ordem_indices = solution[:-1]  # Exclui a última coluna (aceleração)
    ordem_cidades = [nomes_cidades[j] for j in ordem_indices]

    # Coordenadas das cidades
    x_coords = [initial_pos[0]] + [locacoes_cidades[cidade][0] for cidade in ordem_cidades]
    y_coords = [initial_pos[1]] + [locacoes_cidades[cidade][1] for cidade in ordem_cidades]

    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')
    
    # Adiciona rótulos para cada cidade
    for i, cidade in enumerate(['Inicial'] + ordem_cidades):
        plt.text(x_coords[i], y_coords[i], cidade, fontsize=12, ha='right')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rota do Drone')
    plt.grid(True)
    plt.show()

#plot_route(res.X[0], locacoes_cidades, initial_pos)



# Equacionar decolagem horizontal e vertical
# Variaveis discretas 