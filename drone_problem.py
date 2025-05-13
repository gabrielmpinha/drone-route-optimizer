from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import numpy as np
from algorithms import PMXCrossover, CustomPermutationRandomSampling, SwapMutation
from plot_function import plot_results
from drone_calculations import calcular_decolagem, calcular_pouso, calcular_deslocamento, calcular_distancia

class Pacote:
    def __init__(self, nome, x, y, z, peso):
        self.nome = nome
        self.x = x
        self.y = y
        self.z = z
        self.peso = peso

    def __repr__(self):
        return f"Pacote(Nome={self.nome}, X={self.x}, Y={self.y}, Z={self.z}, Peso={self.peso})"


def drone_optimization(locacoes_cidades, initial_pos):
    """
    Função para otimizar a rota do drone.

    Parâmetros:
        locacoes_cidades (list): Lista de objetos Pacote representando as cidades e seus pacotes.
        initial_pos (tuple): Posição inicial do drone (x, y, z).

    Retorna:
        tuple: Resultado do plot_results e a lista de soluções.
    """
    nomes_cidades = [pacote.nome for pacote in locacoes_cidades]
    altura_decolagem = 80  # Altura que o drone vai subir na transição
    peso_drone = 9.2
    peso_pacotes = sum([sum(pacote.peso) for pacote in locacoes_cidades])
    massa_inicial = peso_drone + peso_pacotes  # Massa inicial do drone
    
    vel_asc = 600 # vel maxima subida
    vel_desc = 500 # vel maxima descida
    vel_desloc = 2300 # vel max horizontal

    class DroneOptimizationProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=len(nomes_cidades) + 3,  # Número de variáveis: [cidade, velocidades]
                n_obj=2,  # Número de objetivos: [tempo de entrega, consumo de energia]
                xl=[0] * len(nomes_cidades) + [50] + [50] + [100],  # Limites inferiores: cidades, vel minima subida, descida, desloca
                xu=[len(nomes_cidades) - 1] * len(nomes_cidades) + [vel_asc] + [vel_desc] + [vel_desloc],  # Limites superiores: cidades, vel max subida, descida, desloca
                vtype=int,
                n_ieq_constr=1
            )

        def _evaluate(self, X, out, *args, **kwargs):
            v_subida   = X[:, -3] / 100
            v_descida  = X[:, -2] / 100
            v_horiz    = X[:, -1] / 100
            tempo_total = []
            consumo_energia_total = []
            constraints_violadas = []

            for i in range(X.shape[0]):
                ordem_indices = X[i, :-3]
                ordem_cidades = [locacoes_cidades[j] for j in ordem_indices]

                tempo = 0
                consumo_energia = 0
                massa_atual = massa_inicial
                primeira_cidade = ordem_cidades[0]
                distancia_inicial = calcular_distancia(initial_pos, (primeira_cidade.x, primeira_cidade.y, primeira_cidade.z))

                # Decolando do ponto inicial
                energia_decolagem, tempo_decolagem = calcular_decolagem(
                    massa=massa_atual, velocidade=v_subida[i], altura_voo=altura_decolagem, altura_cidade=initial_pos[2]
                )
                consumo_energia += energia_decolagem
                tempo += tempo_decolagem

                # Pousando na primeira cidade
                energia_pouso, tempo_pouso = calcular_pouso(
                    massa=massa_atual, velocidade=v_descida[i], altura_voo=altura_decolagem, altura_cidade=primeira_cidade.z
                )
                consumo_energia += energia_pouso
                tempo += tempo_pouso

                # Deslocando entre as cidades
                energia_deslocamento, tempo_deslocamento = calcular_deslocamento(
                    massa=massa_atual, velocidade=v_horiz[i], distancia=distancia_inicial, altura_decolagem=altura_decolagem
                )
                consumo_energia += energia_deslocamento
                tempo += tempo_deslocamento

                # Entrega dos pacotes
                massa_atual -= sum(primeira_cidade.peso)

                for k in range(1, len(ordem_cidades)):
                    cidade_anterior = ordem_cidades[k - 1]
                    cidade_atual = ordem_cidades[k]
                    distancia = calcular_distancia(
                        (cidade_anterior.x, cidade_anterior.y, cidade_anterior.z),
                        (cidade_atual.x, cidade_atual.y, cidade_atual.z)
                    )

                    # Decolando da cidade anterior
                    energia_decolagem, tempo_decolagem = calcular_decolagem(
                        massa=massa_atual, velocidade=v_subida[i], altura_voo=altura_decolagem, altura_cidade=cidade_anterior.z
                    )
                    consumo_energia += energia_decolagem
                    tempo += tempo_decolagem

                    # Pousando na cidade atual
                    energia_pouso, tempo_pouso = calcular_pouso(
                        massa=massa_atual, velocidade=v_descida[i], altura_voo=altura_decolagem, altura_cidade=cidade_atual.z
                    )
                    consumo_energia += energia_pouso
                    tempo += tempo_pouso

                    # Deslocando entre as cidades
                    energia_deslocamento, tempo_deslocamento = calcular_deslocamento(
                        massa=massa_atual, velocidade=v_horiz[i], distancia=distancia, altura_decolagem=altura_decolagem
                    )
                    consumo_energia += energia_deslocamento
                    tempo += tempo_deslocamento

                    # Entregando pacotes na cidade atual
                    massa_atual -= sum(cidade_atual.peso)

                # Retornar para o local inicial
                cidade_final = cidade_atual
                distancia_retorno = calcular_distancia(
                    (cidade_final.x, cidade_final.y, cidade_final.z), initial_pos
                )

                # Decolando da última cidade
                energia_decolagem, tempo_decolagem = calcular_decolagem(
                    massa=massa_atual, velocidade=v_subida[i], altura_voo=altura_decolagem, altura_cidade=cidade_final.z
                )
                consumo_energia += energia_decolagem
                tempo += tempo_decolagem

                # Pousando no local inicial
                energia_pouso, tempo_pouso = calcular_pouso(
                    massa=massa_atual, velocidade=v_descida[i], altura_voo=altura_decolagem, altura_cidade=initial_pos[2]
                )
                consumo_energia += energia_pouso
                tempo += tempo_pouso

                # Deslocando de volta ao local inicial
                energia_deslocamento, tempo_deslocamento = calcular_deslocamento(
                    massa=massa_atual, velocidade=v_horiz[i], distancia=distancia_retorno, altura_decolagem=altura_decolagem
                )
                consumo_energia += energia_deslocamento
                tempo += tempo_deslocamento

                tempo_total.append(tempo)
                consumo_energia_total.append(consumo_energia)
                constraints = 0
                nomes_ordem_cidades = [cidade.nome for cidade in ordem_cidades]
                nomes_locacoes_cidades = [cidade.nome for cidade in locacoes_cidades]
                if sorted(nomes_ordem_cidades) != sorted(nomes_locacoes_cidades):
                    constraints += 1
                constraints_violadas.append(constraints)

            out["F"] = np.column_stack([tempo_total, consumo_energia_total])
            out["G"] = np.column_stack([constraints_violadas])

    # Configurar o algoritmo NSGA-II
    algorithm = NSGA2(
        pop_size=100,
        eliminate_duplicates=True,
        sampling=CustomPermutationRandomSampling(),
        crossover=PMXCrossover(prob=0.9),
        mutation=SwapMutation(prob=(1 / (len(nomes_cidades)+3))),
    )

    # Resolver o problema
    problem = DroneOptimizationProblem()
    res = minimize(problem, algorithm, ("n_gen", 100), verbose=True)

    # Processar os resultados
    solutions = []
    for i in range(len(res.X)):
        ordem_indices = res.X[i, :-3]
        ordem_cidades = [nomes_cidades[j] for j in ordem_indices]
        tempo = res.F[i, 0]
        energia = res.F[i, 1]
        velocidade = res.X[i, -1]/100 # Converter para m/s
        velocidade_subida = res.X[i, -3]/100
        velocidade_descida = res.X[i, -2]/100
        constraints_violadas = res.G[i]

        # Formatar a solução como uma string legível
        solution_str = (
            f"Solução {i + 1}:\n"
            f"  Ordem das cidades: {', '.join(ordem_cidades)}\n"
            f"  Velocidade: {velocidade:.2f} m/s\n"
            f"  Velocidade de subida: {velocidade_subida:.2f} m/s\n"
            f"  Velocidade de descida: {velocidade_descida:.2f} m/s\n"
            f"  Tempo: {tempo:.2f} segundos\n"
            f"  Energia: {energia:.2f} Wh\n"
            f"  Restrições violadas: {constraints_violadas}\n"
        )
        solutions.append(solution_str)

    # Combinar todas as soluções em uma única string
    solutions = "\n".join(solutions)
    print(solutions)

    # Gerar o gráfico
    plot_result = plot_results(res, locacoes_cidades, initial_pos, nomes_cidades)

    return plot_result, solutions