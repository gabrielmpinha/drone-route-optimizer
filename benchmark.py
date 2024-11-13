import numpy as np
import matplotlib.pyplot as plt

def gerar_cidades_circulares(centro, raio, num_cidades):
    angulos = np.linspace(0, 2 * np.pi, num_cidades, endpoint=False)
    coordenadas = [(centro[0] + raio * np.cos(angulo), centro[1] + raio * np.sin(angulo)) for angulo in angulos]
    return coordenadas

def gerar_pesos():
    num_pacotes = np.random.randint(1, 5)  # entre 1 e 4 pacotes
    pesos = [round(np.random.uniform(0.5, 3.0), 1) for _ in range(num_pacotes)]
    return pesos

def gerar_locacoes_cidades_circular(centro, raio, num_cidades):
    cidades_coordenadas = gerar_cidades_circulares(centro, raio, num_cidades)
    locacoes_cidades_x = {}
    for i, (x, y) in enumerate(cidades_coordenadas, 1):
        nome_cidade = f"Cidade{i}"
        pacotes = gerar_pesos()
        locacoes_cidades_x[nome_cidade] = (x, y, 0, [])
    return locacoes_cidades_x

