import math

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
    potencia_elevacao = energia_gravitacional # potencia para elevar um drone a altura total em uma unidade de tempo

    # Energia cinética devido à aceleração
    energia_cinetica = 0.5 * massa * (aceleracao ** 2)
    
    # Energia gasta considerando velocidade angular constante
    energia_gasta = (energia_rotacional + potencia_elevacao + energia_cinetica) * tempo_subida

    return (energia_gasta), tempo_subida

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
    potencia_descida = energia_gravitacional  # Potência para descer o drone a altura total em uma unidade de tempo
    
    # Energia cinética devido à aceleração
    energia_cinetica = 0.5 * massa * (aceleracao ** 2)
    
    # Energia gasta considerando velocidade angular constante
    energia_gasta = (energia_rotacional + potencia_descida + energia_cinetica) * tempo_descida

    return (energia_gasta), tempo_descida

def calcular_deslocamento(massa, aceleracao, distancia, altura_decolagem):
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
        potencia_deslocamento = energia_gravitacional   # Potência para manter o drone em voo contra a gravidade
    else:
        potencia_deslocamento = 0
    
    # Energia cinética devido à aceleração
    energia_cinetica = 0.5 * massa * (aceleracao ** 2)
    
    # Energia gasta considerando velocidade angular constante
    energia_gasta = (energia_rotacional + potencia_deslocamento + energia_cinetica) * tempo_deslocamento

    return (energia_gasta), tempo_deslocamento

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