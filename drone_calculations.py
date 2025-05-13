import math

def calcular_decolagem(massa, velocidade, altura_voo, altura_cidade):
    g = 9.81  # Aceleração da gravidade em m/s^2

    # Altura total
    altura_total = altura_voo - altura_cidade

    # Calcular a aceleração a partir da velocidade
    aceleracao = (velocidade**2) / (2 * altura_total)

    if aceleracao <= 0:
        raise ValueError("A aceleração resultante deve ser maior que zero.")

    # Tempo de subida MRUV (considerando aceleração constante)
    tempo_subida = velocidade / aceleracao

    momento_inercia, velocidade_angular = rot_drone()
    energia_rotacional = ((momento_inercia * (velocidade_angular ** 2))/2)

    energia_potencial  = massa * g * altura_total # potencia para elevar um drone a altura total

    # Energia cinética
    energia_cinetica = 0.5 * massa * (velocidade ** 2)
    
    # energia gasta para vencer gravidade + cinética + rotacional
    energia_gasta = energia_rotacional + energia_potencial + energia_cinetica

    return (energia_gasta / 3600), tempo_subida

def calcular_pouso(massa, velocidade, altura_voo, altura_cidade):
    g = 9.81  # Aceleração da gravidade em m/s^2

    # Altura total
    altura_total = altura_voo - altura_cidade

    # Calcular a aceleração a partir da velocidade
    aceleracao = (velocidade**2) / (2 * altura_total) 

    if aceleracao <= 0:
        raise ValueError("A aceleração resultante deve ser maior que zero.")

    # Tempo de descida MRUV (considerando aceleração constante)
    tempo_descida = velocidade / aceleracao

    momento_inercia, velocidade_angular = rot_drone()
    energia_rotacional = ((momento_inercia * (velocidade_angular ** 2)) / 2)

    energia_potencial = massa * g * altura_total # Potência para manter o drone em voo contra a gravidade
    
    # Energia cinética
    energia_cinetica = 0.5 * massa * (velocidade ** 2)
    
    # Energia gasta considerando velocidade angular constante
    energia_gasta =  energia_rotacional + energia_potencial + energia_cinetica

    return (energia_gasta / 3600), tempo_descida

def calcular_deslocamento(massa, velocidade, distancia, altura_decolagem):
    g = 9.81  # Aceleração da gravidade em m/s^2

    if distancia == 0:
        return (0,0)
    
    # Tempo de deslocamento (considerando aceleração constante)
    tempo_deslocamento = distancia / velocidade
     
     
    # energia cinetica rotacional (pás girando)
    momento_inercia, velocidade_angular = rot_drone()
    energia_rotacional = ((momento_inercia * (velocidade_angular ** 2)) / 2)
    
    energia_potencial = massa * g * altura_decolagem # Potência para manter o drone em voo contra a gravidade
    
    # Energia cinética translacional(linha reta)
    energia_cinetica = 0.5 * massa * (velocidade ** 2)
    
    # Energia gasta considerando velocidade angular constante
    energia_gasta = energia_rotacional + energia_cinetica + energia_potencial

    return (energia_gasta / 3600), tempo_deslocamento

def rot_drone(massa_pa=1, comprimento_pa=0.5, rpm=3000):
    # Cálculo do momento de inércia
    momento_inercia = (1/3) * massa_pa * (comprimento_pa ** 2)
    # Cálculo da velocidade angular
    velocidade_angular = (2 * math.pi * rpm) / 60
    return momento_inercia, velocidade_angular

# Função para calcular a distância entre duas posições, inicialmente ignorando altura
def calcular_distancia(pos1=(0,0,0), pos2=(0,0,0)):
    posx = abs(pos1[0] - pos2[0])
    posy = abs(pos1[1] - pos2[1])
    distancia = math.sqrt(posx**2 + posy**2)
    return distancia