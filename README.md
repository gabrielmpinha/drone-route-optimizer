# 🛸 Drone Delivery Optimization – Multiobjective Route Planning

Projeto de monografia que desenvolve e avalia uma **abordagem de otimização multiobjetivo** para roteamento de drones de entrega, minimizando **tempo total de missão** e **consumo energético**, usando o algoritmo NSGA-II implementado com a biblioteca **Pymoo** (Python).

## ✏️ Resumo do projeto

Este projeto apresenta uma solução para o problema de **planejamento de rotas de entrega com drones**, considerando:
- Restrições físicas reais do drone (peso máximo, velocidades de voo, aceleração, etc.)
- Modelagem de consumo energético a partir de conceitos de física clássica (energia cinética, potencial e rotacional)
- **Dois objetivos conflitantes**: redução do tempo total da missão e redução do gasto de energia

A solução foi avaliada em cenários simulados, como:
- Locais com diferentes pesos de pacotes
- Distribuições circulares de pontos
- Entregas com múltiplos pacotes em um mesmo local
- Distribuições de peso decrescentes

O projeto demonstra a capacidade do algoritmo NSGA-II de gerar um **conjunto de soluções Pareto-ótimas**, permitindo ao operador escolher a rota mais adequada segundo as prioridades (mais rápida, mais econômica ou equilibrada).

## ⚙️ Como executar

```bash
git clone https://github.com/gabrielmpinha/drone-route-optimizer.git
cd drone-route-optimizer
pip install -r requirements.txt
python main.py
```

## 📦 Principais dependências

- Python 3.10+
- pymoo – framework para algoritmos evolutivos
- matplotlib – geração de gráficos
- pandas – leitura e manipulação de dados simulados
- numpy – cálculos numéricos
- tkinter ou outra lib para interface gráfica


## 📊 Resultados esperados

- Curvas de Pareto representando o compromisso entre tempo de entrega e consumo de energia
- Gráficos de rotas correspondentes às soluções selecionadas (mais rápida, mais econômica, equilibrada)
- Análise detalhada dos cenários simulados

## ✅ Objetivo acadêmico

Este projeto foi desenvolvido como parte da monografia de conclusão de curso, com o objetivo de:
- Aplicar algoritmos evolutivos (NSGA-II) em um contexto logístico realista
- Modelar o problema considerando restrições físicas de voo
- Demonstrar a utilidade da otimização multiobjetivo para tomada de decisão

## 📄 Licença

Projeto acadêmico open source, disponível sob a licença MIT.  
Sinta-se livre para estudar, adaptar e reutilizar!

> ✉️ **Contato:** Caso queira saber mais ou contribuir, fique à vontade para entrar em contato.
