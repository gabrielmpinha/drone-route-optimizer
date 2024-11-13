import numpy as np
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

# 50% de chance de trocar as posições da acelerações
class PMXCrossover(Crossover):

    def __init__(self, prob=1.0):
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        Y = np.full_like(X, 0, dtype=X.dtype)

        for p in range(n_matings):
                if np.random.random() < self.prob:
                    parent1, parent2 = X[0, p, :-1], X[1, p, :-1]
                    accel1, accel2 = X[0, p, -1], X[1, p, -1]
                    child1, child2 = self.pmx(parent1, parent2)
                    Y[0, p, :-1], Y[1, p, :-1] = child1, child2
                    if np.random.random() <= 0.5:
                       Y[0, p, -1], Y[1, p, -1] = self.mutate_acceleration(accel2, problem.xl[-1], problem.xu[-1]), self.mutate_acceleration(accel1, problem.xl[-1], problem.xu[-1])
                    else:
                        Y[0, p, -1], Y[1, p, -1] = accel1, accel2
      
                else:
                    Y[0, p, :], Y[1, p, :] = X[0, p, :], X[1, p, :]
                    Y[0, p, -1], Y[1, p, -1] = X[0, p, -1], X[1, p, -1]

        return Y
       
    def pmx(self, parent1, parent2):
        size = len(parent1)
        cx_point1, cx_point2 = sorted(np.random.choice(range(size), 2, replace=False)) # pontos de cross
        child1, child2 = np.full(size, -1), np.full(size, -1) # inicializa os filhos

        child1[cx_point1:cx_point2+1] = parent1[cx_point1:cx_point2+1]
        child2[cx_point1:cx_point2+1] = parent2[cx_point1:cx_point2+1]

        mapping1 = {parent1[i]: parent2[i] for i in range(cx_point1, cx_point2 + 1)}
        mapping2 = {parent2[i]: parent1[i] for i in range(cx_point1, cx_point2 + 1)}

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
    
    def mutate_acceleration(self, accel, min_accel, max_accel):
        while True:
            factor = np.random.uniform(0.9, 1.1)  # Pequena mutação
            new_accel = int(accel * factor)
            if min_accel <= new_accel <= max_accel:
                break
        return np.clip(new_accel, min_accel, max_accel)

class CustomPermutationRandomSampling(PermutationRandomSampling):

    def _do(self, problem, n_samples, **kwargs):
        n_var = problem.n_var - 1 
        samples = np.full((n_samples, problem.n_var), int)
        
        for i in range(n_samples):
            permutation = np.random.permutation(n_var)
            samples[i, :-1] = permutation
            samples[i, -1] = np.random.randint(problem.xl[-1], problem.xu[-1])
        
        return samples

class SwapMutation(Mutation):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            X[i] = self.swap_mutation(X[i], self.prob)
            X[i] = self.acceleration_mutation(X[i], self.prob, problem.xl[-1], problem.xu[-1])
        return X
    
    def swap_mutation(self, individual, prob):
        size = len(individual) - 1
        if size < 2:
            return individual

        for idx1 in range(size):
            if np.random.random() < prob:
                idx2 = np.random.randint(0, size)
                while idx1 == idx2:
                    idx2 = np.random.randint(0, size)
                
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        
        return individual

    def acceleration_mutation(self, individual, prob, min, max):
        if np.random.random() < prob:
            while True:
                factor = np.random.uniform(0.6, 1.4)
                new_acceleration = int(individual[-1] * factor)
                if min <= new_acceleration <= max:
                    individual[-1] = new_acceleration
                    break
        
        return individual
