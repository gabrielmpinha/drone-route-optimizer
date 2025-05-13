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
                    parent1, parent2 = X[0, p, :-3], X[1, p, :-3]
                    vel_hor1, vel_hor2 = X[0, p, -1], X[1, p, -1]
                    vel_desc1, vel_desc2 = X[0, p, -2], X[1, p, -2]
                    vel_asc1, vel_asc2 = X[0, p, -3], X[1, p, -3]
                    child1, child2 = self.pmx(parent1, parent2)
                    Y[0, p, :-3], Y[1, p, :-3] = child1, child2
                    if np.random.random() <= 0.5:
                       Y[0, p, -1], Y[1, p, -1] = self.mutate_acceleration(vel_hor2, problem.xl[-1], problem.xu[-1]), self.mutate_acceleration(vel_hor1, problem.xl[-1], problem.xu[-1])
                       Y[0, p, -2], Y[1, p, -2] = self.mutate_acceleration(vel_desc2, problem.xl[-2], problem.xu[-2]), self.mutate_acceleration(vel_desc1, problem.xl[-2], problem.xu[-2])
                       Y[0, p, -3], Y[1, p, -3] = self.mutate_acceleration(vel_asc2, problem.xl[-3], problem.xu[-3]), self.mutate_acceleration(vel_asc1, problem.xl[-3], problem.xu[-3])
                    else:
                        Y[0, p, -1], Y[1, p, -1] = vel_hor1, vel_hor2
                        Y[0, p, -2], Y[1, p, -2] = vel_desc1, vel_desc2
                        Y[0, p, -3], Y[1, p, -3] = vel_asc1, vel_asc2
      
                else:
                    Y[0, p, :], Y[1, p, :] = X[0, p, :], X[1, p, :]
                    Y[0, p, -1], Y[1, p, -1] = X[0, p, -1], X[1, p, -1]
                    Y[0, p, -2], Y[1, p, -2] = X[0, p, -2], X[1, p, -2]
                    Y[0, p, -3], Y[1, p, -3] = X[0, p, -3], X[1, p, -3]

        return Y
       
    def pmx(self, parent1, parent2):
        size = len(parent1)
        cx_point1, cx_point2 = sorted(np.random.choice(range(size), 2, replace=False)) # pontos de cross
        child1, child2 = np.full(size, -3), np.full(size, -3) # inicializa os filhos

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
        n_speeds = 3
        n_cities = problem.n_var - n_speeds

        samples = np.full((n_samples, problem.n_var), int)
        
        for i in range(n_samples):
            perm = np.random.permutation(n_cities) 

            speeds = [
                np.random.randint(problem.xl[n_cities + j],
                                  problem.xu[n_cities + j])
                for j in range(n_speeds)
            ]

            samples[i, :n_cities] = perm
            samples[i, n_cities:] = speeds

        return samples

class SwapMutation(Mutation):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            X[i] = self.swap_mutation(X[i], self.prob)
            X[i] = self.acceleration_mutation(X[i], self.prob, problem.xl[-1], problem.xu[-1], -1)
            X[i] = self.acceleration_mutation(X[i], self.prob, problem.xl[-2], problem.xu[-2], -2)
            X[i] = self.acceleration_mutation(X[i], self.prob, problem.xl[-3], problem.xu[-3], -3)
        return X

    def swap_mutation(self, individual, prob):
        size = len(individual) - 3
        if size < 2:
            return individual

        for idx1 in range(size):
            if np.random.random() < prob:
                idx2 = np.random.randint(0, size)
                while idx1 == idx2:
                    idx2 = np.random.randint(0, size)
                
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        
        return individual

    def acceleration_mutation(self, individual, prob, min, max, pos):
        if np.random.random() < prob:
            while True:
                factor = np.random.uniform(0.6, 1.4)
                new_acceleration = int(individual[pos] * factor)
                if min <= new_acceleration <= max:
                    individual[pos] = new_acceleration
                    break
        
        return individual
