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
        n_matings, n_parents, n_var = X.shape
        Y = np.full_like(X, 0, dtype=X.dtype)

        for p in range(n_parents):
                if np.random.random() < self.prob:
                    if n_parents > 1:
                        parent1, parent2 = X[0, p, :-1], X[1, p, :-1]
                        accel1, accel2 = X[0, p, -1], X[1, p, -1]
                        child1, child2 = self.pmx(parent1, parent2)
                        Y[0, p, :-1], Y[1, p, :-1] = child1, child2
                        if np.random.random() < 0.5:
                            Y[0, p, -1], Y[1, p, -1] = accel2, accel1
                        else:
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

# aplicar nas 51 posições(gene por gene) da solução 1/num cidades ou 2/num cidades
class SwapMutation(Mutation):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            X[i] = self.swap_mutation(X[i], self.prob)
        return X
    
    def swap_mutation(self, individual, prob):
        if np.random.random() < prob:
            size = len(individual) - 1
            if size < 2:
                return individual

            # Select two positions to swap
            idx1, idx2 = np.random.choice(size, 2, replace=False)
            
            # Swap the elements
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
            
            #individual[-1] = individual[-1]+100
        
        return individual
